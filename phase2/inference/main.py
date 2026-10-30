#!/usr/bin/env python3
"""Usage: python main.py INPUT_FOLDER OUTPUT_FOLDER."""
import os
os.environ.setdefault('USE_TF','0');os.environ.setdefault('TRANSFORMERS_NO_TF','1')
import argparse,gc,json,sys
from pathlib import Path
import numpy as np,soundfile as sf,torch,yaml
import mp_core as core
from baseline_code.d_model import SEModel
from predictor_tfgridnet import TFGridNet
from tri_neural_enhance import pcm16_roundtrip
from tta_enhance import averaged_branches,ensemble
from standalone.tflocoformer_separator import TFLocoformerSeparator
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder

def main():
    parser=argparse.ArgumentParser();parser.add_argument('input_folder',type=Path);parser.add_argument('output_folder',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parent
    asr_python=os.environ.get('TRACK2_ASR_PYTHON',sys.executable)
    if not Path(asr_python).is_absolute() or not os.access(asr_python,os.X_OK):
        raise ValueError('TRACK2_ASR_PYTHON must be an absolute executable Python path')
    if not (root/'asr_worker.py').is_file():raise FileNotFoundError('Missing asr_worker.py')
    names=sorted(args.input_folder.glob('*.wav'))
    if not names:raise ValueError('No root-level WAV input files')
    if args.input_folder.resolve()==args.output_folder.resolve():raise ValueError('Input and output must differ')
    args.output_folder.mkdir(parents=True,exist_ok=True)
    if list(args.output_folder.iterdir()):raise ValueError('Use an empty output folder')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg=core.AttrDict(json.loads((root/'vendor/best_ckpt/config.json').read_text()))
    mp=core.load_enhancer(root/'checkpoints/t2_mpdns_publicnoise_hilr_1008.pt',cfg,device).eval()
    bs=SEModel.load_from_checkpoint(str(root/'checkpoints/bsrnn_extended_s1600.ckpt'),map_location='cpu').to(device).eval()
    ck=torch.load(root/'checkpoints/predictor_d32_s800.pt',map_location='cpu',weights_only=False)
    pred=TFGridNet(**ck['cfg']).to(device).eval();pred.load_state_dict(ck['model'],strict=True)
    lc=yaml.safe_load((root/'locoformer_config.yaml').read_text())
    loco=TFLocoformerSeparator(**lc['separator_conf']).to(device).eval()
    weights=torch.load(root/'checkpoints/locoformer_d32.pth',map_location='cpu',weights_only=True)
    loco.load_state_dict({k.removeprefix('separator.'):v for k,v in weights.items()},strict=True)
    loco2=TFLocoformerSeparator(**lc['separator_conf']).to(device).eval()
    loco2.load_state_dict(torch.load(root/'checkpoints/locoformer_orig.pth',map_location='cpu',weights_only=True),strict=True)
    loco3=TFLocoformerSeparator(**lc['separator_conf']).to(device).eval()
    loco3.load_state_dict(torch.load(root/'checkpoints/locoformer_spec.pth',map_location='cpu',weights_only=True),strict=True)
    loco4=TFLocoformerSeparator(**lc['separator_conf']).to(device).eval()
    loco4.load_state_dict(torch.load(root/'checkpoints/locoformer_extra.pth',map_location='cpu',weights_only=True),strict=True)
    enc=STFTEncoder(**lc['encoder_conf']).to(device).eval();dec=STFTDecoder(**lc['decoder_conf']).to(device).eval()
    for index,path in enumerate(names,1):
        wave=core.read_audio(path)
        if wave.ndim!=1 or not len(wave) or not np.isfinite(wave).all():raise ValueError(f'Invalid audio: {path.name}')
        result=ensemble(wave,averaged_branches(wave,mp,bs,pred,(loco,loco2,loco3,loco4),enc,dec,cfg,device,core))
        assert len(result)==len(wave) and np.isfinite(result).all()
        sf.write(args.output_folder/path.name,result,16000,subtype='PCM_16')
        if index%25==0:print('ENHANCED',index,len(names),flush=True)
    del mp,bs,pred,loco,loco2,loco3,loco4,enc,dec,weights,ck
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    # A separate interpreter avoids ESPnet/NeMo's incompatible Librosa requirements.
    # exec replaces this process and releases every enhancement CUDA allocation.
    os.execv(asr_python,[asr_python,str(root/'asr_worker.py'),str(args.output_folder.resolve())])
if __name__=='__main__':main()
