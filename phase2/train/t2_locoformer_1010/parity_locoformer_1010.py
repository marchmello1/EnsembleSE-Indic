"""Reproduce frozen four-network outputs from audio, compare cached pilot PCM."""
import json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch,yaml
package=Path('/mnt/nvme_disk1/t2_tri_neural_rnnt800_codepkg_1009/marchmello01')
sys.path.insert(0,str(package))
import mp_core as core
from baseline_code.d_model import SEModel
from predictor_tfgridnet import TFGridNet
from tri_neural_enhance import enhance,pcm16_roundtrip
from standalone.tflocoformer_separator import TFLocoformerSeparator
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder
root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
val=Path('/mnt/nvme_disk1/official_validation_20260907')
torch.cuda.set_per_process_memory_fraction(.4)
device=torch.device('cuda')
cfg=core.AttrDict(json.loads((package/'vendor/best_ckpt/config.json').read_text()))
mp=core.load_enhancer(package/'checkpoints/t2_mpdns_publicnoise_hilr_1008.pt',cfg,device).eval()
bs=SEModel.load_from_checkpoint('/mnt/nvme_disk1/urgent2026_t2pilot_1008/bsrnn_ls70_rnnt_extended_1009_s1600.ckpt',map_location='cpu').cuda().eval()
ck=torch.load('/mnt/nvme_disk1/t2_gap_urgenet_1009/predictor_fulladapt_1010_s800.pt',map_location='cpu',weights_only=False)
pred=TFGridNet(**ck['cfg']).cuda().eval();pred.load_state_dict(ck['model'],strict=True)
lc=yaml.safe_load((root/'egs2/dns_ins20/enh1/exp/enh_train_enh_tflocoformer_raw/config.yaml').read_text())
loco=TFLocoformerSeparator(**lc['separator_conf']).cuda().eval()
weights=torch.load(root/'dns_pretrained.pth',map_location='cpu',weights_only=True)
loco.load_state_dict({k.removeprefix('separator.'):v for k,v in weights.items()},strict=True)
enc=STFTEncoder(**lc['encoder_conf']).cuda().eval();dec=STFTDecoder(**lc['decoder_conf']).cuda().eval()
meta={r['segmentFileName']:r for r in json.loads((val/'validationMetadata.json').read_text())}
reports=[]
for natural in (False,True):
    cache=Path('/mnt/nvme_disk1')/('t2_locoformer_nat400_1010_blend30' if natural else 't2_locoformer_fixed188_1010_blend30')
    chosen={}
    for p in sorted(cache.glob('*.wav')):chosen.setdefault(meta[p.name]['state'],p)
    assert len(chosen)==(4 if natural else 6)
    for state,p in chosen.items():
        x,sr=sf.read(val/('naturalNoisyAudio' if natural else 'syntheticNoiseAudio')/p.name,dtype='float32')
        assert sr==16000 and x.ndim==1
        best=pcm16_roundtrip(enhance(x,mp,bs,pred,cfg,device,core))
        t=torch.from_numpy(x)[None].cuda();lengths=torch.tensor([len(x)],device=device)
        with torch.inference_mode():
            std=t.std(1,keepdim=True).clamp_min(1e-8)
            z,_=enc(t/std,lengths);z=loco(z)[:,0]
            y,_=dec(z,lengths);y=(y*std)[0].cpu().numpy()
        y/=max(1,float(np.max(np.abs(y)))/.99)
        y=pcm16_roundtrip(y)
        actual=pcm16_roundtrip(.7*best.astype(np.float64)+.3*y.astype(np.float64))
        expected,es=sf.read(p,dtype='float32')
        assert es==sr and actual.shape==expected.shape==x.shape
        difference=np.abs(actual-expected)
        reports.append({'clip':p.stem,'state':state,'natural':natural,'bit_equal':bool(np.array_equal(actual,expected)),'max_abs':float(difference.max()),'changed_samples':int((difference>0).sum())})
        print('PARITY',reports[-1],flush=True)
report={'scope':'10 released state-held clips (6 synthetic states, 4 natural states), full audio-only four-network reproduction','rows':reports,'all_bit_equal':all(r['bit_equal'] for r in reports),'maximum_pcm_difference':max(r['max_abs'] for r in reports)}
Path('/mnt/nvme_disk1/t2_locoformer_fullchain_parity_1010.json').write_text(json.dumps(report,indent=2))
print('FINAL',report['all_bit_equal'],report['maximum_pcm_difference'],flush=True)
