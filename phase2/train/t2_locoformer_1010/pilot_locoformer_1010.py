"""Paired Locoformer screen; references are read only after inference."""
import json
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import yaml
from standalone.tflocoformer_separator import TFLocoformerSeparator
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder

root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
validation=Path('/mnt/nvme_disk1/official_validation_20260907')
parser=argparse.ArgumentParser()
parser.add_argument('--limit',type=int,choices=(24,188,400),default=24)
parser.add_argument('--natural',action='store_true')
parser.add_argument('--checkpoint',type=Path)
parser.add_argument('--tag',default='')
args=parser.parse_args()
assert (args.natural and args.limit==400) or (not args.natural and args.limit in (24,188))
output=Path('/mnt/nvme_disk1/t2_locoformer_'+('nat' if args.natural else 'fixed')+str(args.limit)+'_1010'+args.tag)
output.mkdir(exist_ok=True)
blendout=output.parent/(output.name+'_blend30')
blendout.mkdir(exist_ok=True)
selection=Path('/mnt/nvme_disk1/t2_cleanmel_pilot24_1010') if args.limit==24 else Path('/mnt/nvme_disk1/urgent2026_t2pilot_1008/extended_s1600_ls70_hold188_wavs')
if args.natural:
    selection=Path('/mnt/nvme_disk1/urgent2026_t2pilot_1008/extended_s1600_ls70_nat400_wavs')
names=sorted(p.name for p in selection.glob('*.wav'))
assert len(names)==args.limit
torch.cuda.set_per_process_memory_fraction(.35)
cfg=yaml.safe_load((root/'egs2/dns_ins20/enh1/exp/enh_train_enh_tflocoformer_raw/config.yaml').read_text())
model=TFLocoformerSeparator(**cfg['separator_conf'])
if args.checkpoint:
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)['model']
else:
    raw=torch.load(root/'dns_pretrained.pth',map_location='cpu',weights_only=True)
    state={k.removeprefix('separator.'):v for k,v in raw.items()}
model.load_state_dict(state,strict=True)
model=model.cuda().eval()
encoder=STFTEncoder(**cfg['encoder_conf']).cuda().eval()
decoder=STFTDecoder(**cfg['decoder_conf']).cuda().eval()
def enhance(noisy):
    x=torch.from_numpy(noisy)[None].cuda()
    lengths=torch.tensor([x.shape[-1]],device=x.device)
    with torch.inference_mode():
        std=x.std(1,keepdim=True).clamp_min(1e-8)
        z,flens=encoder(x/std,lengths)
        predicted=model(z)
        assert predicted.shape[1]==1
        y,_=decoder(predicted[:,0],lengths)
        y=y*std
    result=y[0].cpu().numpy()
    assert result.shape==noisy.shape and np.isfinite(result).all()
    return result/max(1,float(np.max(np.abs(result)))/.99)
def si(y,c):
    y=np.asarray(y,np.float64);c=np.asarray(c,np.float64)
    y=y-y.mean();c=c-c.mean()
    target=c*(y@c/(c@c+1e-8))
    return float(10*np.log10((target@target+1e-8)/(np.sum((y-target)**2)+1e-8)))
rows=[]
for name in names:
    x,sr=sf.read(validation/('naturalNoisyAudio' if args.natural else 'syntheticNoiseAudio')/name,dtype='float32')
    assert sr==16000 and x.ndim==1
    y=enhance(x)
    sf.write(output/name,y,sr,subtype='PCM_16')
    y,_=sf.read(output/name)
    if args.natural:
        best,bs=sf.read(Path('/mnt/nvme_disk1/t2_gap_urgenet_1009/predictor_fulladapt_s800_blend30_nat400_wavs')/name)
        assert bs==sr and best.shape==y.shape
        z=.7*best+.3*y
        sf.write(blendout/name,z,sr,subtype='PCM_16')
        rows.append({'id':Path(name).stem,'scores':{}})
        print('DONE_NATURAL',len(rows),flush=True)
        continue
    c,cs=sf.read(validation/'syntheticCleanRefAudio'/name)
    base,bs=sf.read(Path('/mnt/nvme_disk1/urgent2026_t2pilot_1008/extended_s1600_ls70_hold188_wavs')/name)
    best,ss=sf.read(Path('/mnt/nvme_disk1/t2_gap_urgenet_1009/predictor_fulladapt_s800_blend30_hold188_wavs')/name)
    assert cs==bs==ss==sr and c.shape==base.shape==best.shape==y.shape
    values={'direct':si(y,c),'incumbent':si(base,c),'best_predictor':si(best,c)}
    # Global fixed blends, not per-clip reference selection.
    for weight in (.05,.15,.3):
        z=(1-weight)*best+weight*y
        z=np.floor(np.clip(z,-1,1-1/32768)*32768)/32768
        values['blend_'+str(weight)]=si(z,c)
        if weight==.3:
            sf.write(blendout/name,z,sr,subtype='PCM_16')
    rows.append({'id':Path(name).stem,'scores':values})
    print('DONE',len(rows),values,flush=True)
result={'scope':'fixed released state-held-out'+str(args.limit)+'; not official prediction',
        'rows':rows,'means':{k:float(np.mean([r['scores'][k] for r in rows])) for k in rows[0]['scores']},
        'reference_used_inside_enhancement':False,'checkpoint':str(args.checkpoint) if args.checkpoint else 'merlresearch/tf-locoformer DNS-Interspeech2020',
        'license':'Apache-2.0 source; preserve notices/check checkpoint terms before packaging'}
(output/'result.json').write_text(json.dumps(result,indent=2))
print('FINAL',result['means'],flush=True)

