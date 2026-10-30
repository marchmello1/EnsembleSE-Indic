"""Frozen final-checkpoint paired evaluation; references only after inference."""
import argparse,json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from loco_spectral_fusion_1010 import SpectralFusion

p=argparse.ArgumentParser();p.add_argument('--natural',action='store_true')
p.add_argument('--residual-scale',type=float,default=1.)
p.add_argument('--tag',default='loco_spectral1200')
a=p.parse_args();assert 0<=a.residual_scale<=1
base=Path('/mnt/nvme_disk1');root=base/'t2_locoformer_1010'
ck=torch.load(root/'loco_spectral_fusion_1010_s1200.pt',map_location='cpu',weights_only=False)
metadata={r['segmentFileName'].removesuffix('.wav'):r for r in json.loads((base/'official_validation_20260907/validationMetadata.json').read_text())}
torch.cuda.set_per_process_memory_fraction(.25)
model=SpectralFusion().cuda().eval();model.load_state_dict(ck['model'],strict=True)
count=400 if a.natural else 188
suffix='nat400' if a.natural else 'hold188'
three=base/'t2_gap_urgenet_1009'/f'predictor_fulladapt_s800_blend30_{suffix}_wavs'
stem='nat400' if a.natural else 'fixed188'
frozen=base/f't2_locoformer_{stem}_1010'
adapted=base/f't2_locoformer_{stem}_1010_acoustic1600'
output=base/f't2_{a.tag}_{suffix}_1010';output.mkdir(exist_ok=True)
names=sorted(x.name for x in three.glob('*.wav'));assert len(names)==count
assert not set(ck['training_ids'])&{Path(n).stem for n in names}
assert not {metadata[Path(n).stem]['state'] for n in names}&{metadata[i]['state'] for i in ck['training_ids']}
rows=[]
for name in names:
    noisy,sr=sf.read(base/'official_validation_20260907'/('naturalNoisyAudio' if a.natural else 'syntheticNoiseAudio')/name,dtype='float32')
    estimates=[]
    for directory in (three,frozen,adapted):
        audio,rate=sf.read(directory/name,dtype='float32');assert rate==sr==16000 and audio.shape==noisy.shape
        estimates.append(audio)
    with torch.inference_mode():
        y,w=model(torch.from_numpy(noisy)[None].cuda(),torch.from_numpy(np.stack(estimates))[None].cuda())
        frozen_audio=torch.from_numpy(.7*estimates[0]+.3*estimates[1])[None].cuda()
        y=a.residual_scale*y+(1-a.residual_scale)*frozen_audio
    audio=y[0].cpu().numpy();assert audio.shape==noisy.shape and np.isfinite(audio).all()
    sf.write(output/name,audio,sr,subtype='PCM_16')
    row={'id':Path(name).stem,'mean_weights':w.mean((0,2,3)).cpu().tolist()}
    if not a.natural:
        clean,rate=sf.read(base/'official_validation_20260907/syntheticCleanRefAudio'/name)
        audio,rate2=sf.read(output/name);assert rate==rate2==sr and clean.shape==audio.shape
        clean=clean-clean.mean();audio=audio-audio.mean();target=clean*(audio@clean/(clean@clean+1e-8))
        row['si']=float(10*np.log10((target@target+1e-8)/(np.sum((audio-target)**2)+1e-8)))
    rows.append(row);print('DONE',len(rows),flush=True)
result={'scope':'Released state-held-out evaluation, not official estimate','count':len(rows),'rows':rows,'references_inside_enhancer':False}
if not a.natural:result['si']=float(np.mean([r['si'] for r in rows]))
(root/f'{a.tag}_{suffix}_result.json').write_text(json.dumps(result,indent=2));print('COMPLETE',result.get('si'),flush=True)
