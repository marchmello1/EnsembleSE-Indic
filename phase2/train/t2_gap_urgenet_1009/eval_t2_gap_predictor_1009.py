"""State-balanced released-validation screen for the official GAP-URGENet."""
import json
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from models.predictor.tfgridnet import TFGridNet

parser=argparse.ArgumentParser()
parser.add_argument('--limit',type=int,choices=(24,188,400),default=24)
parser.add_argument('--natural',action='store_true')
parser.add_argument('--checkpoint',type=Path)
parser.add_argument('--prefix',default='predictor')
args=parser.parse_args()
root = Path('/mnt/nvme_disk1/t2_gap_urgenet_1009')
validation = Path('/mnt/nvme_disk1/official_validation_20260907')
anchor = Path('/mnt/nvme_disk1/t2_publicnoise_hilr_s5000_fullclip_1008')
if args.natural:
    anchor=Path('/mnt/nvme_disk1/t2_natural400_hilr_1008/candidate')
subset='nat' if args.natural else 'hold'
output = root/f'{args.prefix}_{subset}{args.limit}_wavs'
output.mkdir(exist_ok=True)
meta = {r['segmentFileName']: r for r in json.loads((validation/'validationMetadata.json').read_text())}
groups = {}
held_dir=Path('/mnt/nvme_disk1/urgent2026_t2pilot_1008')/('extended_s1600_ls70_nat400_wavs' if args.natural else 'cont_s3000_hold188_wavs')
held_names = {p.name for p in held_dir.glob('*.wav')}
assert len(held_names) == (400 if args.natural else 188)
for path in sorted(anchor.glob('*.wav')):
    if path.name not in held_names:
        continue
    groups.setdefault(meta[path.name]['state'], []).append(path)
names = []
for index in range(max(map(len, groups.values()))):
    for state in sorted(groups):
        if index < len(groups[state]) and len(names) < args.limit:
            names.append(groups[state][index])
assert len(names) == args.limit
paired_states={meta[p.name]['state'] for p in Path('/mnt/nvme_disk1/urgent2026_t2pilot_1008/cont_s3000_hold188_wavs').glob('*.wav')}
assert set(groups).issubset(paired_states)
if not args.natural:
    assert len(groups)==6
print('PILOT_STATES', {state: sum(meta[p.name]['state']==state for p in names) for state in groups}, flush=True)
ck = root/'checkpoints'
checkpoint=torch.load(args.checkpoint or ck/'Predictor.pt',map_location='cpu',weights_only=False)
model=TFGridNet(**checkpoint['cfg'])
state=checkpoint.get('generator',checkpoint.get('model'))
model.load_state_dict(state,strict=True)
model=model.cuda().eval()
def si(x,y):
    x=np.asarray(x,np.float64)-np.mean(x)
    y=np.asarray(y,np.float64)-np.mean(y)
    target=y*(np.dot(x,y)/(np.dot(y,y)+1e-12))
    return float(10*np.log10((np.sum(target**2)+1e-12)/(np.sum((x-target)**2)+1e-12)))
rows=[]
for path in names:
    noisy,sr=sf.read(validation/('naturalNoisyAudio' if args.natural else 'syntheticNoiseAudio')/path.name,dtype='float32')
    base,bs=sf.read(path,dtype='float32')
    assert sr==bs==16000
    with torch.inference_mode():
        y=model(torch.from_numpy(noisy)[None].cuda(),sr=sr).cpu().numpy().reshape(-1)
    if len(y)<len(noisy):
        y=np.pad(y,(0,len(noisy)-len(y)))
    y=y[:len(noisy)]
    assert len(y)==len(noisy)==len(base) and np.isfinite(y).all()
    # Exact official amplitude convention; no clean-reference matching.
    y=y/(np.max(np.abs(y))+1e-8)*np.max(np.abs(noisy))
    sf.write(output/path.name,y,sr,subtype='PCM_16')
    scores={}
    if not args.natural:
        clean,cs=sf.read(validation/'syntheticCleanRefAudio'/path.name)
        assert cs==sr and len(clean)==len(y)
        scores={'gap':si(y,clean),'anchor':si(base,clean)}
        left=base-base.mean();right=y-y.mean()
        gain=np.clip(np.dot(left,right)/(np.dot(right,right)+1e-12),.1,10)
        for w in [.1,.3,.7]:
            scores[str(w)]=si((1-w)*base+w*gain*y,clean)
    rows.append({'id':path.stem,'state':meta[path.name]['state'],'scores':scores})
    print('DONE',len(rows),scores,flush=True)
result={'rows':rows,'means':{k:float(np.mean([r['scores'][k] for r in rows])) for k in rows[0]['scores']}}
(root/f'{args.prefix}_{subset}{args.limit}_report.json').write_text(json.dumps(result,indent=2))
print('FINAL',result['means'],flush=True)
