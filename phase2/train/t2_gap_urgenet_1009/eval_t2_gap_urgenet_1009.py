"""State-balanced released-validation screen for the official GAP-URGENet."""
import json
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from models.gap_urgenet import GAP_URGENet

parser=argparse.ArgumentParser()
parser.add_argument('--limit',type=int,choices=(24,188),default=24)
args=parser.parse_args()
root = Path('/mnt/nvme_disk1/t2_gap_urgenet_1009')
validation = Path('/mnt/nvme_disk1/official_validation_20260907')
anchor = Path('/mnt/nvme_disk1/t2_publicnoise_hilr_s5000_fullclip_1008')
output = root/f'hold{args.limit}_wavs'
output.mkdir(exist_ok=True)
meta = {r['segmentFileName']: r for r in json.loads((validation/'validationMetadata.json').read_text())}
groups = {}
held_names = {p.name for p in Path('/mnt/nvme_disk1/urgent2026_t2pilot_1008/cont_s3000_hold188_wavs').glob('*.wav')}
assert len(held_names) == 188
for path in sorted(anchor.glob('*.wav')):
    if path.name not in held_names:
        continue
    groups.setdefault(meta[path.name]['state'], []).append(path)
names = []
for index in range(max(map(len, groups.values()))):
    for state in sorted(groups):
        if index < len(groups[state]) and len(names) < args.limit:
            names.append(groups[state][index])
assert len(names) == args.limit and len(groups) == 6
print('PILOT_STATES', {state: sum(meta[p.name]['state']==state for p in names) for state in groups}, flush=True)
ck = root/'checkpoints'
model = GAP_URGENet(**{f'{name}_ckpt_path': str(ck/file) for name, file in
    [('predictor','Predictor.pt'), ('dewavlm','DeWavLM-Omni.pt'),
     ('adapter','Adapter.pt'), ('vocoder','Vocoder.pt'), ('postnet','PostNet.pt')]}).cuda().eval()
def si(x,y):
    x=np.asarray(x,np.float64)-np.mean(x)
    y=np.asarray(y,np.float64)-np.mean(y)
    target=y*(np.dot(x,y)/(np.dot(y,y)+1e-12))
    return float(10*np.log10((np.sum(target**2)+1e-12)/(np.sum((x-target)**2)+1e-12)))
rows=[]
for path in names:
    noisy,sr=sf.read(validation/'syntheticNoiseAudio'/path.name,dtype='float32')
    base,bs=sf.read(path,dtype='float32')
    assert sr==bs==16000
    with torch.inference_mode():
        y=model(torch.from_numpy(noisy)[None].cuda(),sr_in=sr,sr_out=sr,enable_plc=False).cpu().numpy().reshape(-1)
    assert len(y)==len(noisy)==len(base) and np.isfinite(y).all()
    # Exact official amplitude convention; no clean-reference matching.
    y=y/(np.max(np.abs(y))+1e-8)*np.max(np.abs(noisy))
    sf.write(output/path.name,y,sr,subtype='PCM_16')
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
(root/f'hold{args.limit}_report.json').write_text(json.dumps(result,indent=2))
print('FINAL',result['means'],flush=True)
