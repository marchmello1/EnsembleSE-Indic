"""Audio-only, fixed translation ensemble of the deployed neural enhancer."""
import argparse,json,sys
import random
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from render_phase2_dual_neural import _load_mp_core

p=argparse.ArgumentParser()
p.add_argument('cache',type=Path); p.add_argument('output',type=Path)
p.add_argument('--count',type=int,default=24)
p.add_argument('--natural-validation',type=Path)
a=p.parse_args()
package=Path('/mnt/nvme_disk1/t2_bsrnn_s3000_codepkg_1008/marchmello01')
core=_load_mp_core(package)
sys.path.insert(0,'/mnt/nvme_disk1/urgent2026_t2pilot_1008')
from baseline_code.d_model import SEModel
device=torch.device('cuda')
cfg=core.AttrDict(json.loads((core.MP_REPO/'best_ckpt/config.json').read_text()))
mp=core.load_enhancer(package/'checkpoints/t2_mpdns_publicnoise_hilr_1008.pt',cfg,device)
bs=SEModel.load_from_checkpoint('/mnt/nvme_disk1/urgent2026_t2pilot_1008/bsrnn_ls70_rnnt_extended_1009_s1600.ckpt',map_location='cpu').to(device).eval()
def enhance(wave):
    primary=core.enhance_full_clip(wave,mp,cfg,device)
    x=torch.as_tensor(wave,device=device)[None]
    with torch.inference_mode():
        out,_=bs.se_model(x,torch.tensor([len(wave)],device=device),16000)
    secondary=out.cpu().numpy().reshape(-1)
    secondary=secondary*(.9/max(.9,float(np.max(np.abs(secondary)))))
    secondary=np.pad(secondary,(0,max(0,len(wave)-len(secondary))))[:len(wave)]
    primary=primary[:len(wave)]
    left=primary.astype(np.float64)-primary.mean(); right=secondary.astype(np.float64)-secondary.mean()
    gain=np.clip(np.dot(left,right)/max(np.dot(right,right),1e-12),.1,10)
    y=.3*primary+.7*gain*secondary
    return y/max(1,float(np.max(np.abs(y)))/.99)
def score(y,clean):
    y=np.asarray(y,np.float64); clean=np.asarray(clean,np.float64)
    y=y-y.mean(); clean=clean-clean.mean()
    target=clean*np.dot(y,clean)/max(np.dot(clean,clean),1e-12)
    return float(10*np.log10((np.dot(target,target)+1e-12)/(np.sum((y-target)**2)+1e-12)))
rows=sorted(torch.load(a.cache,map_location='cpu',weights_only=False),key=lambda r:r['id'])
if a.natural_validation:
    metadata=json.loads((a.natural_validation/'validationMetadata.json').read_text())
    states={'AndhraPradesh','Chhattisgarh','Karnataka','Meghalaya','Sikkim','WestBengal'}
    selected=[r for r in metadata if not r.get('syntheticData') and r['state'] in states]
    selected=random.Random(1008).sample(selected,min(a.count,len(selected)))
    rows=[]
    for item in selected:
        wave,sr=sf.read(a.natural_validation/'naturalNoisyAudio'/item['segmentFileName'],dtype='float32')
        if sr!=16000 or wave.ndim!=1: raise ValueError('unexpected input format')
        rows.append({'id':item['segmentFileName'].removesuffix('.wav'),'waveform':wave})
elif a.count<len(rows): rows=[rows[i] for i in np.linspace(0,len(rows)-1,a.count,dtype=int)]
a.output.mkdir(parents=True,exist_ok=True)
results=[]
for row in rows:
    wave=np.asarray(row['waveform'],np.float32).reshape(-1)
    outputs=[]
    for shift in [0,64,128,192]:
        padded=np.pad(wave,(shift,0))
        outputs.append(enhance(padded)[shift:shift+len(wave)])
    y=np.mean(outputs,axis=0)
    sf.write(a.output/(row['id']+'.wav'),y,16000,subtype='PCM_16')
    pcm,_=sf.read(a.output/(row['id']+'.wav'))
    baseline=np.clip(outputs[0],-1,1)
    # Compare both in PCM16 to avoid precision advantage.
    sf.write(a.output/(row['id']+'_control.wav'),baseline,16000,subtype='PCM_16')
    control,_=sf.read(a.output/(row['id']+'_control.wav'))
    result={'id':row['id']}
    if 'clean' in row:
        result.update(candidate=score(pcm,row['clean']),baseline=score(control,row['clean']))
    results.append(result)
    print('DONE',len(results),result,flush=True)
report={'scope':'released heldout fixed-shift pilot, PCM SI only','count':len(results),
        'clips':results}
if not a.natural_validation:
    report.update(candidate=float(np.mean([r['candidate'] for r in results])),
                  baseline=float(np.mean([r['baseline'] for r in results])))
    report['delta']=report['candidate']-report['baseline']
(a.output/'result.json').write_text(json.dumps(report,indent=2))
print('RESULT',{k:v for k,v in report.items() if k!='clips'},flush=True)
