"""Full audio-only pipeline reproduction, not cached-branch inference."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--natural',action='store_true');a=p.parse_args()
root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
source=(root/'parity_locoformer_1010.py').read_text().split('meta={')[0]
exec(compile(source,str(root/'parity_locoformer_1010.py'),'exec'))
base=Path('/mnt/nvme_disk1')
suffix='nat400' if a.natural else 'hold188'
cache=base/('t2_locoformer_nat400_1010_blend30' if a.natural else 't2_locoformer_fixed188_1010_blend30')
output=base/f't2_four_reproduction_{suffix}_1010';output.mkdir(exist_ok=True)
names=sorted(p.name for p in cache.glob('*.wav'));assert len(names)==(400 if a.natural else 188)
reports=[]
for name in names:
    x,sr=sf.read(val/('naturalNoisyAudio' if a.natural else 'syntheticNoiseAudio')/name,dtype='float32')
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
    sf.write(output/name,actual,sr,subtype='PCM_16')
    # Cached output is comparison-only, after full inference/save.
    expected,_=sf.read(cache/name,dtype='float32')
    delta=np.abs(actual-expected)
    reports.append(dict(id=Path(name).stem,max_abs=float(delta.max()),changed_samples=int((delta>0).sum())))
    print('DONE',len(reports),reports[-1]['max_abs'],flush=True)
report=dict(scope='Full released588 reproduction, not official estimate',rows=reports,
    inference_uses_cached_outputs=False,maximum_pcm_difference=max(r['max_abs'] for r in reports))
(root/f'four_reproduction_{suffix}_result.json').write_text(json.dumps(report,indent=2))
print('COMPLETE',report['maximum_pcm_difference'],flush=True)
