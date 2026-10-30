"""Released-validation severity diagnostics; not an inference selector."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf

root=Path('/mnt/nvme_disk1/official_validation_20260907')
baseline=Path('/mnt/nvme_disk1/urgent2026_t2pilot_1008/extended_s1600_ls70_hold188_wavs')
def si(x,y):
    x=np.asarray(x,np.float64)-np.mean(x)
    y=np.asarray(y,np.float64)-np.mean(y)
    target=y*np.dot(x,y)/(np.dot(y,y)+1e-12)
    return float(10*np.log10((np.sum(target**2)+1e-12)/(np.sum((x-target)**2)+1e-12)))
groups={key:[] for key in ('negative','0_10','10_20','20_30','30_plus')}
for path in baseline.glob('*.wav'):
    clean,sr=sf.read(root/'syntheticCleanRefAudio'/path.name)
    noisy,nr=sf.read(root/'syntheticNoiseAudio'/path.name)
    pred,pr=sf.read(path)
    assert sr==nr==pr==16000 and len(clean)==len(noisy)==len(pred)
    raw=si(noisy,clean);enh=si(pred,clean)
    index=int(np.digitize(raw,[0,10,20,30]))
    groups[list(groups)[index]].append({'id':path.stem,'noisy':raw,'enhanced':enh,'gain':enh-raw})
result={key:{'count':len(rows),'noisy_mean':float(np.mean([r['noisy'] for r in rows])) if rows else None,
             'enhanced_mean':float(np.mean([r['enhanced'] for r in rows])) if rows else None,
             'gain':float(np.mean([r['gain'] for r in rows])) if rows else None} for key,rows in groups.items()}
Path('/mnt/nvme_disk1/t2_noise_severity_held188_1009.json').write_text(json.dumps({'summary':result,'rows':groups},indent=2))
print(json.dumps(result),flush=True)
