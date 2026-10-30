import json,random
from pathlib import Path
import torch
root=Path('/mnt/nvme_disk1/t2_locoformer_1010');cache=root/'fusion_training_cache'
manifest=json.loads((cache/'manifest.json').read_text())
rows=[torch.load(cache/(i+'.pt'),map_location='cpu',weights_only=False) for i in manifest['ids']]
synthetic=[r for r in rows if not r['natural']]
assert len(rows)==1218 and len(synthetic)==455
assert all(r['state'] not in manifest['heldout_states'] for r in rows)
rng=random.Random(1027);measured=[]
for _ in range(100):
    speech,noise=rng.sample(synthetic,2);assert speech['id']!=noise['id']
    clean=speech['clean'];residual=noise['noisy']-noise['clean']
    residual=torch.roll(residual,rng.randrange(len(residual)))
    residual=residual.repeat((len(clean)+len(residual)-1)//len(residual))[:len(clean)]
    snr=rng.uniform(0,20)
    residual=residual*clean.square().mean().sqrt()/residual.square().mean().sqrt().clamp_min(1e-6)*10**(-snr/20)
    mixture=clean+residual;scale=(mixture.abs().max()/.95).clamp_min(1)
    x=mixture/scale;c=clean/scale
    assert x.shape==c.shape and torch.isfinite(x).all() and torch.isfinite(c).all()
    assert float(x.abs().max())<=.95001
    realized=float(10*torch.log10(c.square().mean()/(x-c).square().mean()))
    assert abs(realized-snr)<.001
    measured.append(realized)
report=dict(scope='100 training-only augmentation draws, not heldout performance',draws=100,
    distinct_sources=True,heldout_states_excluded=True,finite_exactlength_joint_scaling=True,
    realized_snr_min=min(measured),realized_snr_max=max(measured))
(root/'remix_data_audit_1011.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
