"""Prepare state-disjoint released training outputs for learned residual fusion."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

pilot=Path('/mnt/nvme_disk1/urgent2026_t2pilot_1008')
package=Path('/mnt/nvme_disk1/t2_bsrnn_s3000_codepkg_1008/marchmello01')
sys.path.insert(0,str(package))
sys.path.insert(0,str(pilot))
import mp_core as core
from baseline_code.d_model import SEModel

meta={r['segmentFileName'].removesuffix('.wav'):r for r in json.loads(Path('/mnt/nvme_disk1/official_validation_20260907/validationMetadata.json').read_text())}
held=torch.load('/mnt/nvme_disk1/t2_binary_pred_hold_0924_t0.50.pt',map_location='cpu',weights_only=False)
held_ids={r['id'] for r in held}
held_states={meta[i]['state'] for i in held_ids}
rows=torch.load('/mnt/nvme_disk1/t2_natural_predtrain_1008.pt',map_location='cpu',weights_only=False)
assert not held_ids.intersection(r['id'] for r in rows)
assert not any(meta[r['id']]['state'] in held_states for r in rows)
device=torch.device('cuda')
cfg=core.AttrDict(json.loads((package/'vendor/best_ckpt/config.json').read_text()))
mp=core.load_enhancer(package/'checkpoints/t2_mpdns_publicnoise_hilr_1008.pt',cfg,device).eval()
bs=SEModel.load_from_checkpoint(str(pilot/'bsrnn_ls70_rnnt_extended_1009_s1600.ckpt'),map_location='cpu').cuda().eval()
out=[]
for row in rows:
    noisy=np.asarray(row['waveform'],np.float32).reshape(-1)
    primary=core.enhance_full_clip(noisy,mp,cfg,device)
    x=torch.from_numpy(noisy)[None].cuda()
    with torch.inference_mode():
        secondary,_=bs.se_model(x,torch.tensor([len(noisy)],device=device),16000)
    secondary=secondary.cpu().numpy().reshape(-1)[:len(noisy)]
    assert len(primary)==len(secondary)==len(noisy)
    secondary*=.9/max(.9,float(np.max(np.abs(secondary))))
    left=primary-primary.mean();right=secondary-secondary.mean()
    gain=float(np.clip(np.dot(left,right)/(np.dot(right,right)+1e-12),.1,10))
    out.append({'id':row['id'],'state':meta[row['id']]['state'],
                'noisy':torch.from_numpy(noisy),'mp':torch.from_numpy(primary),
                'bs':torch.from_numpy(secondary*gain)})
    if len(out)%20==0:
        print('DONE',len(out),flush=True)
dest=pilot/'neural_fusion_natural763_1009.pt'
torch.save(out,dest)
print('FINAL',len(out),dest,flush=True)
