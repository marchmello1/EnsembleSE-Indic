"""ASR-aware predictor adaptation on released state-disjoint training pairs."""
import argparse
import json
import os
import random
import sys
from pathlib import Path
os.environ.setdefault('USE_TF','0')
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from transformers import AutoModel

sys.path.insert(0,'/mnt/nvme_disk1/urgent2026_t2pilot_1008')
from finetune_mpsenet_rnnt import clean_text,rnnt_loss
from finetune_urgent_bsrnn_vaani import neg_sisdr
from models.predictor.tfgridnet import TFGridNet

p=argparse.ArgumentParser()
p.add_argument('--steps',type=int,default=800)
p.add_argument('--rnnt-weight',type=float,default=3.)
p.add_argument('--lr',type=float,default=1e-5)
p.add_argument('--checkpoint',type=Path)
p.add_argument('--natural-cache',type=Path)
p.add_argument('--natural-prob',type=float,default=.5)
p.add_argument('--seed',type=int,default=1012)
p.add_argument('--prefix',default='predictor_rnnt_1009')
a=p.parse_args()
torch.cuda.set_per_process_memory_fraction(.5)
root=Path('/mnt/nvme_disk1/t2_gap_urgenet_1009')
validation=Path('/mnt/nvme_disk1/official_validation_20260907')
meta={r['segmentFileName'].removesuffix('.wav'):r for r in json.loads((validation/'validationMetadata.json').read_text())}
held={q.stem for q in Path('/mnt/nvme_disk1/urgent2026_t2pilot_1008/cont_s3000_hold188_wavs').glob('*.wav')}
states={meta[i]['state'] for i in held}
rows=torch.load('/mnt/nvme_disk1/urgent2026_t2pilot_1008/neural_fusion_train507_1009.pt',map_location='cpu',weights_only=False)
assert len(held)==188 and not held.intersection(r['id'] for r in rows)
assert not states.intersection(r['state'] for r in rows)
checkpoint=torch.load(a.checkpoint or root/'checkpoints/Predictor.pt',map_location='cpu',weights_only=False)
model=TFGridNet(**checkpoint['cfg'])
model.load_state_dict(checkpoint.get('generator',checkpoint.get('model')),strict=True)
model=model.cuda().train()
for param in model.parameters():
    param.requires_grad_(True)
for block in model.blocks:
    original_forward=block.forward
    def memory_safe_forward(*args, _forward=original_forward, **kwargs):
        return activation_checkpoint(_forward,*args,use_reentrant=False,**kwargs)
    block.forward=memory_safe_forward
asr=AutoModel.from_pretrained('/mnt/nvme_disk1/sravaani_model',trust_remote_code=True).cuda().eval()
asr._ensure_loaded()
for module in (asr.encoder,asr.decoder_joint):
    for param in module.parameters():
        param.requires_grad_(False)
tokenizer=asr._get_tokenizer()
pools=[]
sources=[rows]
if a.natural_cache:
    natural_rows=torch.load(a.natural_cache,map_location='cpu',weights_only=False)
    assert not held.intersection(r['id'] for r in natural_rows)
    assert not states.intersection(r['state'] for r in natural_rows)
    sources.append(natural_rows)
for source in sources:
    pool=[]
    for row in source:
        tokens=tokenizer.encode(clean_text(meta[row['id']]['transcript']))
        if len(row['noisy'])<=8*16000 and tokens:
            pool.append((row,tokens))
    assert pool
    pools.append(pool)
print('POOLS',[len(p) for p in pools],'TRAINABLE',sum(p.numel() for p in model.parameters() if p.requires_grad),flush=True)
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-6)
rng=random.Random(a.seed);torch.manual_seed(a.seed)
for step in range(1,a.steps+1):
    natural=len(pools)>1 and rng.random()<a.natural_prob
    row,tokens=rng.choice(pools[int(natural)])
    x=row['noisy'][None].cuda()
    base=(.3*row['mp']+.7*row['bs'])[None].cuda()
    base=base/(base.abs().amax(-1,keepdim=True)/.99).clamp_min(1)
    # Learn removal of residual noise/artifacts from the fixed neural first stage.
    # This input is produced by models, never by a reference/template lookup.
    staged=torch.floor(base.clamp(-1,1-1/32768)*32768)/32768
    out=model(staged,sr=16000)
    assert out.shape==x.shape
    out=out/out.abs().amax(-1,keepdim=True).clamp_min(1e-8)*x.abs().amax(-1,keepdim=True)
    y=.7*base+.3*out
    y=y/(y.abs().amax(-1,keepdim=True)/.99).clamp_min(1)
    recognition=rnnt_loss(asr,y,tokens)
    if natural:
        acoustic=y.new_zeros(())
        preserve=(y-x).abs().mean()/x.abs().mean().clamp_min(1e-4)
    else:
        target=row['clean'][None].cuda()
        acoustic=neg_sisdr(y,target).clamp(min=-40)
        preserve=y.new_zeros(())
    loss=2*acoustic+a.rnnt_weight*recognition+.5*preserve
    if not torch.isfinite(loss):
        raise RuntimeError('nonfinite loss')
    opt.zero_grad(set_to_none=True);loss.backward()
    grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],3.)
    if not torch.isfinite(grad) or float(grad)==0:
        raise RuntimeError('invalid gradient')
    opt.step()
    if step==1 or step%10==0:
        print('STEP',step,'LOSS',float(loss.detach()),'RNNT',float(recognition.detach()),'SI',float(-acoustic.detach()),'GRAD',float(grad),flush=True)
    if step%200==0 or step==a.steps:
        result={'cfg':checkpoint['cfg'],'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'step':step,'training_ids':[r['id'] for pool in pools for r,_ in pool],
                'heldout_states':sorted(states),'blend_weight':.3,'seed':a.seed,
                'inference_input':'uniform_ls70_neural_output_PCM16',
                'source_checkpoint':str(a.checkpoint),'lr':a.lr,'rnnt_weight':a.rnnt_weight,
                'acoustic_weight':2,'natural_prob':a.natural_prob,
                'optimizer':opt.state_dict(),'rng_state':rng.getstate(),
                'torch_rng_state':torch.get_rng_state(),
                'continuation_note':'Best checkpoint weights, restarted AdamW; all blocks trainable.'}
        path=root/f'{a.prefix}_s{step}.pt'
        torch.save(result,path);print('SAVED',path,flush=True)

