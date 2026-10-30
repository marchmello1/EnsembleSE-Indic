"""State-excluded fusion training; no held/test supervision or routing."""
import os
os.environ.setdefault('USE_TF','0')
import json,random,sys,time
from pathlib import Path
import torch
from transformers import AutoModel
from loco_spectral_fusion_1010 import SpectralFusion
sys.path.insert(0,'/mnt/nvme_disk1/urgent2026_t2pilot_1008')
from finetune_mpsenet_rnnt import clean_text,rnnt_loss
from finetune_urgent_bsrnn_vaani import neg_sisdr

root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
cache=root/'fusion_training_cache'
deadline=time.monotonic()+21600
while not (cache/'manifest.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('Inspect cache job, no automatic restart')
    time.sleep(10)
manifest=json.loads((cache/'manifest.json').read_text())
torch.cuda.set_per_process_memory_fraction(.5)
seed=1025;torch.manual_seed(seed);rng=random.Random(seed)
model=SpectralFusion().cuda().train()
asr=AutoModel.from_pretrained('/mnt/nvme_disk1/sravaani_model',trust_remote_code=True).cuda().eval()
asr._ensure_loaded()
for module in (asr.encoder,asr.decoder_joint):
    for param in module.parameters():param.requires_grad_(False)
tokenizer=asr._get_tokenizer();pools=[[],[]]
for ident in manifest['ids']:
    row=torch.load(cache/(ident+'.pt'),map_location='cpu',weights_only=False)
    assert row['state'] not in manifest['heldout_states']
    tokens=tokenizer.encode(clean_text(row['transcript']))
    if tokens:pools[int(row['natural'])].append((row,tokens))
assert all(pools)
print('POOLS',[len(p) for p in pools],'PARAMS',sum(p.numel() for p in model.parameters()),flush=True)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
for step in range(1,1201):
    natural=rng.random()<.4
    row,tokens=rng.choice(pools[int(natural)])
    x=row['noisy'][None].cuda()
    estimates=torch.stack([row[k] for k in ('base','loco_frozen','loco_acoustic')])[None].cuda()
    target=.7*estimates[:,0]+.3*estimates[:,1]
    audio,weights=model(x,estimates)
    recognition=rnnt_loss(asr,audio,tokens)
    acoustic=audio.new_zeros(()) if natural else neg_sisdr(audio,row['clean'][None].cuda()).clamp(min=-40)
    preservation=(audio-target).abs().mean()/target.abs().mean().clamp_min(1e-4) if natural else audio.new_zeros(())
    smooth=(weights[:,:,:,1:]-weights[:,:,:,:-1]).abs().mean()
    loss=2*acoustic+3*recognition+.3*preservation+.01*smooth
    if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
    optimizer.zero_grad(set_to_none=True);loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
    if not torch.isfinite(grad) or float(grad)==0:raise RuntimeError('invalid gradient')
    optimizer.step()
    if step==1 or step%10==0:
        print('STEP',step,'LOSS',float(loss.detach()),'RNNT',float(recognition.detach()),'SI',float(-acoustic.detach()),'GRAD',float(grad),flush=True)
    if step%200==0:
        target=root/f'loco_spectral_fusion_1010_s{step}.pt'
        temporary=target.with_suffix('.partial')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'seed':seed,
                    'training_ids':[r['id'] for pool in pools for r,_ in pool],
                    'heldout_states':manifest['heldout_states'],'rng_state':rng.getstate(),
                    'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state(),
                    'objective':{'si':2,'rnnt':3,'natural_preserve':.3,'smooth':.01},
                    'inference':'Same audio-only spectral neural fusion for all clips; no raw bypass'},temporary)
        temporary.replace(target);print('SAVED',str(target),flush=True)
