"""Codex remix trainer, ONE change: synthetic remix noise = d32 reused-pool events at Phase-2-like stats.
Speech/state exclusions, objectives, LR, clipping unchanged. Inference stays audio-only (no lookup)."""
import numpy as np
import os
os.environ.setdefault('USE_TF','0')
import json,random,sys
from pathlib import Path
import torch,yaml
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel
from standalone.tflocoformer_separator import TFLocoformerSeparator
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder
sys.path.insert(0,'/mnt/nvme_disk1/urgent2026_t2pilot_1008')
from finetune_mpsenet_rnnt import clean_text,rnnt_loss
from finetune_urgent_bsrnn_vaani import neg_sisdr
root=Path('/mnt/nvme_disk1/t2_locoformer_1010');cache=root/'fusion_training_cache'
manifest=json.loads((cache/'manifest.json').read_text())
seed=1032;rng=random.Random(seed);torch.manual_seed(seed)
torch.cuda.set_per_process_memory_fraction(.5)
cfg=yaml.safe_load((root/'egs2/dns_ins20/enh1/exp/enh_train_enh_tflocoformer_raw/config.yaml').read_text())
model=TFLocoformerSeparator(**cfg['separator_conf'])
raw=torch.load(root/'dns_pretrained.pth',map_location='cpu',weights_only=True)
model.load_state_dict({k.removeprefix('separator.'):v for k,v in raw.items()},strict=True)
model=model.cuda().train()
for block in model.blocks:
    original=block.forward
    def forward(*args,_original=original,**kwargs):
        return checkpoint(_original,*args,use_reentrant=False,**kwargs)
    block.forward=forward
encoder=STFTEncoder(**cfg['encoder_conf']).cuda().eval()
decoder=STFTDecoder(**cfg['decoder_conf']).cuda().eval()
asr=AutoModel.from_pretrained('/mnt/nvme_disk1/sravaani_model',trust_remote_code=True).cuda().eval();asr._ensure_loaded()
for module in (asr.encoder,asr.decoder_joint):
    for parameter in module.parameters():parameter.requires_grad_(False)
tokenizer=asr._get_tokenizer();pools=[[],[]]
_z=np.load('/mnt/nvme_disk1/t2_d32_bank.npz',allow_pickle=True)
D32=[torch.from_numpy(np.asarray(_z[k],np.float32)) for k in _z.keys() if np.abs(_z[k]).max()>0]
for ident in manifest['ids']:
    row=torch.load(cache/(ident+'.pt'),map_location='cpu',weights_only=False)
    assert row['state'] not in manifest['heldout_states']
    tokens=tokenizer.encode(clean_text(row['transcript']))
    if tokens:pools[int(row['natural'])].append((row,tokens))
assert [len(p) for p in pools]==[455,763]
opt=torch.optim.AdamW(model.parameters(),lr=3e-6,weight_decay=1e-6)
def cap(x):return x/(x.abs().amax(-1,keepdim=True)/.99).clamp_min(1)
for step in range(1,1601):
    natural=rng.random()<.35
    row,tokens=rng.choice(pools[int(natural)])
    x=row['noisy'][None].cuda()
    remixed=not natural and rng.random()<.9
    if remixed:
        clean=row['clean'][None].cuda();n=clean.shape[-1]
        act=clean[clean.abs()>.02*clean.abs().max()];srms=act.square().mean().sqrt().clamp_min(1e-6)
        residual=torch.zeros_like(clean);placed=0;guard=0
        while placed<.34*n and guard<60:
            guard+=1;a=D32[rng.randrange(len(D32))].cuda();L=min(len(a),n);p=rng.randrange(0,max(1,n-L))
            seg=a[:L]*(srms*10**(rng.gauss(9.,3.)/20)/a[:L].square().mean().sqrt().clamp_min(1e-9))
            residual[0,p:p+L]+=seg;placed+=L
        x=clean+residual
        peak=(x.abs().max()/.95).clamp_min(1);x=x/peak;target=clean/peak
    elif not natural:target=row['clean'][None].cuda()
    lengths=torch.tensor([x.shape[-1]],device='cuda');std=x.std(1,keepdim=True).clamp_min(1e-8)
    z,_=encoder(x/std,lengths);z=model(z)[:,0]
    audio,_=decoder(z,lengths);audio=cap(audio*std)
    recognition=rnnt_loss(asr,audio,tokens)
    acoustic=audio.new_zeros(()) if natural else neg_sisdr(audio,target).clamp(min=-40)
    teacher=(.7*row['base']+.3*row['loco_frozen'])[None].cuda()
    preservation=(audio-teacher).abs().mean()/teacher.abs().mean().clamp_min(1e-4) if natural else audio.new_zeros(())
    loss=2*acoustic+3*recognition+.2*preservation
    assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
    assert torch.isfinite(grad) and float(grad)>0;opt.step()
    if step==1 or step%10==0:print('STEP',step,'LOSS',float(loss.detach()),'RNNT',float(recognition.detach()),'REMIX',remixed,'GRAD',float(grad),flush=True)
    if step%200==0:
        dest=Path('/mnt/nvme_disk1/claude_d32')/f'loco_d32b_1013_s{step}.pt';temporary=dest.with_suffix('.partial')
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},cfg=cfg,step=step,seed=seed,
            optimizer=opt.state_dict(),rng_state=rng.getstate(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),
            training_ids=manifest['ids'],heldout_states=manifest['heldout_states'],objective=dict(si=2,rnnt=3,natural_preserve=.2,
            synthetic_remix_prob=.9,noise='d32 reused-pool events, +9dB(3) over speech, coverage .34',natural_prob=.35),inference='Uniform general neural enhancement; no noise lookup or metadata routing'),temporary)
        temporary.replace(dest);print('SAVED',dest,flush=True)
