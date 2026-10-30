"""Released state-disjoint RNNT adaptation of the independent Locoformer branch."""
import os
os.environ.setdefault('USE_TF','0')
import argparse,json,random,sys
from pathlib import Path
import torch
import yaml
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from transformers import AutoModel
from standalone.tflocoformer_separator import TFLocoformerSeparator
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder
sys.path.insert(0,'/mnt/nvme_disk1/t2_gap_urgenet_1009')
from models.predictor.tfgridnet import TFGridNet
sys.path.insert(0,'/mnt/nvme_disk1/urgent2026_t2pilot_1008')
from finetune_mpsenet_rnnt import clean_text,rnnt_loss
from finetune_urgent_bsrnn_vaani import neg_sisdr

p=argparse.ArgumentParser()
p.add_argument('--steps',type=int,default=800)
p.add_argument('--seed',type=int,default=1023)
p.add_argument('--lr',type=float,default=1e-6)
p.add_argument('--acoustic-weight',type=float,default=2.)
p.add_argument('--direct-acoustic-weight',type=float,default=0.)
p.add_argument('--rnnt-weight',type=float,default=3.)
p.add_argument('--natural-prob',type=float,default=.5)
p.add_argument('--preserve-weight',type=float,default=.5)
p.add_argument('--preserve-base',action='store_true')
p.add_argument('--tag',default='rnnt')
a=p.parse_args()
torch.cuda.set_per_process_memory_fraction(.5)
root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
validation=Path('/mnt/nvme_disk1/official_validation_20260907')
meta={r['segmentFileName'].removesuffix('.wav'):r for r in json.loads((validation/'validationMetadata.json').read_text())}
held={q.stem for q in Path('/mnt/nvme_disk1/urgent2026_t2pilot_1008/cont_s3000_hold188_wavs').glob('*.wav')}
states={meta[i]['state'] for i in held}
assert len(held)==188 and len(states)==6
cfg=yaml.safe_load((root/'egs2/dns_ins20/enh1/exp/enh_train_enh_tflocoformer_raw/config.yaml').read_text())
model=TFLocoformerSeparator(**cfg['separator_conf'])
ck=torch.load(root/'dns_pretrained.pth',map_location='cpu',weights_only=True)
model.load_state_dict({k.removeprefix('separator.'):v for k,v in ck.items()},strict=True)
model=model.cuda().train()
for block in model.blocks:
    original_forward=block.forward
    def memory_safe_forward(*args,_forward=original_forward,**kwargs):
        return activation_checkpoint(_forward,*args,use_reentrant=False,**kwargs)
    block.forward=memory_safe_forward
encoder=STFTEncoder(**cfg['encoder_conf']).cuda().eval()
decoder=STFTDecoder(**cfg['decoder_conf']).cuda().eval()
predictor_ck=torch.load('/mnt/nvme_disk1/t2_gap_urgenet_1009/predictor_fulladapt_1010_s800.pt',map_location='cpu',weights_only=False)
assert not held.intersection(predictor_ck['training_ids'])
assert not states.intersection(meta[i]['state'] for i in predictor_ck['training_ids'])
predictor=TFGridNet(**predictor_ck['cfg']).cuda().eval()
predictor.load_state_dict(predictor_ck['model'],strict=True)
for param in predictor.parameters():param.requires_grad_(False)
asr=AutoModel.from_pretrained('/mnt/nvme_disk1/sravaani_model',trust_remote_code=True).cuda().eval()
asr._ensure_loaded()
for module in (asr.encoder,asr.decoder_joint):
    for param in module.parameters():param.requires_grad_(False)
tokenizer=asr._get_tokenizer()
pools=[]
for cache in ('neural_fusion_train507_1009.pt','neural_fusion_natural763_1009.pt'):
    rows=torch.load(Path('/mnt/nvme_disk1/urgent2026_t2pilot_1008')/cache,map_location='cpu',weights_only=False)
    assert not held.intersection(r['id'] for r in rows)
    assert not states.intersection(r['state'] for r in rows)
    pool=[]
    for row in rows:
        tokens=tokenizer.encode(clean_text(meta[row['id']]['transcript']))
        if len(row['noisy'])<=8*16000 and tokens:pool.append((row,tokens))
    assert pool
    pools.append(pool)
print('POOLS',[len(q) for q in pools],'TRAINABLE',sum(q.numel() for q in model.parameters()),flush=True)
opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-6)
rng=random.Random(a.seed);torch.manual_seed(a.seed)
def cap(x):return x/(x.abs().amax(-1,keepdim=True)/.99).clamp_min(1)
def pcm(x):return torch.floor(x.clamp(-1,1-1/32768)*32768)/32768
for step in range(1,a.steps+1):
    natural=rng.random()<a.natural_prob
    row,tokens=rng.choice(pools[int(natural)])
    x=row['noisy'][None].cuda()
    with torch.no_grad():
        ls=cap((.3*row['mp']+.7*row['bs'])[None].cuda())
        prediction=predictor(x,sr=16000)
        prediction=prediction/prediction.abs().amax(-1,keepdim=True).clamp_min(1e-8)*x.abs().amax(-1,keepdim=True)
        base=pcm(cap(.7*pcm(ls)+.3*pcm(prediction)))
    lengths=torch.tensor([x.shape[-1]],device=x.device)
    std=x.std(1,keepdim=True).clamp_min(1e-8)
    features,flens=encoder(x/std,lengths)
    spectrum=model(features)
    candidate,_=decoder(spectrum[:,0],lengths)
    candidate=cap(candidate*std)
    y=cap(.7*base+.3*candidate)
    recognition=rnnt_loss(asr,y,tokens)
    if natural:
        acoustic=y.new_zeros(())
        direct_acoustic=y.new_zeros(())
        preserve_target=base if a.preserve_base else x
        preserve=(y-preserve_target).abs().mean()/preserve_target.abs().mean().clamp_min(1e-4)
    else:
        acoustic=neg_sisdr(y,row['clean'][None].cuda()).clamp(min=-40)
        direct_acoustic=neg_sisdr(candidate,row['clean'][None].cuda()).clamp(min=-40)
        preserve=y.new_zeros(())
    loss=a.acoustic_weight*acoustic+a.direct_acoustic_weight*direct_acoustic+a.rnnt_weight*recognition+a.preserve_weight*preserve
    if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
    opt.zero_grad(set_to_none=True);loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
    if not torch.isfinite(grad) or float(grad)==0:raise RuntimeError('invalid gradient')
    opt.step()
    if step==1 or step%10==0:
        print('STEP',step,'LOSS',float(loss.detach()),'RNNT',float(recognition.detach()),'SI',float(-acoustic.detach()),'GRAD',float(grad),flush=True)
    if step%200==0 or step==a.steps:
        output={'cfg':cfg,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'step':step,
                'training_ids':[r['id'] for pool in pools for r,_ in pool],'heldout_states':sorted(states),
                'seed':a.seed,'optimizer':opt.state_dict(),'rng_state':rng.getstate(),'torch_rng_state':torch.get_rng_state(),
                'branch_weight':.3,'acoustic_weight':a.acoustic_weight,'direct_acoustic_weight':a.direct_acoustic_weight,
                'rnnt_weight':a.rnnt_weight,'natural_prob':a.natural_prob,'preserve_weight':a.preserve_weight,
                'preserve_base':a.preserve_base,'lr':a.lr,
                'base_predictor':'predictor_fulladapt_1010_s800.pt','source_checkpoint':'dns_pretrained.pth'}
        target=root/f'locoformer_{a.tag}_1010_s{step}.pt'
        torch.save(output,target);print('SAVED',target,flush=True)
