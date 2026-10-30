"""Objective gradient diagnosis on state-excluded training examples only."""
from pathlib import Path
root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
source=(root/'train_loco_remix_1011.py').read_text().split('opt=torch.optim.AdamW')[0]
exec(compile(source,str(root/'train_loco_remix_1011.py'),'exec'))
ck=torch.load(root/'locoformer_remix_1011_s1600.pt',map_location='cpu',weights_only=False)
model.load_state_dict(ck['model'],strict=True)
parameters=[p for p in model.parameters() if p.requires_grad]
def cap(x):return x/(x.abs().amax(-1,keepdim=True)/.99).clamp_min(1)
def norm(gradients):return sum(g.detach().double().square().sum() for g in gradients if g is not None).sqrt()
reports=[]
remix_mode=os.environ.get('DIAG_REMIX')=='1'
suffix='_remix' if remix_mode else ''
for index in range(8):
    row,tokens=pools[0][index*37]
    x=row['noisy'][None].cuda();target=row['clean'][None].cuda()
    if remix_mode:
        noise_row,_=pools[0][(index*37+111)%len(pools[0])]
        assert noise_row['id']!=row['id']
        residual=noise_row['noisy']-noise_row['clean']
        residual=torch.roll(residual,rng.randrange(len(residual)))
        residual=residual.repeat((x.shape[-1]+len(residual)-1)//len(residual))[:x.shape[-1]][None].cuda()
        snr=rng.uniform(0,20)
        residual=residual*target.square().mean().sqrt()/residual.square().mean().sqrt().clamp_min(1e-6)*10**(-snr/20)
        x=target+residual;peak=(x.abs().max()/.95).clamp_min(1);x=x/peak;target=target/peak
    lengths=torch.tensor([x.shape[-1]],device='cuda');std=x.std(1,keepdim=True).clamp_min(1e-8)
    z,_=encoder(x/std,lengths);z=model(z)[:,0]
    audio,_=decoder(z,lengths);audio=cap(audio*std)
    acoustic=2*neg_sisdr(audio,target).clamp(min=-40)
    recognition=3*rnnt_loss(asr,audio,tokens)
    ag=torch.autograd.grad(acoustic,parameters,retain_graph=True,allow_unused=True)
    rg=torch.autograd.grad(recognition,parameters,allow_unused=True)
    an=norm(ag);rn=norm(rg)
    dot=sum((a.detach().double()*r.detach().double()).sum() for a,r in zip(ag,rg) if a is not None and r is not None)
    assert torch.isfinite(an) and torch.isfinite(rn) and rn>0
    result=dict(id=row['id'],state=row['state'],remixed=remix_mode,weighted_acoustic_gradient_norm=float(an),
        weighted_rnnt_gradient_norm=float(rn),acoustic_to_rnnt_norm_ratio=float(an/rn),
        cosine=float(dot/(an*rn).clamp_min(1e-12)))
    reports.append(result);print('GRADIENT',result,flush=True)
    del ag,rg,audio,z,acoustic,recognition
    (root/f'gradient_balance_progress{suffix}_1011.json').write_text(json.dumps(reports,indent=2))
ratios=sorted(r['acoustic_to_rnnt_norm_ratio'] for r in reports)
report=dict(scope='Eight state-excluded training examples; no held/test measurements',rows=reports,
    median_ratio=.5*(ratios[3]+ratios[4]),negative_cosine_count=sum(r['cosine']<0 for r in reports))
(root/f'gradient_balance_audit{suffix}_1011.json').write_text(json.dumps(report,indent=2));print('COMPLETE',report['median_ratio'],flush=True)
