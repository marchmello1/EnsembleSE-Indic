"""Controlled gradient-balanced continuation of stochastic-remix Locoformer."""
from pathlib import Path
root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
source=(root/'train_loco_remix_1011.py').read_text()
source=source.replace('seed=1027','seed=1028')
source=source.replace('lr=3e-6','lr=1e-6')
source=source.replace('range(1,1601)','range(1,801)')
source=source.replace('model=model.cuda().train()',
    "start=torch.load(root/'locoformer_remix_1011_s1600.pt',map_location='cpu',weights_only=False)\n"
    "model.load_state_dict(start['model'],strict=True)\nmodel=model.cuda().train()")
old='opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),3.)'
assert old in source
new='''opt.zero_grad(set_to_none=True)
    if natural:
        loss.backward();balance=1.
    else:
        parameters=[p for p in model.parameters() if p.requires_grad]
        ag=torch.autograd.grad(2*acoustic,parameters,retain_graph=True,allow_unused=True)
        rg=torch.autograd.grad(3*recognition,parameters,allow_unused=True)
        an=sum(g.detach().square().sum() for g in ag if g is not None).sqrt()
        rn=sum(g.detach().square().sum() for g in rg if g is not None).sqrt()
        assert torch.isfinite(an) and torch.isfinite(rn) and rn>0
        balance=float((rn/an.clamp_min(1e-8)).clamp(.1,10))
        for p,a,r in zip(parameters,ag,rg):
            p.grad=(torch.zeros_like(p) if r is None else r.detach().clone())
            if a is not None:p.grad.add_(a.detach(),alpha=balance)
        del ag,rg
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),3.)'''
source=source.replace(old,new)
source=source.replace("'REMIX',remixed,'GRAD'","'REMIX',remixed,'SI_GRAD_FACTOR',balance,'GRAD'")
source=source.replace('locoformer_remix_1011_s','locoformer_balanced_1011_s')
# The start checkpoint must remain the original remix1600, not the new output prefix.
source=source.replace("root/'locoformer_balanced_1011_s1600.pt'","root/'locoformer_remix_1011_s1600.pt'")
source=source.replace('synthetic_remix_prob=.5,snr_range=[0,20],natural_prob=.35)',
    "synthetic_remix_prob=.5,snr_range=[0,20],natural_prob=.35,gradient_balance='SI parameter-gradient norm matched to RNNT within .1-10 factor',start_checkpoint='remix1600',optimizer_restart=True)")
exec(compile(source,str(root/'train_loco_balanced_1011.py'),'exec'))
