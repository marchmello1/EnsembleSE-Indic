"""Controlled fused-objective long-context adaptation, released training only."""
from pathlib import Path
root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
source=(root/'train_loco_remix_1011.py').read_text()
changes={
    'seed=1027':'seed=1042',
    'set_per_process_memory_fraction(.5)':'set_per_process_memory_fraction(.45)',
    'lr=3e-6':'lr=1e-6',
    'range(1,1601)':'range(1,401)',
    "remixed=not natural and rng.random()<.5":"remixed=False",
    "recognition=rnnt_loss(asr,audio,tokens)":"teacher_base=row['base'][None].cuda()\n    fused=cap(.7*teacher_base+.3*audio)\n    recognition=rnnt_loss(asr,fused,tokens)",
    'neg_sisdr(audio,target)':'neg_sisdr(fused,target)',
    '(audio-teacher).abs().mean()':'(fused-teacher).abs().mean()',
    "f'locoformer_remix_1011_s{step}.pt'":"f'locoformer_longcontext_1012_s{step}.pt'",
    'synthetic_remix_prob=.5,snr_range=[0,20],natural_prob=.35':'synthetic_remix_prob=0,long_context_seconds=[10,16],join_policy="same-state training utterances",natural_prob=.35,fused_objective=True',
}
for old,new in changes.items():
    assert old in source, old
    source=source.replace(old,new)
old="    row,tokens=rng.choice(pools[int(natural)])\n    x=row['noisy'][None].cuda()"
new="""    first,_=rng.choice(pools[int(natural)])
    compatible=[r for r,_ in pools[int(natural)] if r['state']==first['state']]
    assert compatible
    chosen=[first];length=len(first['noisy'])
    while length<10*16000:
        possible=[r for r in compatible if length+len(r['noisy'])<=16*16000]
        if not possible:break
        nxt=rng.choice(possible);chosen.append(nxt);length+=len(nxt['noisy'])
    row={k:torch.cat([r[k] for r in chosen]) for k in ('noisy','base','loco_frozen')}
    if not natural:row['clean']=torch.cat([r['clean'] for r in chosen])
    tokens=tokenizer.encode(' '.join(clean_text(r['transcript']) for r in chosen))
    assert tokens and length<=16*16000
    if step==1 or step%10==0:print('CONTEXT',step,length/16000,first['state'],len(chosen),flush=True)
    x=row['noisy'][None].cuda()"""
assert old in source
source=source.replace(old,new)
exec(compile(source,str(root/'train_loco_longcontext_1012.py'),'exec'))
