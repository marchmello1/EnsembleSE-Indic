"""Resume the controlled long-context run, including optimizer and RNG state."""
from pathlib import Path
root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
wrapper=(root/'train_loco_longcontext_1012.py').read_text()
assert "'range(1,1601)':'range(1,401)'" in wrapper
wrapper=wrapper.replace("'range(1,1601)':'range(1,401)'", "'range(1,1601)':'range(401,1601)'")
ending="exec(compile(source,str(root/'train_loco_longcontext_1012.py'),'exec'))"
assert ending in wrapper
resume="""
resume=torch.load(root/'locoformer_longcontext_1012_s400.pt',map_location='cpu',weights_only=False)
assert resume['step']==400 and set(resume['training_ids'])==set(manifest['ids'])
assert resume['heldout_states']==manifest['heldout_states']
model.load_state_dict(resume['model'],strict=True)
opt.load_state_dict(resume['optimizer'])
rng.setstate(resume['rng_state'])
torch.set_rng_state(resume['torch_rng_state'])
torch.cuda.set_rng_state(resume['cuda_rng_state'])
print('RESUMED',resume['step'],'TO',1600,flush=True)
"""
injection="source=source.replace('def cap(x):', "+repr(resume)+"+'\\ndef cap(x):')\n"+ending
wrapper=wrapper.replace(ending,injection)
exec(compile(wrapper,str(root/'continue_loco_longcontext_1012.py'),'exec'))
