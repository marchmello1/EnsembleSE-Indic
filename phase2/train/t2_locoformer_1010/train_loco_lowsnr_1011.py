"""Controlled range correction: same DNS init, seed, losses, and training sources."""
from pathlib import Path
root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
source=(root/'train_loco_remix_1011.py').read_text()
assert 'snr=rng.uniform(0,20)' in source
source=source.replace('snr=rng.uniform(0,20)','snr=rng.uniform(-15,10)')
source=source.replace('snr_range=[0,20]','snr_range=[-15,10]')
source=source.replace('locoformer_remix_1011_s','locoformer_lowsnr_1011_s')
exec(compile(source,str(root/'train_loco_lowsnr_1011.py'),'exec'))
