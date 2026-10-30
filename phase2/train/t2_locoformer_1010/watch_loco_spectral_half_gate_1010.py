"""One preregistered fixed50% residual-scale diagnostic, both held subsets."""
from pathlib import Path
source=Path('/mnt/nvme_disk1/t2_locoformer_1010/watch_loco_spectral_fusion_gate_1010.py').read_text()
source=source.replace('loco_spectral1200','loco_spectral1200half')
source=source.replace("'eval_loco_spectral_fusion_1010.py'", "'eval_loco_spectral_fusion_1010.py','--residual-scale','0.5','--tag','loco_spectral1200half'")
exec(compile(source,__file__,'exec'))
