"""Reuse audited state-excluded data/loss loop with a non-convex correction head."""
from pathlib import Path
root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
source=(root/'train_loco_spectral_fusion_1010.py').read_text()
assert 'seed=1025' in source and 'model=SpectralFusion()' in source
source=source.replace('from loco_spectral_fusion_1010 import SpectralFusion',
                      'from loco_complex_correction_1011 import ComplexCorrection as SpectralFusion')
source=source.replace('seed=1025','seed=1026')
source=source.replace('loco_spectral_fusion_1010_s','loco_complex_correction_1011_s')
source=source.replace('Same audio-only spectral neural fusion for all clips; no raw bypass',
    'Uniform audio-only bounded complex-mask correction of fixed enhanced waveform; no raw bypass')
exec(compile(source,str(root/'train_loco_complex_correction_1011.py'),'exec'))
