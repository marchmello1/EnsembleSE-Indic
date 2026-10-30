from pathlib import Path
root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
source=(root/'eval_loco_spectral_fusion_1010.py').read_text()
source=source.replace('from loco_spectral_fusion_1010 import SpectralFusion',
                      'from loco_complex_correction_1011 import ComplexCorrection as SpectralFusion')
source=source.replace('loco_spectral_fusion_1010_s1200.pt','loco_complex_correction_1011_s1200.pt')
source=source.replace("default='loco_spectral1200'","default='loco_complex1200'")
exec(compile(source,str(root/'eval_loco_complex_correction_1011.py'),'exec'))
