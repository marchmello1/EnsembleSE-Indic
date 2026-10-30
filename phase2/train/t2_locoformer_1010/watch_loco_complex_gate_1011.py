from pathlib import Path
root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
source=(root/'watch_loco_spectral_fusion_gate_1010.py').read_text()
source=source.replace('loco_spectral_fusion_1010_s1200.pt','loco_complex_correction_1011_s1200.pt')
source=source.replace('eval_loco_spectral_fusion_1010.py','eval_loco_complex_correction_1011.py')
source=source.replace('loco_spectral1200','loco_complex1200')
source=source.replace("CUDA_VISIBLE_DEVICES='7'","CUDA_VISIBLE_DEVICES='0'")
exec(compile(source,str(root/'watch_loco_complex_gate_1011.py'),'exec'))
