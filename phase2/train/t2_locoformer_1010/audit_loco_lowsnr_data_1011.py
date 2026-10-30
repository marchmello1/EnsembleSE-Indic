from pathlib import Path
root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
source=(root/'audit_loco_remix_data_1011.py').read_text()
source=source.replace('rng.uniform(0,20)','rng.uniform(-15,10)')
source=source.replace('remix_data_audit_1011.json','lowsnr_data_audit_1011.json')
exec(compile(source,str(root/'audit_loco_lowsnr_data_1011.py'),'exec'))
