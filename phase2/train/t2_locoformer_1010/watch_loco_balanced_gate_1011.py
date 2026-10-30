from pathlib import Path
root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
source=(root/'watch_loco_remix_gate_1011.py').read_text()
source=source.replace('locoformer_remix_1011_s1600.pt','locoformer_balanced_1011_s800.pt')
source=source.replace('remix1600','balanced800')
exec(compile(source,str(root/'watch_loco_balanced_gate_1011.py'),'exec'))
