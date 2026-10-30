from pathlib import Path
root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
source=(root/'watch_loco_longcontext_gate_1012.py').read_text()
source=source.replace('longcontext_1012_s400.pt','longcontext_1012_s1600.pt')
source=source.replace('longcontext400','longcontext1600').replace('Final400','Final1600')
exec(compile(source,str(root/'watch_loco_longcontext1600_gate_1012.py'),'exec'))
