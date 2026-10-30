from pathlib import Path
root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
source=(root/'watch_loco_remix_gate_1011.py').read_text()
source=source.replace('locoformer_remix_1011_s1600.pt','locoformer_longcontext_1012_s400.pt')
source=source.replace('remix1600','longcontext400')
source=source.replace("CUDA_VISIBLE_DEVICES='6'","CUDA_VISIBLE_DEVICES='7'")
source=source.replace("print('ALL_GATES_COMPLETE',flush=True)","""
import json
for baseline in ('official','frozen'):
    path=Path(f'/mnt/nvme_disk1/t2_locoformer_longcontext400_matched_{baseline}_audit_1010.json')
    report=json.loads(path.read_text())
    report['new_branch']='Released state-disjoint longcontext10..16s TF-Locoformer adaptation, fixed30percent mixture'
    report['adaptation_objective']='Final400 fusedSI/RNNT, same-state training concatenation; no residual remix'
    report['source_checkpoint_sha256']='Adapted DNS weights; provenance and training exclusion scope recorded separately'
    path.with_name(path.stem+'_provenance_corrected.json').write_text(json.dumps(report,indent=2))
print('ALL_GATES_COMPLETE',flush=True)
""")
exec(compile(source,str(root/'watch_loco_longcontext_gate_1012.py'),'exec'))
