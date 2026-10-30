"""Sequential exact PCM/mandated-ASR gates after final training checkpoint."""
import json,os,subprocess,time
from pathlib import Path
root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
ck=root/'locoformer_rnnt_1010_s800.pt'
deadline=time.monotonic()+21600;size=-1
while time.monotonic()<deadline:
    if ck.exists():
        current=ck.stat().st_size
        if current>1000000 and current==size:break
        size=current
    time.sleep(10)
else:raise TimeoutError('Inspect training handle, do not restart automatically')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',USE_TF='0',OMP_NUM_THREADS='4')
def evaluate(natural):
    n=400 if natural else 188
    tag='nat' if natural else 'fixed'
    output=Path('/mnt/nvme_disk1')/f't2_locoformer_{tag}{n}_1010_rnnt800'
    cmd=['/mnt/nvme_disk1/mpsenet_env/bin/python','-u','pilot_locoformer_1010.py','--limit',str(n),'--checkpoint',str(ck),'--tag','_rnnt800']
    if natural:cmd+=['--natural']
    subprocess.run(cmd,cwd=root,env=env,check=True)
    subset='nat400' if natural else 'hold188'
    report=Path('/mnt/nvme_disk1')/f'asr_t2_locoformer_rnnt800_blend30_{subset}_1010.json'
    subprocess.run(['/mnt/nvme_disk1/sravaani_venv/bin/python','-u','/tmp/eval_track2_sravaani_pair.py','/mnt/nvme_disk1/official_validation_20260907',str(output)+'_blend30',str(report),'--natural' if natural else '--all-synthetic','--limit','0','--require-extra-audio',str(output)+'_blend30','--reuse-noisy',f'/mnt/nvme_disk1/asr_t2_extended_s1600_{subset}_1009.transcripts.jsonl','--batch-size','8'],cwd=root,env=env,check=True)
    return json.loads((output/'result.json').read_text()),json.loads(report.read_text())
acoustic,asr=evaluate(False)
si=acoustic['means']['blend_0.3'];errors=asr['errors']['enhanced']
gain=si-18.4525568260+100*(654-errors)/7497
summary={'scope':'released188 synthetic gate only, not official estimate','si':si,'errors':errors,'incremental_proxy':gain}
(root/'rnnt800_gate_summary.json').write_text(json.dumps(summary,indent=2))
print('SYN_GATE',summary,flush=True)
if gain>.05:
    evaluate(True)
    print('NATURAL_GATES_DONE',flush=True)
else:print('REJECT_SYNTHETIC_GATE',flush=True)
