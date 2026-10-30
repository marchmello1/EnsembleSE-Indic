"""CPU-only matched audit after both complete gate reports."""
import json,subprocess,time
from pathlib import Path
root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
gate=root/'rnnt800_gate_summary.json'
natural=Path('/mnt/nvme_disk1/asr_t2_locoformer_rnnt800_blend30_nat400_1010.json')
deadline=time.monotonic()+21600
while time.monotonic()<deadline:
    if gate.exists():
        try:result=json.loads(gate.read_text())
        except json.JSONDecodeError:
            time.sleep(10);continue
        if result['incremental_proxy']<=.05:
            print('REJECTED_AT_SYNTHETIC',result,flush=True);break
        if natural.exists():
            try:
                report=json.loads(natural.read_text())
                assert len(report['results'])==400
            except (json.JSONDecodeError,AssertionError):
                time.sleep(10);continue
            for baseline in ('official','frozen'):
                subprocess.run(['/mnt/nvme_disk1/mpsenet_env/bin/python','/mnt/nvme_disk1/audit_locoformer_1010.py','--checkpoint',str(root/'locoformer_rnnt_1010_s800.pt'),'--prefix','locoformer_rnnt800','--candidate-wavs','/mnt/nvme_disk1/t2_locoformer_fixed188_1010_rnnt800_blend30','--baseline',baseline,'--output',f'/mnt/nvme_disk1/t2_locoformer_rnnt800_matched_{baseline}_audit_1010.json'],check=True)
            break
    time.sleep(10)
else:raise TimeoutError('Observe the live training/gate handles; do not auto-restart.')
