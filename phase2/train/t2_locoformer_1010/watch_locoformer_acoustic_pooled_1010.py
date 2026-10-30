"""Complete CPU audit of the acoustic experiment once natural ASR finishes."""
import json
import subprocess
import time
from pathlib import Path

report = Path('/mnt/nvme_disk1/asr_t2_locoformer_acoustic1600_blend30_nat400_1010.json')
deadline = time.monotonic() + 21600
while time.monotonic() < deadline:
    if report.exists():
        try:
            data = json.loads(report.read_text())
            assert len(data['results']) == 400
        except (json.JSONDecodeError, AssertionError):
            time.sleep(10)
            continue
        for baseline in ('official', 'frozen'):
            subprocess.run([
                '/mnt/nvme_disk1/mpsenet_env/bin/python',
                '/mnt/nvme_disk1/audit_locoformer_1010.py',
                '--checkpoint', '/mnt/nvme_disk1/t2_locoformer_1010/locoformer_acoustic_1010_s1600.pt',
                '--prefix', 'locoformer_acoustic1600',
                '--candidate-wavs', '/mnt/nvme_disk1/t2_locoformer_fixed188_1010_acoustic1600_blend30',
                '--baseline', baseline,
                '--output', f'/mnt/nvme_disk1/t2_locoformer_acoustic1600_matched_{baseline}_audit_1010.json',
            ], check=True)
        break
    time.sleep(10)
else:
    raise TimeoutError('Inspect natural job; do not auto-restart.')
