import os
import subprocess
import time
from pathlib import Path

root = Path('/mnt/nvme_disk1/t2_gap_urgenet_1009')
pilot = Path('/mnt/nvme_disk1/urgent2026_t2pilot_1008')
validation = '/mnt/nvme_disk1/official_validation_20260907'
checkpoint = root / 'predictor_natadapt_1009_s800.pt'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='5', USE_TF='0')
previous = -1
deadline = time.monotonic() + 10800
while True:
    if checkpoint.exists():
        size = checkpoint.stat().st_size
        if size > 1000000 and size == previous:
            break
        previous = size
    if time.monotonic() > deadline:
        raise TimeoutError(str(checkpoint))
    time.sleep(10)
prefix = 'predictor_natadapt_s800'
direct = root / f'{prefix}_nat400_wavs'
blend = root / f'{prefix}_blend30_nat400_wavs'
commands = [
    ['/mnt/nvme_disk1/mpsenet_env/bin/python', '-u', 'eval_t2_gap_predictor_1009.py', '--limit', '400', '--checkpoint', str(checkpoint), '--prefix', prefix, '--natural'],
    ['/mnt/nvme_disk1/clearvoice_env/bin/python', str(pilot / 'eval_t2_gainmatched_blend.py'), validation, str(pilot / 'extended_s1600_ls70_nat400_wavs'), str(direct), str(root / f'{prefix}_gain_nat400.json'), '--output', str(blend), '--method', 'none', '--weight', '.3', '--natural'],
    ['/mnt/nvme_disk1/sravaani_venv/bin/python', '/tmp/eval_track2_sravaani_pair.py', validation, str(blend), '/mnt/nvme_disk1/asr_t2_predictor_natadapt_s800_blend30_nat400_1009.json', '--natural', '--limit', '0', '--require-extra-audio', str(blend), '--reuse-noisy', '/mnt/nvme_disk1/asr_t2_bsrnn_cont_s3000_nat400_w30_1008.transcripts.jsonl', '--batch-size', '8'],
]
for command in commands:
    print('RUN', command, flush=True)
    subprocess.run(command, cwd=root, env=env, check=True)
print('NATURAL_GATE_COMPLETE', flush=True)
