"""Evaluate saved continuation checkpoints; never restart the trainer."""
import os
import subprocess
import time
from pathlib import Path

root = Path('/mnt/nvme_disk1/urgent2026_t2pilot_1008')
validation = '/mnt/nvme_disk1/official_validation_20260907'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='5', USE_TF='0')
def run(cmd):
    print('RUN', cmd, flush=True)
    subprocess.run(cmd, cwd=root, env=env, check=True)

for step in (400, 1600):
    checkpoint = root/f'bsrnn_ls70_rnnt_extended_1009_s{step}.ckpt'
    deadline = time.monotonic()+10800
    while not checkpoint.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(checkpoint))
        time.sleep(15)
    # A checkpoint becomes visible before torch.save completes. Wait for stable size.
    previous = -1
    for _ in range(120):
        size = checkpoint.stat().st_size
        if size > 1000000 and size == previous:
            break
        previous = size
        time.sleep(5)
    else:
        raise RuntimeError('checkpoint did not settle')
    raw = root/f'extended_s{step}_hold188_wavs'
    blend = root/f'extended_s{step}_ls70_hold188_wavs'
    run(['/mnt/nvme_disk1/mpsenet_env/bin/python', '-u', 'eval_urgent_bsrnn_holdout.py',
         validation, str(root/'procnoise_s200_hold188_wavs'), str(checkpoint), str(raw), '--limit', '188'])
    run(['/mnt/nvme_disk1/clearvoice_env/bin/python', 'eval_t2_gainmatched_blend.py', validation,
         '/mnt/nvme_disk1/t2_publicnoise_hilr_s5000_fullclip_1008', str(raw),
         f'/mnt/nvme_disk1/t2_extended_s{step}_gain188_1009.json', '--output', str(blend), '--method', 'ls', '--weight', '.7'])
    run(['/mnt/nvme_disk1/sravaani_venv/bin/python', '/tmp/eval_track2_sravaani_pair.py',
         validation, str(blend), f'/mnt/nvme_disk1/asr_t2_extended_s{step}_hold188_1009.json',
         '--all-synthetic', '--limit', '0', '--require-extra-audio', str(blend), '--reuse-noisy',
         '/mnt/nvme_disk1/asr_t2_bsrnn_cont_s3000_w30_hold188_1008.transcripts.jsonl', '--batch-size', '8'])
    print('GATE_COMPLETE', step, flush=True)
