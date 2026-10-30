"""Released-clean diagnostic only; never render competition submissions."""
import json
import os
import subprocess
import time
from pathlib import Path

deadline = time.monotonic()+7200
while True:
    try:
        report=json.loads(Path('/mnt/nvme_disk1/asr_t2_gap_direct_hold188_1009.json').read_text())
        if report['clips']==188:
            break
    except (FileNotFoundError,ValueError,KeyError):
        pass
    if time.monotonic()>deadline:
        raise TimeoutError('preceding GAP ASR has not completed')
    time.sleep(15)
subprocess.run(['/mnt/nvme_disk1/sravaani_venv/bin/python','/tmp/eval_track2_sravaani_pair.py',
    '/mnt/nvme_disk1/official_validation_20260907',
    '/mnt/nvme_disk1/official_validation_20260907/syntheticCleanRefAudio',
    '/mnt/nvme_disk1/asr_t2_releasedclean_hold188_1009.json',
    '--all-synthetic','--limit','0','--require-extra-audio',
    '/mnt/nvme_disk1/urgent2026_t2pilot_1008/cont_s3000_hold188_wavs',
    '--reuse-noisy','/mnt/nvme_disk1/asr_t2_bsrnn_cont_s3000_w30_hold188_1008.transcripts.jsonl',
    '--batch-size','8'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',USE_TF='0'),check=True)
