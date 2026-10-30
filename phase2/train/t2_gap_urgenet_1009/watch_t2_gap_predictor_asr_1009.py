"""Gate fixed GAP outputs on released validation, after rendering completes."""
import json
import os
import subprocess
import time
from pathlib import Path

root = Path('/mnt/nvme_disk1/t2_gap_urgenet_1009')
pilot = Path('/mnt/nvme_disk1/urgent2026_t2pilot_1008')
validation = '/mnt/nvme_disk1/official_validation_20260907'
deadline = time.monotonic() + 7200
while True:
    try:
        report = json.loads((root/'predictor_hold188_report.json').read_text())
        if len(report['rows']) == 188:
            break
    except (FileNotFoundError, ValueError, KeyError):
        pass
    if time.monotonic() > deadline:
        raise TimeoutError('GAP render not complete')
    time.sleep(15)
# Training owns GPU 4 until its process exits. Never compete for that device.
while subprocess.run(['ps','-p','1795444'],stdout=subprocess.DEVNULL).returncode == 0:
    if time.monotonic() > deadline:
        raise TimeoutError('training still owns GPU 4')
    time.sleep(15)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='5', USE_TF='0')
blend = root/'predictor_blend70_hold188_wavs'
subprocess.run(['/mnt/nvme_disk1/clearvoice_env/bin/python',str(pilot/'eval_t2_gainmatched_blend.py'),
    validation,'/mnt/nvme_disk1/t2_publicnoise_hilr_s5000_fullclip_1008',str(root/'predictor_hold188_wavs'),
    str(root/'predictor_gain188.json'),'--output',str(blend),'--method','ls','--weight','.7'],env=env,check=True)
for label, directory in [('blend70',blend),('direct',root/'predictor_hold188_wavs')]:
    subprocess.run(['/mnt/nvme_disk1/sravaani_venv/bin/python','/tmp/eval_track2_sravaani_pair.py',
        validation,str(directory),f'/mnt/nvme_disk1/asr_t2_gap_predictor_{label}_hold188_1009.json',
        '--all-synthetic','--limit','0','--require-extra-audio',str(directory),'--reuse-noisy',
        '/mnt/nvme_disk1/asr_t2_bsrnn_cont_s3000_w30_hold188_1008.transcripts.jsonl','--batch-size','8'],
        env=env,check=True)
    print('ASR_GATE_COMPLETE',label,flush=True)
