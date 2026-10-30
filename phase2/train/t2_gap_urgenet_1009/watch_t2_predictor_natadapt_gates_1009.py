import os
import subprocess
import time
from pathlib import Path

root=Path('/mnt/nvme_disk1/t2_gap_urgenet_1009')
pilot=Path('/mnt/nvme_disk1/urgent2026_t2pilot_1008')
validation='/mnt/nvme_disk1/official_validation_20260907'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',USE_TF='0')
def run(cmd):
    print('RUN',cmd,flush=True)
    subprocess.run(cmd,cwd=root,env=env,check=True)
for step in (200,800):
    checkpoint=root/f'predictor_natadapt_1009_s{step}.pt'
    deadline=time.monotonic()+10800
    previous=-1
    while True:
        if checkpoint.exists():
            size=checkpoint.stat().st_size
            if size>1000000 and size==previous:
                break
            previous=size
        if time.monotonic()>deadline:
            raise TimeoutError(str(checkpoint))
        time.sleep(10)
    prefix=f'predictor_natadapt_s{step}'
    direct=root/f'{prefix}_hold188_wavs'
    blend=root/f'{prefix}_blend30_hold188_wavs'
    run(['/mnt/nvme_disk1/mpsenet_env/bin/python','-u','eval_t2_gap_predictor_1009.py',
         '--limit','188','--checkpoint',str(checkpoint),'--prefix',prefix])
    run(['/mnt/nvme_disk1/clearvoice_env/bin/python',str(pilot/'eval_t2_gainmatched_blend.py'),
         validation,str(pilot/'extended_s1600_ls70_hold188_wavs'),str(direct),
         str(root/f'{prefix}_gain188.json'),'--output',str(blend),'--method','none','--weight','.3'])
    run(['/mnt/nvme_disk1/sravaani_venv/bin/python','/tmp/eval_track2_sravaani_pair.py',
         validation,str(blend),f'/mnt/nvme_disk1/asr_t2_{prefix}_blend30_hold188_1009.json',
         '--all-synthetic','--limit','0','--require-extra-audio',str(blend),'--reuse-noisy',
         '/mnt/nvme_disk1/asr_t2_bsrnn_cont_s3000_w30_hold188_1008.transcripts.jsonl','--batch-size','8'])
    print('GATE_COMPLETE',step,flush=True)
