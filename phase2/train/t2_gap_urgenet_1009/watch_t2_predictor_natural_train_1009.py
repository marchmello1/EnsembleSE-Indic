import os
import subprocess
import time
from pathlib import Path
cache=Path('/mnt/nvme_disk1/urgent2026_t2pilot_1008/neural_fusion_natural763_1009.pt')
deadline=time.monotonic()+7200
previous=-1
while True:
    if cache.exists():
        size=cache.stat().st_size
        if size>1000000 and size==previous:
            break
        previous=size
    if time.monotonic()>deadline:
        raise TimeoutError(str(cache))
    time.sleep(15)
print('CACHE_READY',cache.stat().st_size,flush=True)
subprocess.run(['/mnt/nvme_disk1/mpsenet_env/bin/python','-u','finetune_t2_predictor_rnnt_1009.py',
    '--checkpoint','/mnt/nvme_disk1/t2_gap_urgenet_1009/predictor_rnnt_1009_s800.pt',
    '--natural-cache',str(cache),'--steps','800','--lr','3e-6','--seed','1013',
    '--prefix','predictor_natadapt_1009'],cwd='/mnt/nvme_disk1/t2_gap_urgenet_1009',
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',USE_TF='0'),check=True)
