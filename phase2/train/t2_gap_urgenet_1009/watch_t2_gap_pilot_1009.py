import json
import os
import subprocess
import time
from pathlib import Path
root=Path('/mnt/nvme_disk1/t2_gap_urgenet_1009')
expected={'Adapter.pt':520092014,'DeWavLM-Omni.pt':1261995206,
          'PostNet.pt':19170170,'Predictor.pt':88469158,'Vocoder.pt':454994736}
gate=Path('/mnt/nvme_disk1/asr_t2_extended_s400_hold188_1009.json')
deadline=time.monotonic()+7200
while True:
    ready=all((root/'checkpoints'/name).exists() and (root/'checkpoints'/name).stat().st_size==size for name,size in expected.items())
    if ready and gate.exists():
        try:
            report=json.loads(gate.read_text())
            if report['clips']==188:
                break
        except (ValueError,KeyError):
            pass
    if time.monotonic()>deadline:
        raise TimeoutError('download or preceding ASR gate incomplete')
    time.sleep(15)
print('READY',flush=True)
subprocess.run(['/mnt/nvme_disk1/mpsenet_env/bin/python','-u','eval_t2_gap_urgenet_1009.py'],cwd=root,
               env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',USE_TF='0'),check=True)
