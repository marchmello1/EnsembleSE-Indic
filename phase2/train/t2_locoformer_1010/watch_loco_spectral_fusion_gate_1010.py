"""Evaluate both subsets and pooled matched gain after fixed final checkpoint."""
import os,subprocess,time
from pathlib import Path

root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
ck=root/'loco_spectral_fusion_1010_s1200.pt'
deadline=time.monotonic()+43200
while not ck.exists():
    if time.monotonic()>deadline:raise TimeoutError('Inspect cache/training jobs; no automatic restart')
    time.sleep(10)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',USE_TF='0',TRANSFORMERS_NO_TF='1',OMP_NUM_THREADS='4')
for natural in (False,True):
    subset='nat400' if natural else 'hold188'
    args=['/mnt/nvme_disk1/mpsenet_env/bin/python','-u','eval_loco_spectral_fusion_1010.py']
    if natural:args+=['--natural']
    subprocess.run(args,cwd=root,env=env,check=True)
    directory=f'/mnt/nvme_disk1/t2_loco_spectral1200_{subset}_1010'
    subprocess.run(['/mnt/nvme_disk1/sravaani_venv/bin/python','-u','/tmp/eval_track2_sravaani_pair.py',
                    '/mnt/nvme_disk1/official_validation_20260907',directory,
                    f'/mnt/nvme_disk1/asr_t2_loco_spectral1200_blend30_{subset}_1010.json',
                    '--natural' if natural else '--all-synthetic','--limit','0',
                    '--require-extra-audio',directory,'--reuse-noisy',
                    f'/mnt/nvme_disk1/asr_t2_extended_s1600_{subset}_1009.transcripts.jsonl',
                    '--batch-size','8'],cwd=root,env=env,check=True)
for baseline in ('official','frozen'):
    subprocess.run(['/mnt/nvme_disk1/mpsenet_env/bin/python','/mnt/nvme_disk1/audit_locoformer_1010.py',
                    '--checkpoint',str(ck),'--prefix','loco_spectral1200',
                    '--candidate-wavs','/mnt/nvme_disk1/t2_loco_spectral1200_hold188_1010',
                    '--baseline',baseline,'--output',
                    f'/mnt/nvme_disk1/t2_loco_spectral1200_matched_{baseline}_audit_1010.json'],check=True)
print('ALL_GATES_COMPLETE',flush=True)
