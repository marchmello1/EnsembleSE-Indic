import os,subprocess,time
from pathlib import Path
root=Path('/mnt/nvme_disk1/t2_locoformer_1010');ck=root/'locoformer_remix_1011_s1600.pt'
deadline=time.monotonic()+43200
while not ck.exists():
    if time.monotonic()>deadline:raise TimeoutError('Inspect training handle; do not restart automatically')
    time.sleep(10)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',USE_TF='0',TRANSFORMERS_NO_TF='1',OMP_NUM_THREADS='4')
for natural in (False,True):
    suffix='nat400' if natural else 'hold188'
    args=['/mnt/nvme_disk1/mpsenet_env/bin/python','-u','pilot_locoformer_1010.py',
        '--limit','400' if natural else '188','--checkpoint',str(ck),'--tag','_remix1600']
    if natural:args+=['--natural']
    subprocess.run(args,cwd=root,env=env,check=True)
    directory='/mnt/nvme_disk1/t2_locoformer_'+('nat400' if natural else 'fixed188')+'_1010_remix1600_blend30'
    subprocess.run(['/mnt/nvme_disk1/sravaani_venv/bin/python','-u','/tmp/eval_track2_sravaani_pair.py',
        '/mnt/nvme_disk1/official_validation_20260907',directory,
        f'/mnt/nvme_disk1/asr_t2_locoformer_remix1600_blend30_{suffix}_1010.json',
        '--natural' if natural else '--all-synthetic','--limit','0','--require-extra-audio',directory,
        '--reuse-noisy',f'/mnt/nvme_disk1/asr_t2_extended_s1600_{suffix}_1009.transcripts.jsonl',
        '--batch-size','8'],cwd=root,env=env,check=True)
for baseline in ('official','frozen'):
    subprocess.run(['/mnt/nvme_disk1/mpsenet_env/bin/python','/mnt/nvme_disk1/audit_locoformer_1010.py',
        '--checkpoint',str(ck),'--prefix','locoformer_remix1600','--candidate-wavs',
        '/mnt/nvme_disk1/t2_locoformer_fixed188_1010_remix1600_blend30','--baseline',baseline,
        '--output',f'/mnt/nvme_disk1/t2_locoformer_remix1600_matched_{baseline}_audit_1010.json'],check=True)
print('ALL_GATES_COMPLETE',flush=True)
