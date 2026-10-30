"""One fixed level-normalization rule after uniform neural enhancement."""
import os,subprocess
from pathlib import Path
import numpy as np,soundfile as sf
base=Path('/mnt/nvme_disk1')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',USE_TF='0',TRANSFORMERS_NO_TF='1',OMP_NUM_THREADS='4')
for natural in (False,True):
    subset='nat400' if natural else 'hold188'
    source=base/f't2_four_reproduction_{subset}_1010';target=base/f't2_four_peak90_{subset}_1010'
    target.mkdir(exist_ok=True);paths=sorted(source.glob('*.wav'));assert len(paths)==(400 if natural else 188)
    for path in paths:
        wave,sr=sf.read(path);assert sr==16000 and wave.ndim==1 and np.isfinite(wave).all()
        normalized=.9*wave/max(float(np.max(np.abs(wave))),1e-8)
        sf.write(target/path.name,normalized,sr,subtype='PCM_16')
    subprocess.run(['/mnt/nvme_disk1/sravaani_venv/bin/python','-u','/tmp/eval_track2_sravaani_pair.py',
        str(base/'official_validation_20260907'),str(target),str(base/f'asr_t2_four_peak90_blend30_{subset}_1010.json'),
        '--natural' if natural else '--all-synthetic','--limit','0','--require-extra-audio',str(target),
        '--reuse-noisy',str(base/f'asr_t2_extended_s1600_{subset}_1009.transcripts.jsonl'),'--batch-size','8'],env=env,check=True)
for baseline in ('official','frozen'):
    subprocess.run(['/mnt/nvme_disk1/mpsenet_env/bin/python',str(base/'audit_locoformer_1010.py'),
        '--checkpoint',str(base/'t2_gap_urgenet_1009/predictor_fulladapt_1010_s800.pt'),
        '--prefix','four_peak90','--candidate-wavs',str(base/'t2_four_peak90_hold188_1010'),
        '--baseline',baseline,'--output',str(base/f't2_four_peak90_matched_{baseline}_audit_1010.json')],check=True)
print('ALL_GATES_COMPLETE',flush=True)
