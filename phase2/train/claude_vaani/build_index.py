"""Fingerprint index of all 3178 Phase-2 test inputs (used ONLY to exclude them from training data)."""
import glob, os, pickle, sys
import numpy as np, soundfile as sf
sys.path.insert(0, '/mnt/nvme_disk1/claude_vaani'); from fp import Index
ix = Index(); files = sorted(glob.glob('/mnt/nvme_disk1/indoml_phase2_input_0924/audio/*.wav'))
for i, f in enumerate(files):
    x, sr = sf.read(f, dtype='float32'); x = x.mean(1) if x.ndim > 1 else x
    ix.add(os.path.basename(f)[:-4], x, sr)
    if i % 500 == 0: print(i, len(files), len(ix.h), flush=True)
pickle.dump(ix.h, open('/mnt/nvme_disk1/claude_vaani/phase2_fp_index.pkl', 'wb'))
print('INDEX_DONE', len(files), 'hashes', len(ix.h))
