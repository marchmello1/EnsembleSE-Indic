import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from clearvoice import ClearVoice
from clearvoice.networks import SpeechModel

# The package otherwise chooses a physical GPU index despite CUDA visibility.
SpeechModel.get_free_gpu = lambda self: 0

def si(x, y):
    x = np.asarray(x, np.float64) - np.mean(x)
    y = np.asarray(y, np.float64) - np.mean(y)
    target = y * (np.dot(x, y)/(np.dot(y, y)+1e-12))
    return float(10*np.log10((np.sum(target**2)+1e-12)/(np.sum((x-target)**2)+1e-12)))

p = argparse.ArgumentParser()
p.add_argument('validation', type=Path)
p.add_argument('anchor', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--limit', type=int, default=24)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
m = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])
rows = []
for path in sorted(a.anchor.glob('*.wav'))[:a.limit]:
    noisy, sr = sf.read(a.validation/'syntheticNoiseAudio'/path.name, dtype='float32')
    anchor, ar = sf.read(path, dtype='float32')
    assert sr == ar == 16000 and noisy.ndim == anchor.ndim == 1
    streams = [np.asarray(x).reshape(-1) for x in m(noisy[None, :])]
    assert all(len(x) == len(noisy) and np.isfinite(x).all() for x in streams)
    correlations = [abs(float(np.dot(x, anchor)))/(float(np.linalg.norm(x)*np.linalg.norm(anchor))+1e-12) for x in streams]
    chosen = int(np.argmax(correlations))
    x = streams[chosen]
    gain = np.clip(np.dot(x, anchor)/(np.dot(x, x)+1e-12), .1, 10)
    x = x*gain
    # References enter only evaluation, never stream selection or gain estimation.
    clean, cr = sf.read(a.validation/'syntheticCleanRefAudio'/path.name)
    assert cr == sr and len(clean) == len(x) == len(anchor)
    scores = {'anchor': si(anchor, clean), 'separator': si(x, clean)}
    for w in [.1, .3, .7]:
        scores[str(w)] = si((1-w)*anchor+w*x, clean)
    sf.write(a.output/path.name, x/max(1., float(np.max(np.abs(x)))/.99), sr, subtype='PCM_16')
    rows.append({'id': path.stem, 'selected': chosen, 'correlations': correlations, 'scores': scores})
    print('DONE', len(rows), scores, flush=True)
result = {'clips': rows, 'means': {k: float(np.mean([r['scores'][k] for r in rows])) for k in rows[0]['scores']}, 'scope': 'Released validation only'}
(a.output/'result.json').write_text(json.dumps(result, indent=2))
print('FINAL', result['means'], flush=True)
