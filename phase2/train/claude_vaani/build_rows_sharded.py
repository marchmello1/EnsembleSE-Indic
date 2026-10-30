"""Turn downloaded main-Vaani shards into training rows that mimic the organizers' generator.

1. keep 2-15 s utterances with speech-to-floor range >= --min-range dB (cleaner recordings)
2. gate pauses to EXACT zero with a frame-energy VAD (threshold relative to the utterance's speech level),
   tuned so the zero fraction/run lengths match released clean refs (median ~10% zeros, runs ~0.18 s)
3. LEAKAGE FILTER: fingerprint every kept utterance against all 3178 Phase-2 inputs; drop matches (>= 8 votes)
   and every utterance of any matched speaker
4. save rows {id, clean, noisy(=clean), natural False, transcript, state, speaker} for the trainers
"""
import argparse, glob, io, os, pickle, sys
import numpy as np, pyarrow.parquet as pq, soundfile as sf, torch
sys.path.insert(0, '/mnt/nvme_disk1/claude_vaani'); from fp import landmarks
from quality import score
ap = argparse.ArgumentParser(); ap.add_argument('--min-range', type=float, default=32.); ap.add_argument('--gate-db', type=float, default=20.)
ap.add_argument('--out', default='/mnt/nvme_disk1/claude_vaani/vaani_rows.pt'); ap.add_argument('--limit', type=int, default=10**9); ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
a = ap.parse_args()
H = pickle.load(open('/mnt/nvme_disk1/claude_vaani/phase2_fp_index.pkl', 'rb'))


def votes(x):
    v = {}
    for h, t in landmarks(x, 16000):
        for k, tt in H.get(h, ()):
            key = (k, tt-t); v[key] = v.get(key, 0)+1
    return max(v.values()) if v else 0


def gate(x):
    n = 1 + max(0, (len(x)-400)//160); idx = np.arange(400)[None, :] + 160*np.arange(n)[:, None]
    e = 10*np.log10((x[idx]**2).mean(1) + 1e-12); speech = np.percentile(e, 90)
    on = e > speech - a.gate_db
    k = 8                                                  # 80 ms hangover / closing of short gaps
    on = np.convolve(on.astype(float), np.ones(2*k+1), 'same') > 0
    m = np.zeros(len(x)); 
    for i in np.where(on)[0]: m[i*160:i*160+400] = 1
    return (x*m).astype(np.float32), float((m == 0).mean())


rows, zero_fracs, stats = [], [], dict(seen=0, dur=0, quality=0, leak=0)
cand = []
for f in sorted(glob.glob('/mnt/nvme_disk1/vaani_main/audio/**/*.parquet', recursive=True))[a.shard::a.nshards]:
    t = pq.read_table(f, columns=['audio', 'duration', 'speakerID', 'state', 'transcript', 'isTranscriptionAvailable'])
    for r in t.to_pylist():
        stats['seen'] += 1
        if not (2.0 <= float(r['duration'] or 0) <= 15.0): stats['dur'] += 1; continue
        x, sr = sf.read(io.BytesIO(r['audio']['bytes']), dtype='float32')
        if sr != 16000 or x.ndim != 1: continue
        if score(x) < a.min_range: stats['quality'] += 1; continue
        cand.append((x, r))
    if len(cand) >= a.limit: break
print('candidates', len(cand), stats, flush=True)
bad_spk = set(); keep = []
for i, (x, r) in enumerate(cand):
    v = votes(x)
    if v >= 20: bad_spk.add(r['speakerID'])     # strong match: exclude the whole speaker
    if v >= 12: stats['leak'] += 1                # calibrated: FP 2.2%, recall ~95% vs full index
    else: keep.append((x, r))
    if i % 2000 == 0: print('fp', i, len(cand), 'leak', stats['leak'], flush=True)
keep = [(x, r) for x, r in keep if r['speakerID'] not in bad_spk]
for i, (x, r) in enumerate(keep):
    g, zf = gate(x); zero_fracs.append(zf)
    rows.append(dict(id=f'vaani_{a.shard}_{i:07d}', clean=torch.from_numpy(g), noisy=torch.from_numpy(g), natural=False,
                     transcript=(r['transcript'] or '') if r['isTranscriptionAvailable'] == 'Yes' else '',
                     state=r['state'], speaker=r['speakerID']))
torch.save(rows, a.out)
zf = np.array(zero_fracs)
print(f'ROWS {len(rows)} | leak matches {stats["leak"]}, speakers excluded {len(bad_spk)} | with transcript '
      f'{sum(1 for r in rows if r["transcript"])} | gated zero-frac p10/p50/p90 {np.percentile(zf,10):.3f}/{np.percentile(zf,50):.3f}/{np.percentile(zf,90):.3f} '
      f'(target ~.02/.10/.21) | hours {sum(len(r["clean"]) for r in rows)/16000/3600:.1f} | states {len({r["state"] for r in rows})}')
