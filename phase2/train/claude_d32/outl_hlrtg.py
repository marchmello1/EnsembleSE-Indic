import io, json, zipfile, numpy as np, soundfile as sf
z = zipfile.ZipFile('/mnt/nvme_disk1/indoml_track2_PHASE2_HLRTG_claude_1014.zip')
f = zipfile.ZipFile('/mnt/nvme_disk1/indoml_track2_PHASE2_D32_TTA4_claude_1013.zip')
def si(a, b):
    m = min(len(a), len(b)); a, b = a[:m]-a[:m].mean(), b[:m]-b[:m].mean()
    s = b*np.dot(a, b)/(np.dot(b, b)+1e-12); e = a-s
    return 10*np.log10((np.dot(s, s)+1e-12)/(np.dot(e, e)+1e-12))
tz = {json.loads(l)['clip_id']: json.loads(l)['text'] for l in z.read('transcripts.jsonl').decode().splitlines() if l.strip()}
tf = {json.loads(l)['clip_id']: json.loads(l)['text'] for l in f.read('transcripts.jsonl').decode().splitlines() if l.strip()}
rows = []
for n in z.namelist():
    if not n.endswith('.wav'): continue
    y = sf.read(io.BytesIO(z.read(n)), dtype='float64')[0]; o = sf.read(io.BytesIO(f.read(n)), dtype='float64')[0]
    s = si(o, y)
    if s < 10: rows.append((s, n, y, o))
rows.sort(key=lambda r: r[0])
for s, n, y, o in rows:
    x = sf.read('/mnt/nvme_disk1/indoml_phase2_input_0924/audio/'+n, dtype='float64')[0]; c = n[:-4]
    wn, wo = len(tz[c].split()), len(tf[c].split())
    same = sum(a == b for a, b in zip(tz[c].split(), tf[c].split()))
    print(f'{s:6.2f} dB  rms in/new/old {np.sqrt(np.mean(x**2)):.4f}/{np.sqrt(np.mean(y**2)):.4f}/{np.sqrt(np.mean(o**2)):.4f}  '
          f'corr(new,in) {np.corrcoef(y,x)[0,1]:+.2f} corr(old,in) {np.corrcoef(o,x)[0,1]:+.2f}  words new/old {wn}/{wo} aligned-same {same}')
