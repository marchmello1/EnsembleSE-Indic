"""Download ~120 main-Vaani shards spread over districts, EXCLUDING the six held-out evaluation states."""
import random, collections
from huggingface_hub import HfApi, hf_hub_download
HELD = {'AndhraPradesh', 'Chhattisgarh', 'Karnataka', 'Meghalaya', 'Sikkim', 'WestBengal'}
api = HfApi(); R = 'ARTPARK-IISc/Vaani'; OUT = '/mnt/nvme_disk1/vaani_main'
info = api.dataset_info(R, files_metadata=True)
fs = [(s.rfilename, s.size or 0) for s in info.siblings if s.rfilename.endswith('.parquet') and s.rfilename.count('/') == 3]
fs = [f for f in fs if f[0].split('/')[1] not in HELD and f[1] < 800e6]
by = collections.defaultdict(list)
for f, z in fs: by[tuple(f.split('/')[1:3])].append((f, z))
rng = random.Random(11); keys = sorted(by); rng.shuffle(keys); pick = []
for k in keys:
    c = by[k]; rng.shuffle(c); pick.append(c[0])
    if len(pick) >= 120: break
print('PICK', len(pick), 'districts', len(keys), 'GB', round(sum(z for _, z in pick)/1e9, 1),
      'states', len({f.split('/')[1] for f, _ in pick}), flush=True)
for i, (f, _) in enumerate(pick):
    try:
        hf_hub_download(R, f, repo_type='dataset', local_dir=OUT); print('GOT', i+1, f, flush=True)
    except Exception as e:
        print('FAIL', f, type(e).__name__, flush=True)
print('DL_MAIN_DONE')
