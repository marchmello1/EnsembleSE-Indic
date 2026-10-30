"""Does ASR batching change transcripts? Re-transcribe submitted F4 WAVs with batch_size=1 and compare to the
submitted batch-of-8 transcripts."""
import json, random
from pathlib import Path
import torch
from transformers import AutoModel
from huggingface_hub import get_token
F = Path('/mnt/nvme_disk1/claude_f4_phase2_1014/final')
sub = {json.loads(l)['clip_id']: json.loads(l)['text'] for l in open(F/'transcripts.jsonl') if l.strip()}
names = sorted(F.glob('*.wav')); random.Random(0).shuffle(names); names = names[:400]
asr = AutoModel.from_pretrained('ARTPARK-IISc/SraVaani-1.0', trust_remote_code=True, token=get_token()).to('cuda').eval()
diff_clip, words, diff_words = 0, 0, 0
for p in names:
    h = asr.transcribe([str(p)], batch_size=1, return_hypotheses=True, timestamps=False)[0]
    t1 = str(getattr(h, 'text', h) or '').strip(); t8 = sub[p.stem]
    a, b = t8.split(), t1.split(); words += max(len(a), 1)
    if t1 != t8:
        diff_clip += 1; diff_words += sum(x != y for x, y in zip(a, b)) + abs(len(a)-len(b))
print(f'clips differing batch1 vs submitted(batch8): {diff_clip}/{len(names)} = {diff_clip/len(names):.3f}; word-level diff rate {diff_words/words:.4f}')
