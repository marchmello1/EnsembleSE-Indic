"""TF-GridNet (deployed predictor_fulladapt_s800) fine-tuned STANDALONE on d32 reused-pool remixes.
Same data cache, state exclusions, d32 event construction and RNNT objective as claude's Loco d32 run;
model swapped to TF-GridNet. Inference stays audio-only (no lookup)."""
import os
os.environ.setdefault('USE_TF', '0')
import json, random, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel
sys.path.insert(0, '/mnt/nvme_disk1/t2_gap_urgenet_1009')
sys.path.insert(0, '/mnt/nvme_disk1/urgent2026_t2pilot_1008')
from finetune_mpsenet_rnnt import clean_text, rnnt_loss
from finetune_urgent_bsrnn_vaani import neg_sisdr
from models.predictor.tfgridnet import TFGridNet

cache = Path('/mnt/nvme_disk1/t2_locoformer_1010/fusion_training_cache')
manifest = json.loads((cache/'manifest.json').read_text())
seed = 1033; rng = random.Random(seed); torch.manual_seed(seed)
torch.cuda.set_per_process_memory_fraction(.5)
ck = torch.load('/mnt/nvme_disk1/t2_four_neural_codepkg_1011/marchmello01/checkpoints/predictor_fulladapt_s800.pt', map_location='cpu', weights_only=False)
model = TFGridNet(**ck['cfg']); model.load_state_dict(ck['model'], strict=True); model = model.cuda().train()
for block in model.blocks:
    orig = block.forward
    def fwd(*a, _o=orig, **k): return checkpoint(_o, *a, use_reentrant=False, **k)
    block.forward = fwd
asr = AutoModel.from_pretrained('/mnt/nvme_disk1/sravaani_model', trust_remote_code=True).cuda().eval(); asr._ensure_loaded()
for m in (asr.encoder, asr.decoder_joint):
    for q in m.parameters(): q.requires_grad_(False)
tok = asr._get_tokenizer(); pools = [[], []]
_z = np.load('/mnt/nvme_disk1/t2_d32_bank.npz', allow_pickle=True)
D32 = [torch.from_numpy(np.asarray(_z[k], np.float32)) for k in _z.keys() if np.abs(_z[k]).max() > 0]
for ident in manifest['ids']:
    row = torch.load(cache/(ident+'.pt'), map_location='cpu', weights_only=False)
    assert row['state'] not in manifest['heldout_states']
    t = tok.encode(clean_text(row['transcript']))
    if t and len(row['noisy']) <= 10*16000: pools[int(row['natural'])].append((row, t))
print('POOLS', [len(p) for p in pools], flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=3e-6, weight_decay=1e-6)
for step in range(1, 801):
    natural = rng.random() < .35
    row, tokens = rng.choice(pools[int(natural)])
    x = row['noisy'][None].cuda(); target = None
    if not natural:
        clean = row['clean'][None].cuda(); n = clean.shape[-1]
        if rng.random() < .7:
            act = clean[clean.abs() > .02*clean.abs().max()]; srms = act.square().mean().sqrt().clamp_min(1e-6)
            res = torch.zeros_like(clean); placed = 0; guard = 0
            while placed < .34*n and guard < 60:
                guard += 1; a = D32[rng.randrange(len(D32))].cuda(); L = min(len(a), n); p = rng.randrange(0, max(1, n-L))
                res[0, p:p+L] += a[:L]*(srms*10**(rng.gauss(9., 3.)/20)/a[:L].square().mean().sqrt().clamp_min(1e-9)); placed += L
            x = clean + res
            pk = (x.abs().max()/.95).clamp_min(1); x = x/pk; clean = clean/pk
        target = clean
    out = model(x, sr=16000)
    out = out/out.abs().amax(-1, keepdim=True).clamp_min(1e-8)*x.abs().amax(-1, keepdim=True)
    recognition = rnnt_loss(asr, out, tokens)
    acoustic = out.new_zeros(()) if natural else neg_sisdr(out, target).clamp(min=-40)
    loss = 2*acoustic + 3*recognition
    assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True); loss.backward(); g = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
    assert torch.isfinite(g) and float(g) > 0; opt.step()
    if step == 1 or step % 10 == 0: print('STEP', step, 'LOSS', float(loss.detach()), 'RNNT', float(recognition.detach()), 'NAT', natural, 'GRAD', float(g), flush=True)
    if step % 200 == 0:
        d = Path('/mnt/nvme_disk1/claude_d32')/f'tfgrid_d32_1013_s{step}.pt'; tmp = d.with_suffix('.partial')
        torch.save(dict(cfg=ck['cfg'], model={k: v.detach().cpu() for k, v in model.state_dict().items()}, step=step, seed=seed,
                        training_ids=manifest['ids'], heldout_states=manifest['heldout_states'],
                        objective=dict(si=2, rnnt=3, remix_prob=.7, noise='d32 events +9dB(3) cov .34', natural_prob=.35),
                        inference='Uniform audio-only; no noise lookup'), tmp)
        tmp.replace(d); print('SAVED', d, flush=True)
