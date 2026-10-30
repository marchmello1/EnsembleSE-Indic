"""Stage 2: Locoformer trained so the SHIPPED ensemble output is better for SraVaani.
final = .5*best + .5*norm(loco(x)), where best (frozen MP/BSRNN/TFGrid part) is cached. Losses on final:
SI (synthetic), RNNT (all), preservation vs the current candidate's final (natural). Weights as Codex's
trainer (2/3/.2). Inference unchanged: audio-only, no lookup. Init: d32 Loco B s1600 (the candidate's)."""
import os
os.environ.setdefault('USE_TF', '0')
import random, sys
from pathlib import Path
import torch, yaml
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel
from standalone.tflocoformer_separator import TFLocoformerSeparator
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder
sys.path.insert(0, '/mnt/nvme_disk1/urgent2026_t2pilot_1008')
from finetune_mpsenet_rnnt import clean_text, rnnt_loss
from finetune_urgent_bsrnn_vaani import neg_sisdr
root = Path('/mnt/nvme_disk1/t2_locoformer_1010')
seed = 1052; rng = random.Random(seed); torch.manual_seed(seed)
torch.cuda.set_per_process_memory_fraction(.5)
cfg = yaml.safe_load((root/'egs2/dns_ins20/enh1/exp/enh_train_enh_tflocoformer_raw/config.yaml').read_text())
model = TFLocoformerSeparator(**cfg['separator_conf'])
model.load_state_dict(torch.load('/mnt/nvme_disk1/claude_d32/loco_d32b_1013_s1600.pt', map_location='cpu', weights_only=False)['model'], strict=True)
model = model.cuda().train()
for b in model.blocks:
    o = b.forward
    def f(*a, _o=o, **k): return checkpoint(_o, *a, use_reentrant=False, **k)
    b.forward = f
enc = STFTEncoder(**cfg['encoder_conf']).cuda().eval(); dec = STFTDecoder(**cfg['decoder_conf']).cuda().eval()
asr = AutoModel.from_pretrained('/mnt/nvme_disk1/sravaani_model', trust_remote_code=True).cuda().eval(); asr._ensure_loaded()
for m in (asr.encoder, asr.decoder_joint):
    for q in m.parameters(): q.requires_grad_(False)
tok = asr._get_tokenizer()
data = [d for d in torch.load('/mnt/nvme_disk1/claude_d32/e2e_cache_1013.pt', weights_only=False)]
data = [(d, tok.encode(clean_text(d['transcript']))) for d in data]; data = [(d, t) for d, t in data if t]
print('E2E items', len(data), flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=3e-6, weight_decay=1e-6)
for step in range(1, 801):
    d, tokens = rng.choice(data)
    x = d['x'][None].cuda(); best = d['best'][None].cuda(); L = x.shape[-1]
    lens = torch.tensor([L], device='cuda'); std = x.std(1, keepdim=True).clamp_min(1e-8)
    z, _ = enc(x/std, lens); z = model(z)[:, 0]; lo, _ = dec(z, lens); lo = lo*std
    lo = lo/(lo.abs().amax(-1, keepdim=True)/.99).clamp_min(1)
    final = .5*best + .5*lo
    recognition = rnnt_loss(asr, final, tokens)
    if d['natural']:
        ff = d['frozen_final'][None].cuda()
        acoustic = final.new_zeros(()); preserve = (final-ff).abs().mean()/ff.abs().mean().clamp_min(1e-4)
    else:
        acoustic = neg_sisdr(final, d['clean'][None].cuda()).clamp(min=-40); preserve = final.new_zeros(())
    loss = 2*acoustic + 3*recognition + .2*preserve
    assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True); loss.backward(); g = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
    assert torch.isfinite(g) and float(g) > 0; opt.step()
    if step == 1 or step % 10 == 0: print('STEP', step, 'LOSS', float(loss.detach()), 'RNNT', float(recognition.detach()), 'NAT', d['natural'], 'GRAD', float(g), flush=True)
    if step % 200 == 0:
        dst = Path('/mnt/nvme_disk1/claude_d32')/f'loco_e2e_1013_s{step}.pt'; tmp = dst.with_suffix('.partial')
        torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, cfg=cfg, step=step, seed=seed,
                        objective='end-to-end on shipped mix .5*best+.5*loco: SI2 RNNT3 preserve.2',
                        inference='Uniform audio-only; no lookup'), tmp)
        tmp.replace(dst); print('SAVED', dst, flush=True)
