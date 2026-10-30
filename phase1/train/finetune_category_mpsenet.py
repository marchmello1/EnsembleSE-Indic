#!/usr/bin/env python3
"""Fine-tune MP-SENet conditioned on per-category event masks.

This uses only supplied event metadata and paired noisy/clean training audio.
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

CATEGORIES = [
    "human_non_speech", "vehicle_traffic", "animal", "baby_child",
    "phone_signal_alarm", "singing_music", "appliance_machine", "other",
]
CAT_INDEX = {name: i for i, name in enumerate(CATEGORIES[:-1])}
TOP_TAGS = [
    "<horn>", "[breathing]", "<bird sound>", "<bird squawking>",
    "<child talking>", "[inhaling]", "<child yelling>", "<barking>",
    "<vehicle noise>", "<animal sound>", "<machine noise>", "<dog barking>",
    "<phone ringing>", "<ringing>", "[lip smacking]", "<music>",
    "<child crying>", "<baby talking>", "<bird chirping>", "<bell sound>",
    "other",
]


def category_masks(events, n, sr=16000):
    masks = np.zeros((len(CATEGORIES), n), np.float32)
    for event in events:
        left = max(0, int(round(float(event["start"]) * sr)))
        right = min(n, int(round(float(event["end"]) * sr)))
        idx = CAT_INDEX.get(str(event.get("category", "")), len(CATEGORIES) - 1)
        masks[idx, left:right] = 1.0
    return masks


def tag_masks(events, n, sr=16000):
    masks = np.zeros((len(TOP_TAGS), n), np.float32)
    index = {name: i for i, name in enumerate(TOP_TAGS[:-1])}
    for event in events:
        left = max(0, int(round(float(event["start"]) * sr)))
        right = min(n, int(round(float(event["end"]) * sr)))
        idx = index.get(str(event.get("tag", "")), len(TOP_TAGS) - 1)
        masks[idx, left:right] = 1.0
    return masks


def perturb_events(events, rng, shift, endpoint_jitter, scale_min, scale_max):
    """Simulate detector boundary errors using only training annotations.

    The event identity is retained (the transcript usually supplies count/tag),
    while centre and duration are distorted to make enhancement robust to the
    imperfect Track-1 timeline used at inference.
    """
    output = []
    for event in events:
        start, end = float(event["start"]), float(event["end"])
        centre = .5 * (start + end) + rng.uniform(-shift, shift)
        duration = max(.02, end - start) * rng.uniform(scale_min, scale_max)
        left = centre - .5 * duration + rng.uniform(-endpoint_jitter, endpoint_jitter)
        right = centre + .5 * duration + rng.uniform(-endpoint_jitter, endpoint_jitter)
        copied = dict(event)
        copied["start"] = max(0., min(left, right - .02))
        copied["end"] = max(copied["start"] + .02, right)
        output.append(copied)
    return output


def sample_crop(row, length, rng, mask_fn, perturb=None):
    noisy = np.asarray(row["waveform"], np.float32).reshape(-1)
    clean = np.asarray(row["clean"], np.float32).reshape(-1)
    n = min(len(noisy), len(clean)); noisy, clean = noisy[:n], clean[:n]
    events = row.get("events", [])
    conditioning_events = perturb(events, rng) if perturb else events
    masks = mask_fn(conditioning_events, n)
    if not events:
        # Some released-pair caches omit timestamp metadata.  The paired
        # residual is available during training, so derive a conservative
        # union mask and place it in the generic/other conditioning channel.
        residual = np.abs(noisy - clean)
        if float(residual.max(initial=0.)) > 0:
            smooth = np.convolve(residual, np.ones(160, np.float32) / 160, mode="same")
            active = smooth > max(3e-5, .05 * float(smooth.max(initial=0.)))
            active = np.convolve(active.astype(np.float32), np.ones(801, np.float32), mode="same") > 0
            masks[-1] = active.astype(np.float32)
    if n <= length:
        pad = length - n
        return np.pad(noisy, (0, pad)), np.pad(clean, (0, pad)), np.pad(masks, ((0, 0), (0, pad)))
    if events and rng.random() < .9:
        event = rng.choice(events)
        center = .5 * (float(event["start"]) + float(event["end"])) * 16000
        start = int(center - length / 2 + rng.uniform(-.15, .15) * length)
    else:
        start = rng.randrange(n - length + 1)
    start = max(0, min(n - length, start))
    return noisy[start:start+length], clean[start:start+length], masks[:, start:start+length]


def remix_crop(rows, length, rng, mask_fn, perturb, snr_min, snr_max,
               speech_rows=None, noise_rows=None, public_noise_prob=0.):
    """Remix a residual event onto unrelated clean speech at a random SNR."""
    _, speech, _ = sample_crop(rng.choice(speech_rows or rows), length, rng, mask_fn, None)
    noise_pool = (noise_rows if noise_rows and rng.random() < public_noise_prob else rows)
    noisy_n, clean_n, masks = sample_crop(rng.choice(noise_pool), length, rng, mask_fn, perturb)
    noise = noisy_n - clean_n
    cp = float(np.mean(speech * speech) + 1e-9)
    npow = float(np.mean(noise * noise) + 1e-9)
    snr = rng.uniform(snr_min, snr_max)
    scale = float(np.clip(np.sqrt(cp / (npow * 10 ** (snr / 10))), .05, 12.0))
    mixture = speech + scale * noise
    peak = max(1., float(np.max(np.abs(mixture)) / .98))
    return mixture / peak, speech / peak, masks


def neg_sisdr(y, c):
    y = y - y.mean(-1, keepdim=True); c = c - c.mean(-1, keepdim=True)
    scale = (y*c).sum(-1, keepdim=True) / (c.square().sum(-1, keepdim=True) + 1e-8)
    target = scale*c; error = y-target
    return (-10*torch.log10((target.square().sum(-1)+1e-8)/(error.square().sum(-1)+1e-8))).mean()


def multires_spectral_loss(y, clean):
    """Penalize speech harmonics and transients at several time resolutions."""
    result = y.new_zeros(())
    for fft, hop in ((256, 64), (512, 128), (1024, 256)):
        window = torch.hann_window(fft, device=y.device, dtype=y.dtype)
        pred = torch.stft(y, fft, hop, fft, window=window, return_complex=True).abs()
        truth = torch.stft(clean, fft, hop, fft, window=window, return_complex=True).abs()
        result = result + (pred.sub(truth).abs().mean() / truth.mean().clamp_min(1e-4))
    return result / 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("train_cache", type=Path); p.add_argument("repo", type=Path)
    p.add_argument("checkpoint", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--steps", type=int, default=5000); p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seconds", type=float, default=3); p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--conditioning", choices=["category", "tag"], default="category")
    p.add_argument("--secondary-cache", type=Path)
    p.add_argument("--speech-cache", type=Path,
                   help="Additional training-only event-free speech for residual remixes")
    p.add_argument("--noise-cache", type=Path,
                   help="Additional public event-noise residuals for residual remixes")
    p.add_argument("--public-noise-prob", type=float, default=0.,
                   help="Chance a remix draws a public noise residual instead of paired-training residual")
    p.add_argument("--full-mask-prob", type=float, default=0.,
                   help="Chance to train a whole-clip generic enhancement condition")
    p.add_argument("--secondary-prob", type=float, default=.5)
    p.add_argument("--gated-loss", action="store_true")
    p.add_argument("--gate-pad", type=float, default=.125); p.add_argument("--gate-fade", type=float, default=.025)
    p.add_argument("--mask-shift", type=float, default=0.,
                   help="Maximum random event-centre shift in seconds")
    p.add_argument("--mask-endpoint-jitter", type=float, default=0.,
                   help="Independent onset/offset perturbation in seconds")
    p.add_argument("--mask-scale-min", type=float, default=1.)
    p.add_argument("--mask-scale-max", type=float, default=1.)
    p.add_argument("--sravaani-dir", type=Path)
    p.add_argument("--asr-feature-weight", type=float, default=0.)
    p.add_argument("--remix-prob", type=float, default=0.)
    p.add_argument("--high-coverage-prob", type=float, default=0.,
                   help="Probability of drawing from training clips whose events cover >70%% of audio")
    p.add_argument("--stft-weight", type=float, default=0.)
    p.add_argument("--snr-min", type=float, default=-10.)
    p.add_argument("--snr-max", type=float, default=25.)
    p.add_argument("--seed", type=int, default=20260920); a = p.parse_args()
    conditions = CATEGORIES if a.conditioning == "category" else TOP_TAGS
    mask_fn = category_masks if a.conditioning == "category" else tag_masks
    sys.path.insert(0, str(a.repo))
    from env import AttrDict
    from dataset import mag_pha_istft, mag_pha_stft
    from models.model import DenseEncoder, MPNet

    class CategoryMPNet(MPNet):
        def __init__(self, cfg):
            super().__init__(cfg); self.dense_encoder = DenseEncoder(cfg, in_channel=2+len(conditions))
        def forward(self, amp, phase, frame_masks):
            maps = frame_masks[:, :, :, None].expand(-1, -1, -1, amp.shape[1])
            features = torch.stack((amp, phase), -1).permute(0, 3, 2, 1)
            features = self.dense_encoder(torch.cat((features, maps), 1))
            for block in self.TSTransformer: features = block(features)
            out_amp = amp * self.mask_decoder(features); out_phase = self.phase_decoder(features)
            return out_amp, out_phase

    h = AttrDict(json.loads((a.repo/"best_ckpt/config.json").read_text()))
    model = CategoryMPNet(h).cuda(); source = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    source = source.get("generator", source.get("model", source)); target = model.state_dict()
    for key, value in source.items():
        if key == "dense_encoder.dense_conv_1.0.weight":
            if value.shape == target[key].shape:
                target[key] = value
            else:
                target[key][:, :2] = value[:, :2]
                target[key][:, 2:].zero_()
            if value.shape[1] == 3 and target[key].shape[1] > 3:
                # Replication preserves the binary-mask initialization when one
                # category is active, while allowing categories to specialize.
                target[key][:, 2:] = value[:, 2:3]
        elif key in target and target[key].shape == value.shape:
            target[key] = value
    model.load_state_dict(target); model.train()
    asr_encoder = None
    if a.asr_feature_weight:
        if not a.sravaani_dir:
            p.error("--sravaani-dir is required with --asr-feature-weight")
        from sravaani_features import SraVaaniFeatures
        asr_encoder = SraVaaniFeatures(a.sravaani_dir).cuda().eval()
    rows = torch.load(a.train_cache, map_location="cpu", weights_only=False)
    secondary = (torch.load(a.secondary_cache, map_location="cpu", weights_only=False)
                 if a.secondary_cache else None)
    speech_rows = (torch.load(a.speech_cache, map_location="cpu", weights_only=False)
                   if a.speech_cache else None)
    noise_rows = (torch.load(a.noise_cache, map_location="cpu", weights_only=False)
                  if a.noise_cache else None)
    if not 0 <= a.public_noise_prob <= 1:
        p.error("--public-noise-prob must be in [0, 1]")
    if not 0 <= a.full_mask_prob <= 1:
        p.error("--full-mask-prob must be in [0, 1]")
    if a.public_noise_prob and not noise_rows:
        p.error("--public-noise-prob requires a nonempty --noise-cache")
    if noise_rows:
        print("PUBLIC_NOISE_POOL", len(noise_rows), flush=True)
    if speech_rows:
        train_ids = {row.get("id") for row in rows}
        if secondary:
            train_ids.update(row.get("id") for row in secondary)
        overlap = train_ids.intersection(row.get("id") for row in speech_rows)
        if overlap:
            p.error(f"Speech cache overlaps paired training IDs: {len(overlap)}")
        print("SPEECH_POOL", len(speech_rows), flush=True)
    if not 0 <= a.high_coverage_prob <= 1:
        p.error("--high-coverage-prob must be in [0, 1]")
    def high_coverage_pool(pool):
        selected = []
        for row in pool:
            n = len(row["waveform"])
            coverage = sum(max(0., float(event["end"]) - float(event["start"]))
                           for event in row.get("events", [])) / max(n / 16000, 1e-8)
            if coverage > .7:
                selected.append(row)
        return selected
    high_rows = high_coverage_pool(rows)
    high_secondary = high_coverage_pool(secondary) if secondary else None
    if a.high_coverage_prob and not high_rows:
        p.error("No high-coverage clips in training cache")
    print("TRAIN_POOL", len(rows), "HIGH_COVERAGE", len(high_rows), flush=True)
    rng = random.Random(a.seed); torch.manual_seed(a.seed); np.random.seed(a.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-6)
    length = int(a.seconds*16000)
    perturb = None
    if a.mask_shift or a.mask_endpoint_jitter or a.mask_scale_min != 1 or a.mask_scale_max != 1:
        perturb = lambda events, random_state: perturb_events(
            events, random_state, a.mask_shift, a.mask_endpoint_jitter,
            a.mask_scale_min, a.mask_scale_max)
    for step in range(1, a.steps+1):
        use_secondary = bool(secondary and rng.random() < a.secondary_prob)
        pool = secondary if use_secondary else rows
        high_pool = high_secondary if use_secondary else high_rows
        if high_pool and rng.random() < a.high_coverage_prob:
            pool = high_pool
        batch = []
        for _ in range(a.batch_size):
            item = (remix_crop(pool, length, rng, mask_fn, perturb, a.snr_min, a.snr_max,
                               speech_rows=speech_rows, noise_rows=noise_rows,
                               public_noise_prob=a.public_noise_prob)
                    if rng.random() < a.remix_prob else
                    sample_crop(rng.choice(pool), length, rng, mask_fn, perturb))
            if rng.random() < a.full_mask_prob:
                full_mask = np.zeros_like(item[2]); full_mask[-1, :] = 1
                item = item[0], item[1], full_mask
            batch.append(item)
        noisy = torch.from_numpy(np.stack([z[0] for z in batch])).cuda()
        clean = torch.from_numpy(np.stack([z[1] for z in batch])).cuda()
        sample_masks = torch.from_numpy(np.stack([z[2] for z in batch])).cuda()
        optimizer.zero_grad(set_to_none=True)
        norm = torch.sqrt(noisy.shape[-1]/noisy.square().sum(-1).clamp_min(1e-8)); xn = noisy*norm[:, None]
        amp, phase, _ = mag_pha_stft(xn, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
        frame_masks = F.interpolate(sample_masks, size=amp.shape[-1], mode="linear", align_corners=False)
        out_amp, out_phase = model(amp, phase, frame_masks)
        enhanced = mag_pha_istft(out_amp, out_phase, h.n_fft, h.hop_size, h.win_size, h.compress_factor)/norm[:, None]
        union = sample_masks.amax(1)
        if a.gated_loss:
            radius=max(0,int(round(a.gate_pad*16000))); gate=F.max_pool1d(union[:,None],2*radius+1,stride=1,padding=radius).squeeze(1)
            width=max(1,int(round(2*a.gate_fade*16000))|1); gate=F.avg_pool1d(gate[:,None],width,stride=1,padding=width//2).squeeze(1).clamp(0,1)
            output=noisy+gate*(enhanced-noisy)
        else: gate=union; output=enhanced
        error = (output-clean).abs(); outside = 1-gate
        si = neg_sisdr(output, clean)
        event_l1 = (error*gate).sum()/gate.sum().clamp_min(1)/(clean.abs().mean()+1e-4)
        preserve = ((output-noisy).abs()*outside).sum()/outside.sum().clamp_min(1)/(noisy.abs().mean()+1e-4)
        if asr_encoder is not None:
            from sravaani_features import feature_distance
            asr_out=output/(output.square().mean(-1,keepdim=True).sqrt()+1e-5)
            asr_clean=clean/(clean.square().mean(-1,keepdim=True).sqrt()+1e-5)
            with torch.no_grad(): clean_features=asr_encoder(asr_clean)
            asr_loss=feature_distance(asr_encoder(asr_out),clean_features)
        else:
            asr_loss=output.new_zeros(())
        stft_loss = multires_spectral_loss(output, clean) if a.stft_weight else output.new_zeros(())
        loss = si + .15*event_l1 + .10*preserve + a.asr_feature_weight*asr_loss + a.stft_weight*stft_loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3); optimizer.step()
        if step % 20 == 0:
            print(f"STEP {step} loss={loss.item():.4f} sisdr={-si.item():.3f} event={event_l1.item():.4f} preserve={preserve.item():.4f} asr={asr_loss.item():.4f} stft={stft_loss.item():.4f}", flush=True)
        if step % 200 == 0 or step == a.steps:
            out = a.output.with_name(a.output.stem+f"_s{step}.pt")
            torch.save({"generator": model.state_dict(), "conditions": conditions,
                        "conditioning": a.conditioning, "step": step,
                        "mask_augmentation": vars(a)}, out)
    torch.save({"generator": model.state_dict(), "conditions": conditions,
                "conditioning": a.conditioning, "step": a.steps,
                "mask_augmentation": vars(a)}, a.output)


if __name__ == "__main__": main()
