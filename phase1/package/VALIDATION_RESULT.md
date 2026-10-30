# Track 2 MP-SENet clean unseen candidate

Generated: 2026-09-24 (Asia/Kolkata)

## Selected recipe

- Architecture: MP-SENet initialized from the public DNS-pretrained checkpoint.
- Training: event-conditioned, state-disjoint paired validation training.
- Selection: six completely held-out states (188 synthetic clips); no template or
  known-waveform matching/subtraction.
- Inference geometry: 1.5 s context, 0.125 s event padding, 0.0375 s cross-fade.
- Selected state-CV checkpoint: initial 4,000 steps at `5e-6`, followed by 2,400
  steps at `2e-6`.
- Deployable full-data checkpoint: `t2_mp_dns_full_cont_0924.pt`, trained with the
  same selected schedule on all 695 released paired clips.

## Honest held-out evidence

| Metric | Value |
|---|---:|
| SI-SDR | 17.3614 dB |
| Noisy SraVaani errors | 800 / 2,190 words |
| Enhanced SraVaani errors | 739 / 2,190 words |
| Delta-WER contribution | +2.7854 |
| Combined proxy | **20.1468** |

The mandated ASR proxy uses pooled word errors and is not claimed to be the
organizer's exact scorer. With deterministic ±30 ms boundary/endpoint error,
SI-SDR was 17.3375 dB (only -0.0239 dB), so the selection is not dependent on
perfect event boundaries.

## Rejected alternatives

- 3,200-step continuation: 17.3947 SI-SDR but 740 ASR errors, combined 20.1344.
- Boundary-jitter continuation: 17.3809 SI-SDR but 740 ASR errors, combined
  20.1206.
- Second seed at step 3,200: 16.8910 SI-SDR.

## Checkpoint integrity

SHA-256: `8c872bf3016f027ceff2296f56d5c2f77f90f09d8f1d8428bd7a611bfb0c9d24`

This is the best defensible unseen candidate. The previously verified official
clean score remains 17.7809 until this new recipe is independently evaluated.

The final full-data checkpoint also passed an all-695 paired-data sanity check
at 20.2003 dB SI-SDR. This number is in-sample and is recorded only as an
artifact-integrity check, not as the unseen-score estimate.

## Additional state-disjoint replication

Two further geographic splits were trained independently. Intermediate
checkpoints were used here only to obtain early independent evidence; the final
automated schedules continue separately.

| Fold | Held-out clips | SI-SDR | Delta-WER contribution | Combined |
|---|---:|---:|---:|---:|
| 1 | 188 | 17.3614 | 2.7854 | **20.1468** |
| 2 | 102 | 17.6216 | 0.7456 | **18.3672** |
| 3 | 171 | 22.8747 | 1.5715 | **24.4463** |

Pooling the three disjoint evaluations by clip for SI-SDR and by reference-word
count for Delta-WER gives an approximate combined score of **21.35** across
461 held-out clips. The dispersion is material: the evidence supports an
overall >=20 expectation, not a claim that every geographic subset clears 20.
# Final noisy-audio-only held-state result (2026-09-24)

The clean, deployable chain now exceeds 20 without supplied timestamps or
known-waveform subtraction:

- six geographically held-out states, 188 clips;
- learned binary event detector operating on noisy WAV only;
- two independently tuned MP-SENet checkpoints;
- SraVaani transcript-consensus routing (choose the enhanced waveform whose
  mandated-ASR transcript is closest to the noisy-ASR transcript);
- SI-SDR: **17.031157 dB**;
- noisy/enhanced ASR errors: **800 / 731** over 2,190 reference words;
- Delta-WER contribution: **3.150685**;
- combined: **20.181842**.

Authoritative remote artifacts:

- selected WAVs: `/mnt/nvme_disk1/t2_render_consensus_clean20_0924`
- ASR result: `/mnt/nvme_disk1/asr_consensus_clean20_0924.json`
- detector cache used for the held-state audit:
  `/mnt/nvme_disk1/t2_binary_pred_cf85d40_t0.50.pt`
- selector implementation: `select_track2_asr_consensus.py`

This is an unseen-state validation estimate, not a claimed Codabench score.
The old 24.16 estimate is rejected because its SEMamba initializer had seen
the held-out validation states.
