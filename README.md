# EnsembleSE-Indic — diverse-member neural speech-enhancement ensembles for noise event removal

**IndoML Datathon 2026 · Track 2 (Noise Event Removal) · team marchmello01 · 2nd place in Phase 1**

EnsembleSE-Indic removes noise events from Indian-language speech while keeping it recognisable by the mandated ASR model
(SraVaani-1.0). The final Phase 2 system is a fixed ensemble of four neural enhancer architectures (seven
checkpoints) whose TF-Locoformer slot fuses four differently-trained members, with eight-view test-time
augmentation, applied identically to every clip.

| Phase | Result |
|---|---|
| **Phase 1** | **2nd place** on the Track 2 leaderboard — event-local MP-SENet system, verification package in `phase1/` |
| **Phase 2 (final)** | **Combined 19.8688** — SI-SDR 16.2889 dB (1,099 synthetic clips) + pooled ΔWER 3.5799 (3,178 clips); ΔWER synthetic 10.4323, natural 0.3501; ASR audit PASS |

Contact: Mohammad Agwan (team marchmello01), mohammad.agwan@nuvo.ai

## Repository structure

```
README.md                    this file (setup, training, inference, environment)
requirements_inference.txt   enhancement environment for inference (ASR environment: phase2/inference/asr_requirements.txt)
requirements_train.txt       training environment
phase2/
  inference/                 final Phase 2 inference package (main.py INPUT OUTPUT) — code + configs
  checkpoints/README.md      the seven final checkpoints: download links (GitHub release) and SHA-256
  scripts/run_inference.sh   one-command inference
  train/                     complete training and data-preparation code, in the original working folders
    urgent2026_t2pilot_1008/ MP-SENet and BSRNN fine-tuning, ASR (RNNT) loss utilities, early fusion caches
    t2_gap_urgenet_1009/     URGENT TF-GridNet / BSRNN adaptation (predictor and BSRNN "extend1600")
    t2_locoformer_1010/      TF-Locoformer fine-tuning from the public DNS checkpoint; training-cache builder
    claude_d32/              reused-noise ("d32") remix trainers for Locoformer / TF-GridNet / BSRNN, second-slot lineage
    claude_vaani/            public-Vaani speech pipeline (leakage filter, quality gate, row builder) and the scaled trainers
phase1/
  package/                   Phase 1 verification package as submitted (code + three small checkpoints)
  train/                     Phase 1 detector and MP-SENet expert training scripts
docs/
  technical_report.pdf       IEEE-format technical report (2 pages; built from technical_report.html, LaTeX source technical_report.tex)
```

## Inference (Phase 2 final system)

```bash
# 1. environments (Python 3.10)
python3.10 -m venv enhancement_env && enhancement_env/bin/pip install -r requirements_inference.txt
python3.10 -m venv asr_env && asr_env/bin/pip install -r phase2/inference/asr_requirements.txt
# 2. checkpoints -> phase2/inference/checkpoints/   (see phase2/checkpoints/README.md)
# 3. run
bash phase2/scripts/run_inference.sh /path/to/input_wavs /path/to/empty_output_dir
```

Input: a folder of root-level 16-kHz `.wav` files. Output: same-name enhanced 16-kHz mono PCM16 WAVs plus
`transcripts.jsonl` (`{"clip_id": ..., "text": ...}` per clip, produced by SraVaani on the enhanced audio).
Method details and the exact combination arithmetic: `phase2/inference/README.md`.

## Training data and preprocessing

* **Released Track 2 data.** Training clean/noisy pairs from the released Vaani noise-event training set
  (two Indian states held out of training for validation), the released natural clips (used only with a
  preservation loss against a frozen teacher), and the released validation set.
* **Reused validation noise ("d32" bank).** 518 noise-event recordings extracted from the released validation
  noisy-minus-clean residuals. Training mixtures place these events over clean speech at a mean +9 dB
  event-to-speech level (SD 3 dB) covering about 34% of the clip. Training on validation noise was confirmed as
  allowed by the organizers. The bank is used **only for training**; inference uses no noise recordings.
* **Public Vaani speech (additional dataset).** 6,069 clean utterances from the public Vaani corpus
  (`claude_vaani/`). Leakage filter: an audio landmark-fingerprint index of all 3,178 Phase 2 test inputs was built
  (`fp.py`, `build_index.py`, `fp_validate.py`); any Vaani recording matching a test input at ≥ 12 votes was
  removed, and every recording of a speaker with ≥ 20 matches was removed; the two held-out states were excluded.
  Targets were pause-gated to mimic the organizers' clean references (`quality.py`, `build_rows*.py`).
  Training on public Vaani was confirmed as allowed by the organizers.
* **Augmentation pipeline.** On-the-fly remixing in every trainer: clean speech + (d32 events | released training
  residuals | the clip's own noise), random gain, peak normalisation to 0.95; for the event remix see `events()`
  in `claude_vaani/train_*.py`.

## Training (method and lineage of the seven final checkpoints)

All networks are trained with `2 · (−SI-SDR) + 3 · RNNT-loss` (RNNT through the frozen SraVaani encoder/joint,
transcripts from the released data) on synthetic mixtures, plus `0.2 ·` an L1 preservation loss against a frozen
teacher on natural clips; AdamW, gradient clipping 3. Exact objectives are in each trainer.

| Final checkpoint | Network | Lineage (each step: script) |
|---|---|---|
| `t2_mpdns_publicnoise_hilr_1008.pt` | MP-SENet | public MP-SENet DNS checkpoint → fine-tune on public-noise mixtures at a high LR with `urgent2026_t2pilot_1008/finetune_mpsenet_rnnt.py` (the exact command line of this run was not logged; the filename records its settings) |
| `bsrnn_extended_s1600.ckpt` | URGENT BSRNN | URGENT 2025 BSRNN baseline → Vaani fine-tune → extended to 1,600 steps (`t2_gap_urgenet_1009/finetune_extend1600_1010.py`) |
| `predictor_d32_s800.pt` | TF-GridNet | URGENT TF-GridNet → full adaptation → d32 specialist (`claude_vaani/train_tfgrid_spec.py`) → four continuation rounds of 40k steps at LR 5e-6 |
| `locoformer_d32.pth` | TF-Locoformer, member 1 | public DNS TF-Locoformer (`t2_locoformer_1010/finetune_locoformer_1010.py`) → d32 remix (`claude_d32/train_loco_d32_1013.py`) → scaled Vaani recipe (`claude_vaani/train_loco_big.py`, 60k) → four continuation rounds (round 4 at 36k steps) |
| `locoformer_orig.pth` | TF-Locoformer, member 2 | d32 long-context (`claude_d32/train_loco_d32long_1013.py`) → released-data recipe with natural preservation (`train_loco_hlr_1013.py`, 9k) → continued to 33k at LR 5e-6 (`train_loco_hlr2_1016.py`) |
| `locoformer_spec.pth` | TF-Locoformer, member 3 | reused-noise specialist recipe, d32-only mixtures (`claude_vaani/train_loco_spec.py`) from member-1 lineage at 30k → 40k steps |
| `locoformer_extra.pth` | TF-Locoformer, member 4 | same specialist recipe initialised from the round-4 member-1 checkpoint, 14k steps |

Training is single-GPU per run (`CUDA_VISIBLE_DEVICES=<g> python train_*.py --init <ckpt> --vaani <rows> --lr 5e-6
--steps <n> --tag <name>`); continuation rounds simply pass the previous round's checkpoint as `--init`.
Paths inside the scripts point to our server layout (`/mnt/nvme_disk1/...`) and must be edited for a new machine.

## Environment and hardware

| Item | Value |
|---|---|
| Python | 3.10.12 (training and inference); 3.12 on the H200 host (also verified for inference) |
| CUDA | driver 12.8; PyTorch 2.9.1+cu128 (H100 host), 2.7.0+cu128 (H200 host) |
| GPUs | 8 × NVIDIA H100 80 GB HBM3 (training and inference); 2 × NVIDIA H200 141 GB (continuation rounds) |
| Key libraries | ESPnet 202412, PyTorch-Lightning 2.5.0.post0, Transformers 4.56.2, NeMo 2.x (ASR env), librosa 0.9.2 (enhancement env) |
| System | Ubuntu Linux, ffmpeg/libsndfile |

Pretrained dependencies: SraVaani-1.0 (`ARTPARK-IISc/SraVaani-1.0`, Hugging Face; used for the RNNT training loss and
for transcription), TF-Locoformer DNS checkpoint (github.com/merlresearch/tf-locoformer), URGENT challenge BSRNN and
TF-GridNet baselines (github.com/urgent-challenge/urgent2025_challenge), MP-SENet (github.com/yxlu-0102/MP-SENet).

## Phase 1

`phase1/package/` is the verification package submitted for Phase 1 (`README.md` inside describes it): a binary
CRNN event-span detector (`phase1/train/train_t2_binary_event_detector.py`) and two event-local MP-SENet experts
fine-tuned with an ASR loss (`phase1/train/finetune_mpsenet_rnnt.py`). It predates the organizers' October ruling
on routing and is superseded by the Phase 2 system above.

## Licence

Code written for this challenge: MIT. Bundled third-party code keeps its own licence (see the licence files in
`phase2/inference/`: MP-SENet MIT, URGENT challenge code Apache-2.0, TF-Locoformer (MERL) Apache-2.0, ESPnet
Apache-2.0).
