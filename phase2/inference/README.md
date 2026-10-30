# marchmello01 — Track 2 Phase 2 inference package (final system, "F15a")

Official Phase 2 result: **Combined 19.8688** (SI-SDR 16.2889 dB on the 1,099 synthetic clips; pooled ΔWER 3.5799
over all 3,178 clips; ΔWER synthetic 10.4323, natural 0.3501; ASR audit PASS).

Contact: Mohammad Agwan (team marchmello01), mohammad.agwan@nuvo.ai

## Method

One audio-only enhancement pipeline is applied identically to every input clip. There is no noise dictionary,
correlation or template search, natural-vs-synthetic routing, passthrough, filename rule or cached output: every
clip is processed by the same networks with the same fixed arithmetic.

The enhancer is a fixed ensemble of four neural speech-enhancement architectures (seven checkpoints):

| Slot | Network | Checkpoint(s) | Effective weight |
|---|---|---|---|
| A | MP-SENet (magnitude/phase) | `t2_mpdns_publicnoise_hilr_1008.pt` | 0.027 |
| B | URGENT BSRNN (input level-normalised to RMS 0.10) | `bsrnn_extended_s1600.ckpt` | 0.063 |
| C | TF-GridNet (complex spectrum) | `predictor_d32_s800.pt` | 0.21 |
| D | TF-Locoformer (complex spectrum), four checkpoints, equal peak-normalised average | `locoformer_d32.pth`, `locoformer_orig.pth`, `locoformer_spec.pth`, `locoformer_extra.pth` | 0.70 (0.175 each) |

Fixed combination (see `tta_enhance.py`):

```
base = 0.3 * MP + 0.7 * g * BSRNN          # g = least-squares gain of BSRNN onto MP (from the two outputs only)
best = 0.3 * base + 0.7 * TFGridNet
out  = 0.3 * best + 0.7 * Locoformer_avg   # with peak-safety normalisation and PCM16 round trips as in the code
```

**Test-time augmentation.** Each network runs on eight transformed copies of the input (polarity ±1 × leading
shift of 0, 40, 80 or 120 samples); each output is un-transformed and the eight are averaged before combination.

The four Locoformer checkpoints come from deliberately different training lineages (see the repository-level
`README.md`); averaging diverse members is what moved the official score from 19.754 to 19.869.

## Setup

Linux, Python 3.10, CUDA GPU (tested on NVIDIA H100 80 GB, driver CUDA 12.8; and on H200). ESPnet and NeMo need
different Librosa versions, so two environments are used:

    python3.10 -m venv enhancement_env
    enhancement_env/bin/python -m pip install --upgrade pip
    enhancement_env/bin/python -m pip install -r requirements.txt
    python3.10 -m venv asr_env
    asr_env/bin/python -m pip install --upgrade pip
    asr_env/bin/python -m pip install -r asr_requirements.txt

Exact tested versions: `enhancement.lock.txt`, `asr.lock.txt`, `clean_environment_manifest.json`.

Download the seven checkpoints into `checkpoints/` (see `../checkpoints/README.md`; they are attached to the GitHub
release because several exceed GitHub's 100 MB file limit). SraVaani is loaded from Hugging Face at runtime
(`ARTPARK-IISc/SraVaani-1.0`).

## Run

    TRACK2_ASR_PYTHON=/absolute/path/to/asr_env/bin/python CUDA_VISIBLE_DEVICES=0 \
        enhancement_env/bin/python main.py INPUT_FOLDER OUTPUT_FOLDER

`INPUT_FOLDER` holds root-level `.wav` files; `OUTPUT_FOLDER` must be empty. Output: same-name 16-kHz mono PCM16
WAVs plus `transcripts.jsonl`, one record per WAV: `{"clip_id": "<name without .wav>", "text": "<SraVaani output>"}`.
After enhancement the process replaces itself with `asr_worker.py` (frees GPU memory) and SraVaani transcribes the
saved enhanced WAVs.

Runtime: about 12 GPU-seconds per clip (seven checkpoints × eight TTA views); the 3,178-clip Phase 2 test set takes
about 2.5 h on four H100s. Inputs can be split into shards and processed in parallel.

## Reproduction notes

GPU numerical differences change outputs by about one PCM16 unit (we measured 86 dB agreement between an H100/torch
2.9.1 run and an H200/torch 2.7.0 run) and, rarely, a diacritic in a transcript.

## Third-party licences

Retain the bundled MP-SENet, URGENT and TF-Locoformer notices and `licenses/LICENSE_espnet` (Apache-2.0) when
redistributing.
