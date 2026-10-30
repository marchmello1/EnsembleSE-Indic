# marchmello01 — IndoML Track 2 Phase 1 inference package

This is the team's strongest reproducible noisy-WAV-only Phase 1 system. It
scored **20.1818 Combined** on a speaker/state-held-out 188-clip development
split (SI-SDR 17.0312 dB and 3.1507 points from delta WER). This is local
held-out evidence, not a claimed leaderboard or organizer-audit score.

## Method

1. A binary 100-Hz CRNN predicts noise-event spans from the input waveform.
2. Two event-local MP-SENet experts enhance only predicted spans, preserving the
   original signal elsewhere with 50-ms crossfades.
3. The mandated `ARTPARK-IISc/SraVaani-1.0` model transcribes the noisy waveform
   and both estimates. The estimate with the smaller word edit distance from the
   noisy transcript is retained (ties select expert A).
4. The retained SraVaani hypothesis is written to `transcripts.jsonl`.

No validation/test template lookup, known-waveform subtraction, test label,
clip identity, clean reference, or leaderboard feedback is used at inference.

## Run

Tested with Python 3.10, CUDA, PyTorch 2.x, and 16-kHz mono WAV input.

```bash
python -m pip install -r requirements.txt
python main.py /path/to/input_wavs /path/to/output
```

`main.py` accepts exactly the organizer-requested positional input and output
folders. It resamples non-16-kHz input and writes same-name 16-kHz mono PCM-16
WAV files plus root-level `transcripts.jsonl`.

The invocation must contain exactly these two arguments; there are no hidden
clip lists, metadata inputs, timestamps, templates, or reference files:

```bash
python main.py INPUT_FOLDER OUTPUT_FOLDER
```

The first run downloads the mandated SraVaani model from Hugging Face. A CUDA
GPU is strongly recommended. The package intentionally excludes SraVaani.

## Provenance and integrity

`checkpoints/` contains the full released-data deployment models. `vendor/`
contains the minimal MP-SENet inference source and its upstream README. File
hashes are recorded in `SHA256SUMS`.

Before uploading, run `python verify_package.py`. The organizer may run the
same command before executing `main.py` on independently selected noisy WAVs.

Primary contact: **Mohammad Agwan**, **Nuvo AI**  
Email: **mohammad.agwan@nuvo.ai**  
