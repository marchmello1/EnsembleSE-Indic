#!/bin/bash
# Usage: bash phase2/scripts/run_inference.sh INPUT_FOLDER OUTPUT_FOLDER
# Expects enhancement_env/ and asr_env/ next to the repository root (see README.md) and the seven checkpoints in
# phase2/inference/checkpoints/. Set CUDA_VISIBLE_DEVICES to choose the GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IN="$(realpath "$1")"; OUT="$(realpath -m "$2")"
export TRACK2_ASR_PYTHON="${TRACK2_ASR_PYTHON:-$ROOT/asr_env/bin/python}"
cd "$ROOT/phase2/inference"
exec "${ENHANCE_PYTHON:-$ROOT/enhancement_env/bin/python}" main.py "$IN" "$OUT"
