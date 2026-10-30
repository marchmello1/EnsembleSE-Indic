#!/usr/bin/env python3
"""Compare enhancement on paired synthetic plus natural WER; no test labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from eval_t2_gainmatched_blend import sisdr


def main():
    p = argparse.ArgumentParser()
    for name in ("validation", "baseline_wavs", "candidate_wavs", "baseline_syn_asr",
                 "candidate_syn_asr", "baseline_nat_asr", "candidate_nat_asr", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--synthetic-word-share", type=float, default=0.320345)
    a = p.parse_args()
    meta = {r["segmentFileName"]: r for r in json.loads((a.validation / "validationMetadata.json").read_text())}
    reports = [json.loads(path.read_text()) for path in (a.baseline_syn_asr, a.candidate_syn_asr,
                                                       a.baseline_nat_asr, a.candidate_nat_asr)]
    deltas = []
    paired = []
    subset = []
    for index in (0, 2):
        baseline = {r["clip"]: r for r in reports[index]["results"]}
        candidate = {r["clip"]: r for r in reports[index + 1]["results"]}
        if baseline.keys() != candidate.keys():
            raise ValueError("ASR clip coverage differs")
        error_delta = 0
        words = 0
        for name, row in baseline.items():
            other = candidate[name]
            if row["reference_words"] != other["reference_words"] or row["noisy_edits"] != other["noisy_edits"]:
                raise ValueError("reference/noisy ASR mismatch")
            delta = row["enhanced_edits"] - other["enhanced_edits"]
            words += row["reference_words"]
            error_delta += delta
            si_delta = None
            if index == 0:
                b, sr = sf.read(a.baseline_wavs / name, dtype="float32")
                c, sr2 = sf.read(a.candidate_wavs / name, dtype="float32")
                ref, sr3 = sf.read(a.validation / "syntheticCleanRefAudio" / name, dtype="float32")
                if len(b) != len(c) or len(b) != len(ref) or sr != 16000 or sr2 != sr or sr3 != sr:
                    raise ValueError(name)
                si_delta = sisdr(c, ref) - sisdr(b, ref)
                deltas.append(si_delta)
            paired.append({"state": meta[name]["state"], "words": row["reference_words"],
                           "edit_reduction": delta, "si_delta": si_delta})
        subset.append({"clips": len(baseline), "words": words,
                       "baseline_errors": reports[index]["errors"]["enhanced"],
                       "candidate_errors": reports[index + 1]["errors"]["enhanced"],
                       "wer_gain_points": 100 * error_delta / words})
    si_gain = float(np.mean(deltas))
    total_words = sum(r["words"] for r in paired)
    pooled_gain = si_gain + 100 * sum(r["edit_reduction"] for r in paired) / total_words
    states = sorted({r["state"] for r in paired})
    groups = {state: [r for r in paired if r["state"] == state] for state in states}
    rng = np.random.default_rng(1009)
    boots = []
    for _ in range(5000):
        sample = [row for state in rng.choice(states, len(states), replace=True) for row in groups[state]]
        si = [r["si_delta"] for r in sample if r["si_delta"] is not None]
        if si:
            boots.append(float(np.mean(si)) + 100 * sum(r["edit_reduction"] for r in sample) /
                         sum(r["words"] for r in sample))
    report = {"synthetic": subset[0], "natural": subset[1], "si_gain_db": si_gain,
              "pooled_combined_gain": pooled_gain,
              "state_bootstrap_95ci": np.quantile(boots, [.025, .975]).tolist(),
              "phase2_word_share_weighted_validation_gain": si_gain +
                  a.synthetic_word_share * subset[0]["wer_gain_points"] +
                  (1 - a.synthetic_word_share) * subset[1]["wer_gain_points"],
              "states": states,
              "scope": "Released held-out validation; local WER normalization; not an official score prediction."}
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
