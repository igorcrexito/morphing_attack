"""Step 1 - Calibrate face-recognition decision thresholds on BONA FIDE data only.

Why this has to be its own, separate, first step: a threshold picked by looking
at the morphs would make the vulnerability numbers in step 2 meaningless (you'd
be grading your own homework). Instead we fix each face-recognition system's
operating point the same way a real deployment would: from real genuine/impostor
comparisons, at a target False Match Rate (FMR), before the morphs are ever
touched.

Data used: the 102 FRLL subjects, each with a neutral capture (output_dataset/
frll/image_<id>.jpg) and a smiling capture (output_dataset/frll_probe/
image_<id>.jpg) of the SAME person.
  * genuine pairs  = (neutral_i, smiling_i)            for every subject i  -> 102 scores
  * impostor pairs = (neutral_i, smiling_j), i != j     for every other subject -> 102*101 scores

Output: evaluation/artifacts/thresholds.json, one entry per available FR system:
    {"facenet": {"fmr_1pct": 0.41, "fmr_0.1pct": 0.52, "baseline_eer": 0.021, ...}, ...}

Usage:
    python evaluation/01_calibrate_thresholds.py
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.common import paths as P
from evaluation.common import metrics
from evaluation.common.fr_systems import available_systems, cosine_matrix

TARGET_FMRS = {"fmr_1pct": 0.01, "fmr_0.1pct": 0.001}


def _subject_ids():
    with open(P.FRLL_INDEX_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    return sorted(int(r["identity"]) for r in rows)


def main():
    ids = _subject_ids()
    neutral_paths = [os.path.join(P.FRLL_NEUTRAL_DIR, f"image_{i}.jpg") for i in ids]
    smiling_paths = [os.path.join(P.FRLL_SMILING_DIR, f"image_{i}.jpg") for i in ids]
    missing = [p for p in neutral_paths + smiling_paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} FRLL images missing, e.g. {missing[0]}. "
                          f"Run main_prepare_frll.py / main_save_landmarks.py first.")

    systems = available_systems()
    print(f"FR systems available: {list(systems)}")

    results = {}
    for name, system in systems.items():
        try:
            emb_neutral = system.embed(neutral_paths)
            emb_smiling = system.embed(smiling_paths)
        except KeyError as e:
            print(f"[skip] {name}: {e}")
            continue

        sims = cosine_matrix(emb_neutral, emb_smiling)  # (102, 102)
        genuine = np.diag(sims)
        mask = ~np.eye(len(ids), dtype=bool)
        impostor = sims[mask]

        entry = {
            "n_subjects": len(ids),
            "n_genuine_pairs": int(len(genuine)),
            "n_impostor_pairs": int(len(impostor)),
            "genuine_mean": float(genuine.mean()),
            "genuine_std": float(genuine.std()),
            "impostor_mean": float(impostor.mean()),
            "impostor_std": float(impostor.std()),
        }
        baseline_eer, eer_thr = metrics.eer(genuine, impostor)
        entry["baseline_eer"] = baseline_eer
        entry["baseline_eer_threshold"] = eer_thr
        for label, fmr in TARGET_FMRS.items():
            entry[label] = metrics.calibrate_threshold(genuine, impostor, fmr)

        results[name] = entry
        print(f"  {name:10s}: baseline EER={baseline_eer:.4f} | "
              f"thr@FMR1%={entry['fmr_1pct']:.4f} | thr@FMR0.1%={entry['fmr_0.1pct']:.4f}")

    out_dir = P.ensure_artifacts_dir()
    out_path = os.path.join(out_dir, "thresholds.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote thresholds for {len(results)} system(s) to {out_path}")


if __name__ == "__main__":
    main()
