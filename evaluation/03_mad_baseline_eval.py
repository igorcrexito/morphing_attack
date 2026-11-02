"""Step 3 - Detectability: train + cross-validate the statistical MAD baseline.

See evaluation/common/mad_features.py for exactly what this detector is (and, "
importantly, what it is NOT a reproduction of). In short: a compression-error +
spectral-energy statistical feature vector per image, classified with an SVM,
cross-validated so every image is scored by a model that never trained on it.

Bona fide class : every FRLL neutral + smiling capture (204 real photos).
Attack class    : every generated SYN-MAD morph (1000 by default).

Usage:
    python evaluation/03_mad_baseline_eval.py
    python evaluation/03_mad_baseline_eval.py --morph-root <dir> --label ablation_full
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.common import paths as P
from evaluation.common import metrics
from evaluation.common.mad_features import compression_error_features


def _load_rgb01(path, size=224):
    img = Image.open(path).convert("RGB").resize((size, size))
    return np.asarray(img).astype(np.float32) / 255.0


def _bonafide_paths(bonafide_dirs=None):
    dirs = bonafide_dirs or [P.FRLL_NEUTRAL_DIR, P.FRLL_SMILING_DIR]
    paths = []
    for d in dirs:
        paths += glob.glob(os.path.join(d, "*.jpg"))
    return sorted(paths)


def _attack_paths(morph_root):
    paths = []
    for expr in P.EXPRESSION_DIRS:
        paths += glob.glob(os.path.join(morph_root, expr, "morph_*_*.jpg"))
    return sorted(paths)


def _extract_features(paths, desc):
    feats = []
    for p in tqdm(paths, desc=desc, unit="img"):
        feats.append(compression_error_features(_load_rgb01(p)))
    return np.stack(feats)


def run_mad_eval(morph_root, label, n_folds=5, seed=42, bonafide_dirs=None):
    bonafide_paths = _bonafide_paths(bonafide_dirs)
    attack_paths = _attack_paths(morph_root)
    if not attack_paths:
        raise SystemExit(f"No morphs found under {morph_root}")

    print(f"[{label}] bona fide: {len(bonafide_paths)} | attacks: {len(attack_paths)}")
    X_bf = _extract_features(bonafide_paths, f"{label}: bona fide features")
    X_at = _extract_features(attack_paths, f"{label}: attack features")

    X = np.concatenate([X_bf, X_at], axis=0)
    # y = 1 for attack (morph), 0 for bona fide - matches "higher score = more
    # suspicious" convention used by common/metrics.apcer_bpcer_at_threshold.
    y = np.concatenate([np.zeros(len(X_bf)), np.ones(len(X_at))])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_score = np.zeros(len(y))  # out-of-fold predicted probability of "attack"

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=4.0, gamma="scale", probability=True,
               class_weight="balanced", random_state=seed))
        clf.fit(X[train_idx], y[train_idx])
        oof_score[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        print(f"  fold {fold + 1}/{n_folds} done "
              f"({len(train_idx)} train / {len(test_idx)} test)")

    attack_scores = oof_score[y == 1]
    bonafide_scores = oof_score[y == 0]

    eer_val, eer_thr = metrics.eer(1.0 - bonafide_scores, 1.0 - attack_scores)
    # metrics.eer expects (genuine, impostor) with HIGHER = more genuine-like;
    # our score is HIGHER = more attack-like, so invert both to keep the shared
    # `eer` implementation's convention, then translate back for reporting.
    apcer, bpcer = metrics.apcer_bpcer_at_threshold(attack_scores, bonafide_scores,
                                                     1.0 - eer_thr)
    bpcer_at_5, apcer_at_5 = metrics.bpcer_at_apcer(attack_scores, bonafide_scores, 0.05)
    bpcer_at_10, apcer_at_10 = metrics.bpcer_at_apcer(attack_scores, bonafide_scores, 0.10)

    result = {
        "label": label,
        "n_bonafide": len(bonafide_paths),
        "n_attacks": len(attack_paths),
        "n_folds": n_folds,
        "eer": eer_val,
        "apcer_at_eer_threshold": apcer,
        "bpcer_at_eer_threshold": bpcer,
        "bpcer_at_apcer_5pct": bpcer_at_5,
        "bpcer_at_apcer_10pct": bpcer_at_10,
    }

    out_dir = P.ensure_artifacts_dir("mad_baseline")
    out_path = os.path.join(out_dir, f"{label}_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    np.savez(os.path.join(out_dir, f"{label}_scores.npz"),
             attack_scores=attack_scores, bonafide_scores=bonafide_scores)

    print(f"[{label}] EER={eer_val:.4f} | BPCER@APCER5%={bpcer_at_5:.4f} | "
          f"BPCER@APCER10%={bpcer_at_10:.4f}  -> {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--morph-root", default=P.SYNMAD_MORPH_ROOT)
    ap.add_argument("--label", default="full_pipeline")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    run_mad_eval(args.morph_root, args.label, n_folds=args.folds)


if __name__ == "__main__":
    main()
