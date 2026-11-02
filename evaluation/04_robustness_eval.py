"""Step 4 - Robustness: do the results survive JPEG compression / a print-scan pass?

Rationale (see README.md "Robustness pass"): a real passport photo pipeline
compresses and sometimes prints+scans the submitted photo. If an attack or its
detectability numbers only "work" on the pristine PNG-quality image, they don't
represent a real threat. This script degrades the already-generated morphs (and,
for the MAD baseline, the bona fide images too, since a detector inspects
whatever photo was actually submitted) and re-runs steps 2 and 3 on the degraded
copies.

Only the SUBMITTED document photo (the morph) is degraded for the vulnerability
test - the probe/live-capture side of the comparison is left pristine, matching
the realistic scenario of an old, compressed enrollment photo being checked
against a fresh camera capture at the gate.

Usage:
    python evaluation/04_robustness_eval.py                  # both transforms
    python evaluation/04_robustness_eval.py --transform jpeg50
    python evaluation/04_robustness_eval.py --skip-mad        # vulnerability only (faster)
"""
import argparse
import glob
import importlib
import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.common import paths as P
from evaluation.common.degrade import jpeg_recompress, print_scan_simulate

vuln_mod = importlib.import_module("evaluation.02_vulnerability_eval")
mad_mod = importlib.import_module("evaluation.03_mad_baseline_eval")

TRANSFORMS = {
    "jpeg50": lambda img: jpeg_recompress(img, quality=50),
    "printscan": lambda img: print_scan_simulate(img, seed=42),
}


def _load_rgb01(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def _save_rgb01(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def _degrade_dir(src_dir, dst_dir, transform, desc):
    paths = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
    for p in tqdm(paths, desc=desc, unit="img"):
        img = _load_rgb01(p)
        out = transform(img)
        _save_rgb01(out, os.path.join(dst_dir, os.path.basename(p)))
    return dst_dir


def run_transform(transform_name, skip_mad=False):
    transform = TRANSFORMS[transform_name]
    root = P.ensure_artifacts_dir("degraded", transform_name)

    # --- degrade the morphs (both expression splits) -----------------------
    degraded_morph_root = os.path.join(root, "frll_synmad")
    for expr in P.EXPRESSION_DIRS:
        src = os.path.join(P.SYNMAD_MORPH_ROOT, expr)
        if not os.path.isdir(src):
            continue
        _degrade_dir(src, os.path.join(degraded_morph_root, expr), transform,
                    desc=f"[{transform_name}] degrading {expr} morphs")

    # --- vulnerability under degradation (probes stay pristine) ------------
    thr_path = os.path.join(P.ARTIFACTS_DIR, "thresholds.json")
    with open(thr_path) as f:
        thresholds = json.load(f)
    vuln_mod.evaluate_vulnerability(degraded_morph_root, thresholds,
                                    label=f"robust_{transform_name}")

    if skip_mad:
        return

    # --- MAD baseline under degradation (bona fide degraded the same way) --
    degraded_bf_neutral = os.path.join(root, "frll")
    degraded_bf_smiling = os.path.join(root, "frll_probe")
    _degrade_dir(P.FRLL_NEUTRAL_DIR, degraded_bf_neutral, transform,
                desc=f"[{transform_name}] degrading bona fide (neutral)")
    _degrade_dir(P.FRLL_SMILING_DIR, degraded_bf_smiling, transform,
                desc=f"[{transform_name}] degrading bona fide (smiling)")
    mad_mod.run_mad_eval(degraded_morph_root, label=f"robust_{transform_name}",
                         bonafide_dirs=[degraded_bf_neutral, degraded_bf_smiling])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transform", choices=list(TRANSFORMS), default=None,
                    help="run only this transform (default: run all)")
    ap.add_argument("--skip-mad", action="store_true",
                    help="skip the (slower) MAD baseline retraining pass")
    args = ap.parse_args()

    names = [args.transform] if args.transform else list(TRANSFORMS)
    for name in names:
        print(f"\n=== Robustness transform: {name} ===")
        run_transform(name, skip_mad=args.skip_mad)


if __name__ == "__main__":
    main()
