"""Step 5 - Generate matched ablation variants to isolate each pipeline stage.

For a subset of the SYN-MAD pairs, three versions of "the same morph" are
produced so steps 2/3 can be re-run on each and the DIFFERENCE attributed to a
single design choice:

  geo_only        - classical warp + alpha cross-dissolve only. No neural
                    network, no Poisson blending, no restoration. This is what
                    every purely geometric morphing paper (Ferrara 2014 style)
                    produces.
  network_only     - adds the trained residual-refinement network (Stage 3 of
                    the paper) on top of geo_only, but WITHOUT the Poisson seam
                    blend or GFPGAN restoration that Stage 4 normally applies.
  full_pipeline    - the complete pipeline (network + Poisson blend + pose-back
                    + GFPGAN), i.e. exactly what main_generate_morphed_dataset.py
                    already produced in morphed_dataset/frll_synmad. Reused by
                    copying the matching files rather than regenerating them.

All three variants share the same identity pairs and the same geometric warp,
so any metric difference between them is attributable to that one added stage,
not to a different pair being morphed.

Usage:
    python evaluation/05_ablation_generate.py --limit 120
"""
import argparse
import csv
import os
import shutil
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
tf.get_logger().setLevel("ERROR")

from evaluation.common import paths as P
from morphing.warp import morph_pair, warp_to_landmarks
from morphing.dataset import landmarks_to_heatmaps
from model.diffusion_model import DiffusionModel

WIDTH = HEIGHT = 224
ALPHA = 0.5


def _load_image(path):
    return np.asarray(Image.open(path).convert("RGB").resize((WIDTH, HEIGHT)),
                      dtype=np.float32) / 255.0


def _load_landmarks(jpg_path):
    csv_path = os.path.splitext(jpg_path)[0] + ".csv"
    lm = np.loadtxt(csv_path, delimiter=",", skiprows=1).astype(np.float32)
    with Image.open(jpg_path) as im:
        src_w, src_h = im.size
    if (src_w, src_h) != (WIDTH, HEIGHT):
        lm[:, 0] *= WIDTH / src_w
        lm[:, 1] *= HEIGHT / src_h
    return lm


def _read_subset(limit_per_group):
    with open(P.PAIRS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    subset = []
    for group, grows in by_group.items():
        # take an evenly spaced sample (by rank/similarity) rather than just the
        # first N, so the subset isn't biased toward the single most-similar pairs.
        step = max(1, len(grows) // limit_per_group)
        subset += grows[::step][:limit_per_group]
    return subset


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=30,
                    help="pairs PER group (4 groups) to include, default 30 "
                         "(-> up to 120 pairs total, matched across all 3 variants)")
    args = ap.parse_args()

    if not os.path.exists(P.PAIRS_CSV):
        raise SystemExit(f"{P.PAIRS_CSV} not found - run main_syn_mad_pairs.py first.")
    subset = _read_subset(args.limit)
    print(f"Ablation subset: {len(subset)} pairs")

    model = DiffusionModel()
    if os.path.exists(P.MORPH_WEIGHTS):
        model.load(P.MORPH_WEIGHTS)
        print(f"Loaded trained weights from {P.MORPH_WEIGHTS}")
    else:
        print("WARNING: no trained weights found - 'network_only' will use an "
              "untrained network (delta ~ noise). Train with main_train_model.py first.")

    geo_root = P.ensure_artifacts_dir("ablation", "geo_only")
    net_root = P.ensure_artifacts_dir("ablation", "network_only")
    full_root = P.ensure_artifacts_dir("ablation", "full_pipeline")

    n_copied_full, n_missing_full = 0, 0
    for row in tqdm(subset, desc="Generating ablation variants", unit="pair"):
        expression = row["expression"]
        id_a, id_b = int(row["id_a"]), int(row["id_b"])
        src_dir = P.EXPRESSION_DIRS[expression]
        path_a = os.path.join(src_dir, f"image_{id_a}.jpg")
        path_b = os.path.join(src_dir, f"image_{id_b}.jpg")
        if not (os.path.exists(path_a) and os.path.exists(path_b)):
            continue

        lm_a, lm_b = _load_landmarks(path_a), _load_landmarks(path_b)
        img_a, img_b = _load_image(path_a), _load_image(path_b)

        warped_a, warped_b, mean_lm, mask = morph_pair(
            img_a, img_b, lm_a, lm_b, ALPHA, WIDTH, HEIGHT)

        # ---- geo_only: classical alpha cross-dissolve, masked exactly like the
        # network's own "base" term (see model/diffusion_model.py:_build_morph),
        # so the ONLY thing missing relative to network_only is the learned delta.
        blended = (1.0 - ALPHA) * warped_a + ALPHA * warped_b
        base = mask * blended + (1.0 - mask) * warped_a
        geo_img = np.clip(base, 0.0, 1.0)
        geo_posed = warp_to_landmarks(geo_img, mean_lm, lm_a, WIDTH, HEIGHT)

        # ---- network_only: + trained residual correction, no Poisson/GFPGAN.
        heatmaps = landmarks_to_heatmaps(mean_lm, WIDTH, HEIGHT)
        morph, _ = model.predict(
            warped_a[None].astype(np.float32), warped_b[None].astype(np.float32),
            heatmaps[None].astype(np.float32), mask[None].astype(np.float32),
            alpha=ALPHA)
        net_img = np.clip(morph[0].numpy(), 0.0, 1.0)
        net_posed = warp_to_landmarks(net_img, mean_lm, lm_a, WIDTH, HEIGHT)

        out_dir_geo = os.path.join(geo_root, expression)
        out_dir_net = os.path.join(net_root, expression)
        os.makedirs(out_dir_geo, exist_ok=True)
        os.makedirs(out_dir_net, exist_ok=True)
        fname = f"morph_{id_a}_{id_b}.jpg"
        Image.fromarray((geo_posed * 255).astype(np.uint8)).save(os.path.join(out_dir_geo, fname))
        Image.fromarray((net_posed * 255).astype(np.uint8)).save(os.path.join(out_dir_net, fname))

        # ---- full_pipeline: reuse the already-generated final morph (network +
        # Poisson blend + pose-back + GFPGAN) instead of recomputing it.
        src_full = os.path.join(P.SYNMAD_MORPH_ROOT, expression, fname)
        out_dir_full = os.path.join(full_root, expression)
        os.makedirs(out_dir_full, exist_ok=True)
        if os.path.exists(src_full):
            shutil.copy(src_full, os.path.join(out_dir_full, fname))
            n_copied_full += 1
        else:
            n_missing_full += 1

    print(f"\nDone. geo_only/network_only written for {len(subset)} pairs.")
    print(f"full_pipeline: copied {n_copied_full} existing morphs "
          f"({n_missing_full} missing - run main_generate_morphed_dataset.py "
          f"--pairs pairs_synmad.csv first if this is > 0).")


if __name__ == "__main__":
    main()
