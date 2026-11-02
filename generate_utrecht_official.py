"""One-off: generate morphs on the Utrecht dataset with the official pipeline.

Mirrors official_pipeline/main_generate.py's --pairs branch exactly (same
warp -> network -> Poisson blend -> pose-blend -> GFPGAN steps, same
POSE_BLEND constant), just pointed at output_dataset/utrecht instead of the
FRLL neutral/smiling directories, since Utrecht only has one pair list
(pairs_utrecht.csv, built by a one-off ranking script, not a full SYN-MAD-style
protocol - see the evaluation writeup for the caveat this implies).

Usage:
    python generate_utrecht_official.py
"""
import csv
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from morphing.warp import morph_pair, poisson_seam_blend, warp_to_landmarks
from morphing.dataset import landmarks_to_heatmaps
from official_pipeline.model import BalancedMorphGenerator
from image_utils.face_restore import restore_face

WIDTH = HEIGHT = 224
ALPHA = 0.5
POSE_BLEND = 0.5  # same default as official_pipeline/main_generate.py
SRC_DIR = "output_dataset/utrecht"
OUT_DIR = "morphed_dataset_official/utrecht"
WEIGHTS = "cache/morph_model_official.weights.h5"


def _load_image(path):
    return np.asarray(Image.open(path).convert("RGB").resize((WIDTH, HEIGHT)),
                      dtype=np.float32) / 255.0


def _load_landmarks(jpg_path):
    csv_path = os.path.splitext(jpg_path)[0] + ".csv"
    return np.loadtxt(csv_path, delimiter=",", skiprows=1).astype(np.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open("pairs_utrecht.csv", newline="") as f:
        pairs = list(csv.DictReader(f))
    print(f"{len(pairs)} pairs to morph")

    model = BalancedMorphGenerator()
    model.load(WEIGHTS)
    print(f"Loaded {WEIGHTS}")

    saved, failed = 0, []
    for r in tqdm(pairs, desc="Morphing Utrecht pairs", unit="pair"):
        id_a, id_b = int(r["id_a"]), int(r["id_b"])
        path_a = os.path.join(SRC_DIR, f"image_{id_a}.jpg")
        path_b = os.path.join(SRC_DIR, f"image_{id_b}.jpg")
        out_path = os.path.join(OUT_DIR, f"morph_{id_a}_{id_b}.jpg")
        if os.path.exists(out_path):
            saved += 1
            continue
        try:
            lm_a, lm_b = _load_landmarks(path_a), _load_landmarks(path_b)
            img_a, img_b = _load_image(path_a), _load_image(path_b)

            warped_a, warped_b, mean_lm, mask = morph_pair(
                img_a, img_b, lm_a, lm_b, ALPHA, WIDTH, HEIGHT)
            heatmaps = landmarks_to_heatmaps(mean_lm, WIDTH, HEIGHT)

            morph, _ = model.predict(
                warped_a[None].astype(np.float32), warped_b[None].astype(np.float32),
                heatmaps[None].astype(np.float32), mask[None].astype(np.float32),
                alpha=ALPHA)
            blended = np.clip(morph[0].numpy(), 0.0, 1.0)
            blended = poisson_seam_blend(blended, warped_a, mean_lm, WIDTH, HEIGHT)

            pose_target_lm = (1.0 - POSE_BLEND) * mean_lm + POSE_BLEND * lm_a
            blended = warp_to_landmarks(blended, mean_lm, pose_target_lm, WIDTH, HEIGHT)
            restored = restore_face(blended, upscale=1)

            Image.fromarray((restored * 255).astype(np.uint8)).save(out_path)
            saved += 1
        except Exception as e:                              # noqa: BLE001
            failed.append((id_a, id_b, str(e)))
            tqdm.write(f"[fail] {id_a}+{id_b}: {e}")

    print(f"\nDone. Saved {saved}/{len(pairs)} morphs to {OUT_DIR}/ ({len(failed)} failed)")


if __name__ == "__main__":
    main()
