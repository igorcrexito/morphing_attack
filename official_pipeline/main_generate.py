"""official_pipeline/main_generate.py

*** Generates morphs with the official / recommended pipeline for this project. ***

Near-identical copy of ../main_generate_morphed_dataset.py. Generates morphs
the SAME way as the original (align -> neural network -> Poisson seam blend ->
pose back toward identity A -> GFPGAN restoration) - the differences are
marked below with "=== CHANGED ===":

  1. It loads official_pipeline.model.BalancedMorphGenerator (the
     fixed model) instead of the original model.diffusion_model.DiffusionModel,
     and by default looks for its weights at
     cache/morph_model_official.weights.h5 (produced by
     official_pipeline/main_train.py), not cache/morph_model.weights.h5.
  2. Output goes to morphed_dataset_official/ instead of morphed_dataset/, so
     running this script can NEVER overwrite the original pipeline's output -
     both datasets exist side by side and can be evaluated and compared
     directly (see official_pipeline/README.md "How to run").
  3. === FIX #4 (simple version) === The original always warps the finished
     morph ALL THE WAY onto identity A's own landmarks (`warp_to_landmarks(...,
     mean_lm, lm_a, ...)`). A quick diagnostic (see official_pipeline/README.md,
     section "FIX #4") measured that this single step roughly TRIPLES the
     identity-balance gap between the morph's similarity to A and to B -
     it is the single biggest remaining source of the A-bias, bigger than the
     other 3 fixes combined. The full fix (training the network through a
     differentiable version of this warp) is out of scope for "simple" - see
     model.py's docstring. This is the cheap, effective compromise:
     instead of warping fully onto A's geometry, warp only PART of the way
     there (POSE_BLEND below), landing on a geometry BETWEEN the identity-
     neutral mean shape and A's own posture. At POSE_BLEND=1.0 this is
     byte-for-byte the original behaviour; POSE_BLEND=0.0 leaves the morph in
     the mean-shape geometry (never posed toward A at all).

Everything else - argument parsing, the partner-selection logic, the
--pairs protocol-fixed mode, the frontality/shape gates - is copied verbatim.

Usage (run from the PROJECT ROOT, exactly like the original):
    # reproduce the SYN-MAD 2022 benchmark with the balanced model:
    python official_pipeline/main_generate.py --pairs pairs_synmad.csv --out-name frll_synmad

    # or the default partner-search mode, same as the original:
    python official_pipeline/main_generate.py
"""
import argparse
import csv
import os
import sys
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import yaml

# Make the project root importable (morphing/, model/, image_utils/,
# landmarks/ all live one level up) - same convention as evaluation/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from morphing.warp import (morph_pair, poisson_seam_blend, shape_distance,
                           warp_to_landmarks)
from morphing.dataset import landmarks_to_heatmaps
# === CHANGED: balanced model instead of the original DiffusionModel =========
from official_pipeline.model import BalancedMorphGenerator
from image_utils.face_restore import restore_face
from image_utils.face_embedding import embed_paths
from landmarks.landmark import Landmarks

tf.get_logger().setLevel("ERROR")

# landmarked images are read from INPUT_ROOT/<dataset_name> - identical to
# the original, since the fixes don't touch data preparation at all.
INPUT_ROOT = "output_dataset"
# === CHANGED: separate output tree so this never collides with / overwrites
# the original pipeline's morphed_dataset/ folder.
OUTPUT_ROOT = "morphed_dataset_official"

# === FIX #4 (simple version, tunable) =======================================
# How far to warp the finished morph from the identity-neutral mean-shape
# geometry toward identity A's own landmarks, before GFPGAN restoration.
#   1.0 = original behaviour: fully A's geometry (strongest remaining source
#         of the A-identity bias - see the module docstring above).
#   0.0 = leave the morph in the mean-shape geometry entirely (most balanced,
#         but the output is an "averaged" face shape, not a normal single-
#         person portrait - may look slightly less passport-plausible).
#   0.5 = the default here: a genuine compromise between the two.
# Re-tune by regenerating a small batch at a few values and checking both
# evaluation/02_vulnerability_eval.py's score gap and visual plausibility.
POSE_BLEND = 0.5


def _image_number(path):
    """Extract the integer N from a path like '.../image_123.jpg' for sorting."""
    name = os.path.splitext(os.path.basename(path))[0]
    return int(name.split("_")[1])


def _load_image(path, width, height):
    """Return an HxWx3 float image in [0, 1], resized to the model input."""
    img = Image.open(path).convert("RGB").resize((width, height))
    return np.asarray(img).astype(np.float32) / 255.0


def _load_landmarks(jpg_path, width, height):
    """Read the 68x2 landmark array from the .csv next to the image."""
    csv_path = os.path.splitext(jpg_path)[0] + ".csv"
    lm = np.loadtxt(csv_path, delimiter=",", skiprows=1).astype(np.float32)
    with Image.open(jpg_path) as im:
        src_w, src_h = im.size
    if (src_w, src_h) != (width, height):
        lm[:, 0] *= width / src_w
        lm[:, 1] *= height / src_h
    return lm


VALID_STRATEGIES = ("most_similar", "most_distant", "median")


def _read_identities(input_dir, jpgs):
    """image path -> subject identity, from index.csv if the dataset has one."""
    index_path = os.path.join(input_dir, "index.csv")
    if not os.path.exists(index_path):
        print(f"NOTE: no {index_path}; assuming one image per subject. If this "
              "dataset has several captures per person, same-identity pairs "
              "cannot be excluded.")
        return None
    with open(index_path, newline="") as f:
        by_name = {r["image"]: r["identity"] for r in csv.DictReader(f)}
    return [by_name.get(os.path.basename(p), os.path.basename(p)) for p in jpgs]


def _select_partner(i, embeddings, landmarks, used_pairs, strategy, shape_threshold,
                    identities=None):
    """Pick a shape-compatible partner for image i by embedding-distance strategy.
    Identical logic to the original script."""
    sims = embeddings @ embeddings[i]
    candidates = []
    for j in range(len(embeddings)):
        if j == i:
            continue
        if identities is not None and identities[j] == identities[i]:
            continue
        if frozenset((i, j)) in used_pairs:
            continue
        if shape_distance(landmarks[i], landmarks[j]) <= shape_threshold:
            candidates.append(j)
    if not candidates:
        return None

    candidates.sort(key=lambda j: sims[j])
    if strategy == "most_similar":
        j = candidates[-1]
    elif strategy == "most_distant":
        j = candidates[0]
    else:
        j = candidates[len(candidates) // 2]
    return j, float(sims[j])


EXPRESSION_DIRS = {
    "neutral": os.path.join(INPUT_ROOT, "frll"),
    "smiling": os.path.join(INPUT_ROOT, "frll_probe"),
}


def _read_pair_list(path):
    """Read a pairs csv (id_a, id_b, and optionally expression) into a list."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    pairs = []
    for r in rows:
        expression = r.get("expression", "neutral")
        src_dir = EXPRESSION_DIRS.get(expression)
        if src_dir is None:
            raise SystemExit(f"unknown expression '{expression}' in {path}")
        pairs.append((int(r["id_a"]), int(r["id_b"]), expression, src_dir))
    return pairs


def _morph_from_pair_list(pairs, model, params_bundle, output_root):
    """Morph every pair in a protocol-fixed list. Returns (saved, failed).
    Identical logic to the original script - only `model` differs."""
    (width, height, alpha, feather, erode_iters) = params_bundle
    saved, failures = 0, []
    for id_a, id_b, expression, src_dir in tqdm(pairs, desc="Morphing pairs",
                                                unit="pair"):
        path_a = os.path.join(src_dir, f"image_{id_a}.jpg")
        path_b = os.path.join(src_dir, f"image_{id_b}.jpg")
        out_dir = os.path.join(output_root, expression)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"morph_{id_a}_{id_b}.jpg")
        if os.path.exists(out_path):
            saved += 1
            continue
        try:
            lm_a = _load_landmarks(path_a, width, height)
            lm_b = _load_landmarks(path_b, width, height)
            img_a = _load_image(path_a, width, height)
            img_b = _load_image(path_b, width, height)

            warped_a, warped_b, mean_lm, mask = morph_pair(
                img_a, img_b, lm_a, lm_b, alpha, width, height, feather=feather)
            heatmaps = landmarks_to_heatmaps(mean_lm, width, height)

            morph, _ = model.predict(
                warped_a[None].astype(np.float32),
                warped_b[None].astype(np.float32),
                heatmaps[None].astype(np.float32),
                mask[None].astype(np.float32),
                alpha=alpha)
            blended = np.clip(morph[0].numpy(), 0.0, 1.0)
            blended = poisson_seam_blend(blended, warped_a, mean_lm, width, height,
                                         erode_iters=erode_iters)
            # === FIX #4 (simple version): warp PART WAY toward A's geometry
            # instead of all the way - see POSE_BLEND above.
            pose_target_lm = (1.0 - POSE_BLEND) * mean_lm + POSE_BLEND * lm_a
            blended = warp_to_landmarks(blended, mean_lm, pose_target_lm, width, height)
            restored = restore_face(blended, upscale=1)
            Image.fromarray((restored * 255.0).astype(np.uint8)).save(out_path)
            saved += 1
        except Exception as e:                           # noqa: BLE001
            failures.append((expression, id_a, id_b, str(e)))
            tqdm.write(f"[fail] {expression} {id_a}+{id_b}: {e}")
    return saved, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=None,
                    help="csv of protocol-fixed morph pairs (id_a, id_b, "
                         "expression); disables the built-in partner search")
    ap.add_argument("--out-name", default="frll_synmad",
                    help="subdirectory of morphed_dataset_official/ for --pairs output")
    args = ap.parse_args()

    with open("execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    width = int(params["image_parameters"]["image_width"])
    height = int(params["image_parameters"]["image_height"])
    alpha = float(params["morphing_parameters"]["alpha"])
    strategy = str(params["morphing_parameters"]["morph_strategy"])
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"morph_strategy must be one of {VALID_STRATEGIES}, "
                         f"got '{strategy}'.")
    cache_dir = str(params["dataset_parameters"]["cache_dir"])
    dataset_name = str(params["dataset_parameters"]["dataset_name"])

    quality = params.get("quality_parameters", {}) or {}
    shape_threshold = float(quality.get("shape_threshold", 0.12))
    max_yaw = quality.get("max_yaw", None)
    max_yaw = None if max_yaw is None else float(max_yaw)
    feather = int(quality.get("hull_feather", 11))
    erode_iters = int(quality.get("poisson_erode", 1))

    def _load_model():
        # === CHANGED: balanced model + its own weights file. ===============
        m = BalancedMorphGenerator()
        weights_path = os.path.join(cache_dir, "morph_model_official.weights.h5")
        if os.path.exists(weights_path):
            m.load(weights_path)
            print(f"Loaded balanced weights from {weights_path}")
        else:
            print(f"WARNING: {weights_path} not found - using an untrained model "
                  "(output will be the raw aligned blend). Train first with "
                  "official_pipeline/main_train.py.")
        return m

    if args.pairs:
        pairs = _read_pair_list(args.pairs)
        output_root = os.path.join(OUTPUT_ROOT, args.out_name)
        print(f"Pair list '{args.pairs}': {len(pairs)} pairs -> {output_root}/"
              "<expression>/")
        saved, failures = _morph_from_pair_list(
            pairs, _load_model(),
            (width, height, alpha, feather, erode_iters), output_root)
        print(f"\nDone. Saved {saved}/{len(pairs)} morphs to {output_root}/ "
              f"({len(failures)} failed).")
        if failures:
            log = os.path.join(output_root, "failures.csv")
            with open(log, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["expression", "id_a", "id_b", "error"])
                w.writerows(failures)
            print(f"Failure log: {log}")
        return

    input_dir = os.path.join(INPUT_ROOT, dataset_name)
    output_dir = os.path.join(OUTPUT_ROOT, dataset_name, strategy)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Dataset '{dataset_name}' [{strategy}]: {input_dir} -> {output_dir}")

    jpgs = sorted(glob.glob(os.path.join(input_dir, "*.jpg")), key=_image_number)
    jpgs = [p for p in jpgs if os.path.exists(os.path.splitext(p)[0] + ".csv")]

    landmarks = [_load_landmarks(p, width, height) for p in jpgs]

    if max_yaw is not None:
        kept = [(p, lm) for p, lm in zip(jpgs, landmarks)
                if Landmarks.frontality(lm) <= max_yaw]
        dropped = len(jpgs) - len(kept)
        if dropped:
            print(f"Frontality gate: dropped {dropped}/{len(jpgs)} non-frontal "
                  f"faces (max_yaw={max_yaw}).")
        jpgs = [p for p, _ in kept]
        landmarks = [lm for _, lm in kept]

    identities = _read_identities(input_dir, jpgs)
    embeddings = embed_paths(jpgs, width)
    model = _load_model()

    used_pairs = set()
    saved = 0
    skipped = 0
    for i, path_a in enumerate(tqdm(jpgs, desc="Morphing faces", unit="face")):
        try:
            pick = _select_partner(i, embeddings, landmarks, used_pairs, strategy,
                                   shape_threshold, identities)
            if pick is None:
                skipped += 1
                tqdm.write(f"[skip] {os.path.basename(path_a)}: no new "
                           f"shape-compatible partner.")
                continue
            j, sim = pick
            used_pairs.add(frozenset((i, j)))
            path_b = jpgs[j]
            lm_a, lm_b = landmarks[i], landmarks[j]

            img_a = _load_image(path_a, width, height)
            img_b = _load_image(path_b, width, height)

            warped_a, warped_b, mean_lm, mask = morph_pair(
                img_a, img_b, lm_a, lm_b, alpha, width, height, feather=feather)
            heatmaps = landmarks_to_heatmaps(mean_lm, width, height)

            morph, _ = model.predict(
                warped_a[None].astype(np.float32),
                warped_b[None].astype(np.float32),
                heatmaps[None].astype(np.float32),
                mask[None].astype(np.float32),
                alpha=alpha)
            blended = np.clip(morph[0].numpy(), 0.0, 1.0)
            blended = poisson_seam_blend(blended, warped_a, mean_lm, width, height,
                                         erode_iters=erode_iters)
            # === FIX #4 (simple version): warp PART WAY toward A's geometry
            # instead of all the way - see POSE_BLEND above.
            pose_target_lm = (1.0 - POSE_BLEND) * mean_lm + POSE_BLEND * lm_a
            blended = warp_to_landmarks(blended, mean_lm, pose_target_lm, width, height)
            restored = restore_face(blended, upscale=1)

            num_a, num_b = _image_number(path_a), _image_number(path_b)
            out_path = os.path.join(output_dir, f"morph_{num_a}_{num_b}.jpg")
            Image.fromarray((restored * 255.0).astype(np.uint8)).save(out_path)
            saved += 1
        except Exception as e:                       # noqa: BLE001
            tqdm.write(f"[skip] {os.path.basename(path_a)} + "
                       f"{os.path.basename(path_b)}: {e}")

    print(f"\nDone. Saved {saved}/{len(jpgs)} morphs to {output_dir}/ "
          f"({skipped} skipped with no shape-compatible partner).")


if __name__ == "__main__":
    main()
