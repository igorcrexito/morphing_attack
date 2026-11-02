"""Generate a morphed dataset by pairing look-alike faces.

For every top-level image in ``output_dataset`` it picks the most identity-similar
partner (FaceNet embedding) that is also within the shape/pose threshold, then runs
the full morphing pipeline (align -> diffusion model -> Poisson seam blend -> pose
to face A -> GFPGAN restoration) and writes ONLY the final GFPGAN output to
``morphed_dataset``. Pairing the most similar faces yields the most realistic morphs.

Landmarks are read from the per-image ``.csv`` files (68 (x, y) rows in the same
224x224 coordinate space as the ``.jpg``), so no dlib detection is needed here.

    python main_generate_morphed_dataset.py

Alternatively a **fixed pair list** can be supplied with ``--pairs`` (see
``main_syn_mad_pairs.py``). That path reproduces the SYN-MAD 2022 / DCMorph
evaluation protocol: the pairs are decided by the protocol, not by this script's
partner search, so the shape/frontality gates are *not* applied - dropping pairs
would silently change the benchmark. Every pair in the list is attempted and any
failure is reported.

    python main_syn_mad_pairs.py
    python main_generate_morphed_dataset.py --pairs pairs_synmad.csv
"""
import argparse
import csv
import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import yaml

from morphing.warp import (morph_pair, poisson_seam_blend, shape_distance,
                           warp_to_landmarks)
from morphing.dataset import landmarks_to_heatmaps
from model.diffusion_model import DiffusionModel
from image_utils.face_restore import restore_face
from image_utils.face_embedding import embed_paths
from landmarks.landmark import Landmarks

tf.get_logger().setLevel("ERROR")

# landmarked images are read from INPUT_ROOT/<dataset_name>; morphs are written to
# OUTPUT_ROOT/<dataset_name> (the dataset name comes from execution_parameters.yaml).
INPUT_ROOT = "output_dataset"
OUTPUT_ROOT = "morphed_dataset"

# Partner selection: for each face A, pick its most identity-similar face (FaceNet
# cosine) that is also shape-compatible (shape_distance <= shape_threshold). Each
# unordered pair is morphed once (no 4->5 and 5->4). The blended face is always
# warped back to face A's posture. shape_threshold (and the frontality/feather/
# erode knobs) are read from execution_parameters.yaml -> quality_parameters.


def _image_number(path):
    """Extract the integer N from a path like '.../image_123.jpg' for sorting."""
    name = os.path.splitext(os.path.basename(path))[0]   # image_123
    return int(name.split("_")[1])


def _load_image(path, width, height):
    """Return an HxWx3 float image in [0, 1], resized to the model input."""
    img = Image.open(path).convert("RGB").resize((width, height))
    return np.asarray(img).astype(np.float32) / 255.0


def _load_landmarks(jpg_path, width, height):
    """Read the 68x2 landmark array from the .csv next to the image.

    The csv has an 'x,y' header followed by 68 coordinate rows in the image's
    own pixel space. Images are already at model resolution, but we rescale
    defensively in case width/height ever differ from the stored image size.
    """
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
    """image path -> subject identity, from index.csv if the dataset has one.

    Datasets landmarked by the old ``main_save_landmarks.py`` renumber images to
    image_<counter> and lose the identity, so two photos of the *same* person look
    like two different subjects. With `most_similar` pairing that reliably morphs
    someone with their own second capture, which is not an attack at all. When an
    index.csv is present (see main_prepare_frll.py) we use it to forbid those
    pairs; without one we fall back to "every image is its own identity" and warn.
    """
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

    Candidates are every other face that is shape-compatible (shape_distance <=
    SHAPE_THRESHOLD), belongs to a *different* subject (when ``identities`` is
    known) and is not already paired with i. They are ranked by FaceNet
    cosine similarity and the partner is chosen according to ``strategy``:

        most_similar -> highest similarity (closest identity)
        most_distant -> lowest similarity (farthest identity)
        median       -> the 50th-percentile candidate by similarity

    Returns (j, similarity) or None if no candidate qualifies.
    """
    sims = embeddings @ embeddings[i]
    candidates = []
    for j in range(len(embeddings)):
        if j == i:                          # never partner with itself
            continue
        if identities is not None and identities[j] == identities[i]:
            continue                        # never morph a subject with themselves
        if frozenset((i, j)) in used_pairs:
            continue
        if shape_distance(landmarks[i], landmarks[j]) <= shape_threshold:
            candidates.append(j)
    if not candidates:
        return None

    # sort candidates by similarity ascending, then pick by strategy.
    candidates.sort(key=lambda j: sims[j])
    if strategy == "most_similar":
        j = candidates[-1]
    elif strategy == "most_distant":
        j = candidates[0]
    else:                                   # median (50th percentile)
        j = candidates[len(candidates) // 2]
    return j, float(sims[j])


# Landmarked source directories for the SYN-MAD protocol, by expression. Both
# hold image_<identity>.jpg, so the number in a morph filename is the subject id.
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
    """Morph every pair in a protocol-fixed list. Returns (saved, failed)."""
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
            blended = warp_to_landmarks(blended, mean_lm, lm_a, width, height)
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
                    help="subdirectory of morphed_dataset/ for --pairs output")
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

    # blend-error mitigations (no retraining needed): see quality_parameters.
    quality = params.get("quality_parameters", {}) or {}
    shape_threshold = float(quality.get("shape_threshold", 0.12))
    max_yaw = quality.get("max_yaw", None)
    max_yaw = None if max_yaw is None else float(max_yaw)
    feather = int(quality.get("hull_feather", 11))
    erode_iters = int(quality.get("poisson_erode", 1))

    def _load_model():
        m = DiffusionModel()
        weights_path = os.path.join(cache_dir, "morph_model.weights.h5")
        if os.path.exists(weights_path):
            m.load(weights_path)
            print(f"Loaded weights from {weights_path}")
        else:
            print(f"WARNING: {weights_path} not found - using an untrained model "
                  "(output will be the raw aligned blend).")
        return m

    # ---- protocol-fixed pair list (SYN-MAD 2022 / DCMorph comparability) ----
    if args.pairs:
        pairs = _read_pair_list(args.pairs)
        output_root = os.path.join(OUTPUT_ROOT, args.out_name)
        print(f"Pair list '{args.pairs}': {len(pairs)} pairs -> {output_root}/"
              "<expression>/")
        print("Shape/frontality gates are NOT applied: the pair list defines the "
              "benchmark and every pair must be attempted.")
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

    # per-dataset input/output: read landmarked images from output_dataset/<name>,
    # write morphs into morphed_dataset/<name>/<strategy>.
    input_dir = os.path.join(INPUT_ROOT, dataset_name)
    output_dir = os.path.join(OUTPUT_ROOT, dataset_name, strategy)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Dataset '{dataset_name}' [{strategy}]: {input_dir} -> {output_dir}")

    # images with a matching .csv landmark file, in numeric order.
    jpgs = sorted(glob.glob(os.path.join(input_dir, "*.jpg")), key=_image_number)
    jpgs = [p for p in jpgs if os.path.exists(os.path.splitext(p)[0] + ".csv")]

    # precompute landmarks for every image so each face can be paired with its most
    # look-alike (and shape-compatible) partner.
    landmarks = [_load_landmarks(p, width, height) for p in jpgs]

    # frontality gate: drop non-frontal faces before pairing so they never feed a
    # ghost-prone morph (the case behind visible blend seams).
    if max_yaw is not None:
        kept = [(p, lm) for p, lm in zip(jpgs, landmarks)
                if Landmarks.frontality(lm) <= max_yaw]
        dropped = len(jpgs) - len(kept)
        if dropped:
            print(f"Frontality gate: dropped {dropped}/{len(jpgs)} non-frontal "
                  f"faces (max_yaw={max_yaw}).")
        jpgs = [p for p, _ in kept]
        landmarks = [lm for _, lm in kept]

    # subject identities, so a face is never morphed with another capture of
    # itself (None when the dataset carries no index.csv).
    identities = _read_identities(input_dir, jpgs)

    # FaceNet identity embeddings for the surviving faces.
    embeddings = embed_paths(jpgs, width)

    # load the trained model once.
    model = _load_model()

    used_pairs = set()                      # frozenset({i, j}) already morphed
    saved = 0
    skipped = 0
    for i, path_a in enumerate(tqdm(jpgs, desc="Morphing faces", unit="face")):
        try:
            # partner = the shape-compatible face chosen by the embedding-distance
            # strategy, skipping any pair already morphed in the other direction.
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

            # always pose the blend to face A: warp from the averaged mean shape
            # onto A's own landmarks so the morph adopts A's posture.
            blended = warp_to_landmarks(blended, mean_lm, lm_a, width, height)

            # final realism pass: this GFPGAN output is the only thing we store.
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
