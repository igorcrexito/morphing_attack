"""Generate a morphed dataset by pairing look-alike faces.

For every top-level image in ``output_dataset`` it picks the most identity-similar
partner (FaceNet embedding) that is also within the shape/pose threshold, then runs
the full morphing pipeline (align -> diffusion model -> Poisson seam blend -> pose
to face A -> GFPGAN restoration) and writes ONLY the final GFPGAN output to
``morphed_dataset``. Pairing the most similar faces yields the most realistic morphs.

Landmarks are read from the per-image ``.csv`` files (68 (x, y) rows in the same
224x224 coordinate space as the ``.jpg``), so no dlib detection is needed here.

    python main_generate_morphed_dataset.py
"""
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


def _select_partner(i, embeddings, landmarks, used_pairs, strategy, shape_threshold):
    """Pick a shape-compatible partner for image i by embedding-distance strategy.

    Candidates are every other face that is shape-compatible (shape_distance <=
    SHAPE_THRESHOLD) and not already paired with i. They are ranked by FaceNet
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


def main():
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

    # FaceNet identity embeddings for the surviving faces.
    embeddings = embed_paths(jpgs, width)

    # load the trained model once.
    model = DiffusionModel()
    weights_path = os.path.join(cache_dir, "morph_model.weights.h5")
    if os.path.exists(weights_path):
        model.load(weights_path)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"WARNING: {weights_path} not found - using an untrained model "
              "(output will be the raw aligned blend).")

    used_pairs = set()                      # frozenset({i, j}) already morphed
    saved = 0
    skipped = 0
    for i, path_a in enumerate(tqdm(jpgs, desc="Morphing faces", unit="face")):
        try:
            # partner = the shape-compatible face chosen by the embedding-distance
            # strategy, skipping any pair already morphed in the other direction.
            pick = _select_partner(i, embeddings, landmarks, used_pairs, strategy,
                                   shape_threshold)
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
