"""Run the full morphing inference pipeline on user-supplied images.

Unlike main_inference.py (which consumes the pre-landmarked output_dataset), this
script starts from raw images dropped into the ``user_test/`` folder and performs
landmark detection itself before morphing. It then shows exactly the same outputs
as main_inference.py: the printed per-zone cosine similarities and the three plots
(morphing result, zone crops, zone-embedding PCA + similarity heatmap).

    # morph the first two images found in user_test/
    python user_test_images.py

    # or pick two explicit images
    python user_test_images.py user_test/alice.jpg user_test/bob.jpg
"""
import os
import sys
import glob
import yaml
import numpy as np
import tensorflow as tf
from PIL import Image

from landmarks.landmark import Landmarks
from morphing.warp import morph_pair, poisson_seam_blend, warp_to_landmarks
from morphing.dataset import landmarks_to_heatmaps
from model.diffusion_model import DiffusionModel
from image_utils.face_restore import restore_face

# reuse the analysis + plotting helpers so the output matches main_inference.py.
from main_inference import (LANDMARK_ZONES, _zone_embeddings,
                            _cosine_similarity, plot_morphing_results,
                            plot_zone_crops, plot_zone_embeddings)

tf.get_logger().setLevel("ERROR")

# raw, un-landmarked images live here.
USER_TEST_DIR = "user_test"


def _list_user_images(folder):
    """Return image paths in ``folder`` sorted by name (jpg/jpeg/png)."""
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def main():
    with open("execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    width = int(params["image_parameters"]["image_width"])
    height = int(params["image_parameters"]["image_height"])
    alpha = float(params["morphing_parameters"]["alpha"])
    cache_dir = str(params["dataset_parameters"]["cache_dir"])

    # blend-error mitigations (no retraining needed): see quality_parameters.
    quality = params.get("quality_parameters", {}) or {}
    max_yaw = quality.get("max_yaw", None)
    max_yaw = None if max_yaw is None else float(max_yaw)
    feather = int(quality.get("hull_feather", 11))
    erode_iters = int(quality.get("poisson_erode", 1))
    align_faces = bool(quality.get("align_faces", False))

    # explicitly choose the two faces (CLI: path_a path_b, else the first two
    # images found in user_test/).
    if len(sys.argv) >= 3:
        path_a, path_b = sys.argv[1], sys.argv[2]
    else:
        imgs = _list_user_images(USER_TEST_DIR)
        if len(imgs) < 2:
            raise SystemExit(f"Need at least 2 images in '{USER_TEST_DIR}/' "
                             f"(found {len(imgs)}).")
        path_a, path_b = imgs[0], imgs[1]
    print(f"User test images\nA: {path_a}\nB: {path_b}")

    # --- landmark computation: detect 68 landmarks on the raw images (this is the
    # step main_inference.py relies on the cached .csv files for). generate_landmarks
    # also returns the model-resolution colour image (aligned+cropped when
    # align_faces is on, else a plain resize), keeping pixels and landmarks in sync.
    landmark_descriptor = Landmarks(number_of_landmarks=68, max_yaw=max_yaw,
                                    align=align_faces)
    raw_a = np.array(Image.open(path_a).convert("RGB"))
    raw_b = np.array(Image.open(path_b).convert("RGB"))
    out_a, lm_a = landmark_descriptor.generate_landmarks(raw_a, 3, width, height)
    out_b, lm_b = landmark_descriptor.generate_landmarks(raw_b, 3, width, height)

    # color images at model resolution (for warping + display)
    img_a = np.asarray(out_a).astype(np.float32) / 255.0
    img_b = np.asarray(out_b).astype(np.float32) / 255.0

    # align both faces to the mean shape, build network inputs
    warped_a, warped_b, mean_lm, mask = morph_pair(
        img_a, img_b, lm_a, lm_b, alpha, width, height, feather=feather)
    heatmaps = landmarks_to_heatmaps(mean_lm, width, height)

    # add batch dimension
    wA = warped_a[None].astype(np.float32)
    wB = warped_b[None].astype(np.float32)
    hm = heatmaps[None].astype(np.float32)
    mk = mask[None].astype(np.float32)

    # load the trained model
    model = DiffusionModel()
    weights_path = os.path.join(cache_dir, "morph_model.weights.h5")
    if os.path.exists(weights_path):
        model.load(weights_path)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"WARNING: {weights_path} not found - using an untrained model "
              "(output will be the raw aligned blend). Train first with "
              "main_train_model.py.")

    morph, _ = model.predict(wA, wB, hm, mk, alpha=alpha)
    blended = np.clip(morph[0].numpy(), 0.0, 1.0)

    # gradient-domain seam removal: re-integrate the morphed face over the
    # single-identity background so the hull boundary is invisible.
    blended = poisson_seam_blend(blended, warped_a, mean_lm, width, height,
                                 erode_iters=erode_iters)

    # always pose the blend to face A: warp from the averaged mean shape onto A's
    # own landmarks so the morph adopts A's posture.
    blended = warp_to_landmarks(blended, mean_lm, lm_a, width, height)

    # final realism pass: GFPGAN re-synthesizes photographic facial detail.
    blended = restore_face(blended, upscale=1)

    # compute zone embeddings on the warped/aligned images so landmarks align
    emb_a     = _zone_embeddings(warped_a, mean_lm)
    emb_b     = _zone_embeddings(warped_b, mean_lm)
    emb_blend = _zone_embeddings(blended,  mean_lm)

    # print cosine similarities
    header = f"{'Zone':<16} {'A vs B':>8} {'A vs blend':>11} {'B vs blend':>11}"
    print(f"\n--- Cosine similarity (zone embeddings) ---\n{header}")
    for zname in LANDMARK_ZONES:
        sab  = _cosine_similarity(emb_a[zname], emb_b[zname])
        sabl = _cosine_similarity(emb_a[zname], emb_blend[zname])
        sbbl = _cosine_similarity(emb_b[zname], emb_blend[zname])
        print(f"{zname:<16} {sab:>8.4f} {sabl:>11.4f} {sbbl:>11.4f}")

    # plot morphing result
    plot_morphing_results(img_a, img_b, blended, alpha)

    # plot zone crops for visual inspection
    plot_zone_crops(img_a, img_b, blended, lm_a, lm_b, mean_lm)

    # plot zone embedding PCA and cosine similarity heatmap
    plot_zone_embeddings(emb_a, emb_b, emb_blend)


if __name__ == "__main__":
    main()
