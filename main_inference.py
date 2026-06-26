import os
import sys
import yaml
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA

from landmarks.landmark import Landmarks
from morphing.warp import morph_pair, poisson_seam_blend, warp_to_landmarks
from morphing.dataset import landmarks_to_heatmaps
from model.diffusion_model import DiffusionModel
from image_utils.face_restore import restore_face

tf.get_logger().setLevel("ERROR")

# --- explicitly choose the two faces to morph (or pass two paths on the CLI) --
DEFAULT_DIR = "output_dataset/1000"
DEFAULT_A = "image_8.jpg"
DEFAULT_B = "image_3668.jpg"

# Facial zones defined by 68-landmark indices
LANDMARK_ZONES = {
    "jaw":           list(range(0, 17)),
    "left_eyebrow":  list(range(17, 22)),
    "right_eyebrow": list(range(22, 27)),
    "nose":          list(range(27, 36)),
    "left_eye":      list(range(36, 42)),
    "right_eye":     list(range(42, 48)),
    "mouth":         list(range(48, 68)),
}

ZONE_PATCH_SIZE = 32  # each crop is resized to this before flattening

# The blended face is always warped back to face A's posture.


def _load_image(path, width, height):
    """Return an HxWx3 float image in [0, 1], resized to the model input."""
    img = Image.open(path).convert("RGB").resize((width, height))
    return np.asarray(img).astype(np.float32) / 255.0


def _crop_zone(img, landmarks, indices, pad=4):
    """Return a fixed-size crop of the zone defined by landmark indices."""
    pts = landmarks[indices]
    x0, y0 = pts[:, 0].min() - pad, pts[:, 1].min() - pad
    x1, y1 = pts[:, 0].max() + pad, pts[:, 1].max() + pad
    h, w = img.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        crop = np.zeros((ZONE_PATCH_SIZE, ZONE_PATCH_SIZE, 3), dtype=np.float32)
    patch = np.array(Image.fromarray((crop * 255).astype(np.uint8))
                     .resize((ZONE_PATCH_SIZE, ZONE_PATCH_SIZE))) / 255.0
    return patch.astype(np.float32)


def _zone_embeddings(img, landmarks):
    """Return a dict of zone_name -> flat embedding vector."""
    return {
        name: _crop_zone(img, landmarks, indices).flatten()
        for name, indices in LANDMARK_ZONES.items()
    }


def _cosine_similarity(u, v):
    u, v = u / (np.linalg.norm(u) + 1e-8), v / (np.linalg.norm(v) + 1e-8)
    return float(np.dot(u, v))


def plot_morphing_results(img_a, img_b, blended, alpha):
    images = [img_a, img_b, blended]
    titles = ["image_A", "image_B", f"blended (alpha={alpha})"]

    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    for ax, im, title in zip(axes, images, titles):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.suptitle("Morphing Result", y=1.02, fontsize=13, fontweight="bold")
    plt.show()


def plot_zone_embeddings(emb_a, emb_b, emb_blend):
    """PCA scatter of zone embeddings and cosine-similarity heatmap."""
    zone_names = list(LANDMARK_ZONES.keys())

    color_map = {"A": "#2196F3", "B": "#FF5722", "blend": "#4CAF50"}
    marker_map = {"A": "o", "B": "s", "blend": "^"}

    sources_list = [(emb_a, "A"), (emb_b, "B"), (emb_blend, "blend")]

    # --- build embedding matrix ---
    rows, labels, sources = [], [], []
    for zname in zone_names:
        for emb, src in sources_list:
            rows.append(emb[zname])
            labels.append(zname)
            sources.append(src)

    X = np.array(rows)
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    # --- cosine similarity per zone ---
    sim_rows, row_labels = [], []
    pairs = [("A vs B", emb_a, emb_b),
             ("A vs blend", emb_a, emb_blend),
             ("B vs blend", emb_b, emb_blend)]
    for label, ea, eb in pairs:
        sim_rows.append([_cosine_similarity(ea[z], eb[z]) for z in zone_names])
        row_labels.append(label)

    sim_matrix = np.array(sim_rows)

    fig = plt.figure(figsize=(18, max(7, 4 + len(row_labels))))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ---- left: PCA scatter ----
    ax_pca = fig.add_subplot(gs[0])
    for i, (x, y) in enumerate(X2):
        src = sources[i]
        ax_pca.scatter(x, y, color=color_map[src], marker=marker_map[src],
                       s=90, zorder=3)
        ax_pca.annotate(labels[i], (x, y), fontsize=7, ha="left",
                        xytext=(3, 3), textcoords="offset points")

    for src, color in color_map.items():
        if src in set(sources):
            ax_pca.scatter([], [], color=color, label=src,
                           marker=marker_map[src], s=80)
    ax_pca.legend(title="Image", fontsize=9)
    ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax_pca.set_title("Zone Embeddings — PCA Projection")
    ax_pca.grid(True, alpha=0.3)

    # ---- right: cosine similarity heatmap ----
    ax_sim = fig.add_subplot(gs[1])
    im = ax_sim.imshow(sim_matrix, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax_sim.set_xticks(range(len(zone_names)))
    ax_sim.set_xticklabels(zone_names, rotation=35, ha="right", fontsize=9)
    ax_sim.set_yticks(range(len(row_labels)))
    ax_sim.set_yticklabels(row_labels, fontsize=9)
    ax_sim.set_title("Cosine Similarity per Facial Zone")

    for r in range(sim_matrix.shape[0]):
        for c in range(sim_matrix.shape[1]):
            ax_sim.text(c, r, f"{sim_matrix[r, c]:.3f}", ha="center", va="center",
                        fontsize=8, color="black")

    plt.colorbar(im, ax=ax_sim, label="cosine similarity")
    plt.suptitle("Landmark Zone Embedding Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_zone_crops(img_a, img_b, blended, lm_a, lm_b, mean_lm):
    """Grid of zone crops for all images."""
    zone_names = list(LANDMARK_ZONES.keys())
    n_zones = len(zone_names)

    image_rows = [
        (img_a,   lm_a,    "A"),
        (img_b,   lm_b,    "B"),
        (blended, mean_lm, "blend"),
    ]

    n_rows = len(image_rows)
    fig, axes = plt.subplots(n_rows, n_zones, figsize=(n_zones * 2, n_rows * 2))

    for col, zname in enumerate(zone_names):
        indices = LANDMARK_ZONES[zname]
        for row, (img, lm, title) in enumerate(image_rows):
            crop = _crop_zone(img, lm, indices)
            axes[row, col].imshow(crop)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(zname, fontsize=8)
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=9)

    plt.suptitle("Facial Zone Crops", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    with open("execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    width = int(params["image_parameters"]["image_width"])
    height = int(params["image_parameters"]["image_height"])
    alpha = float(params["morphing_parameters"]["alpha"])
    cache_dir = str(params["dataset_parameters"]["cache_dir"])

    # explicitly choose the two faces (CLI: path_a path_b, else the defaults)
    if len(sys.argv) >= 3:
        path_a, path_b = sys.argv[1], sys.argv[2]
    else:
        path_a = os.path.join(DEFAULT_DIR, DEFAULT_A)
        path_b = os.path.join(DEFAULT_DIR, DEFAULT_B)
    print(f"A: {path_a}\nB: {path_b}")

    # detect landmarks on the (original-resolution) images, scaled to model input
    landmark_descriptor = Landmarks(number_of_landmarks=68)
    raw_a = np.array(Image.open(path_a).convert("RGB"))
    raw_b = np.array(Image.open(path_b).convert("RGB"))
    _, lm_a = landmark_descriptor.generate_landmarks(raw_a, 3, width, height)
    _, lm_b = landmark_descriptor.generate_landmarks(raw_b, 3, width, height)

    # color images at model resolution (for warping + display)
    img_a = _load_image(path_a, width, height)
    img_b = _load_image(path_b, width, height)

    # align both faces to the mean shape, build network inputs
    warped_a, warped_b, mean_lm, mask = morph_pair(
        img_a, img_b, lm_a, lm_b, alpha, width, height)
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
    blended = poisson_seam_blend(blended, warped_a, mean_lm, width, height)

    # always pose the blend to face A: warp from the averaged mean shape onto A's
    # own landmarks so the morph adopts A's posture.
    blended = warp_to_landmarks(blended, mean_lm, lm_a, width, height)

    # final realism pass: GFPGAN re-synthesizes photographic facial detail
    # (skin/eyes/hair) that the upscale + L2 losses leave soft. No-op if GFPGAN
    # isn't installed (see image_utils/face_restore.py).
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
