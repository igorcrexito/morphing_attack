# Morphing Attack Generation Pipeline

A face-morphing system that fuses two distinct identities (A and B) into a single
synthetic face that a face-recognition system will accept as either of them — the
canonical *morphing attack* used to probe the robustness of biometric
verification (e.g. e-passport gates).

The approach is **hybrid**: the geometry is solved *classically* (landmark
detection + piecewise-affine warping onto a common shape), and the photometry is
solved *neurally* (a U-Net residual refiner trained with reconstruction,
perceptual, identity-balance and adversarial objectives), with a final
gradient-domain composite and an optional GFPGAN restoration pass for
photographic realism.

The core idea is a clean separation of concerns:

> **Classical methods handle "where pixels go" (alignment); the network only
> handles "what color they should be" (a small residual correction).**

This keeps the network's job easy — it never has to hallucinate large geometric
motion — which is what makes the morph sharp and identity-balanced.

---

## Pipeline

The system runs as four stages, three of which have a dedicated entry-point
script. Run them in order.

### 1. Landmark extraction — `main_save_landmarks.py`

Every source image is loaded, the face is detected, and 68 facial landmarks are
located. The image is resized to the model resolution (224×224, must be divisible
by 8 for the U-Net's three pooling levels) and saved alongside its landmark
coordinates as a `.csv`.

- **Face detection & landmarks** — `dlib`'s HOG frontal-face detector
  (`get_frontal_face_detector`) followed by the 68-point shape predictor
  (`shape_predictor_68_face_landmarks.dat`). See `landmarks/landmark.py`.
- **Coordinate rescaling** — landmarks are detected on the original-resolution
  image and then rescaled by `width/orig_width`, `height/orig_height` so they
  line up with the resized image that is actually fed to the network.
- Images with no detectable face are skipped (the morph requires a reliable
  landmark correspondence).

**Output:** `output_dataset/<n>/image_k.jpg` + `image_k.csv` — one landmarked
image per identity.

### 2. Pairing + warp caching — `main_prepare_pairs.py`

Defines *which* identities get morphed together and precomputes the expensive,
CPU-bound warping once so training never repeats it.

- **Leakage-free split** (`morphing/pairing.py`) — images (identities) are split
  into disjoint train/val sets *first*, and pairs are only formed *within* each
  set. No identity ever appears in both splits, so validation loss is honest.
- **Pair construction** — random distinct unordered pairs of two different
  identities (no self-pairs, no duplicates), capped by `max_pairs` /
  `max_val_pairs`. `build_pairs` is the hook for a deliberate policy (e.g.
  same-gender / same-pose) once that metadata is available.
- **Geometric alignment** (`morphing/warp.py`, the heart of the method) — for
  each pair:
  1. A common **mean shape** is defined as
     `mean = (1 − α)·landmarks_A + α·landmarks_B`.
  2. Eight image-border anchor points are appended so the warp is defined over
     the whole frame, not just the face's convex hull.
  3. A **Delaunay triangulation** is built over the mean-shape points
     (`cv2.Subdiv2D`).
  4. Each triangle is warped from A's shape (and B's shape) onto the mean shape
     via a **per-triangle affine transform** (`cv2.getAffineTransform` +
     `warpAffine`). After this, the eyes/nose/mouth of A and B sit at *identical*
     pixel coordinates — so a later cross-dissolve no longer ghosts.
  5. **Photometric harmonization** (`match_color`) — B's tone is matched to A
     inside the face (per-channel mean/std alignment in CIE-Lab) so the blend
     doesn't bake in a two-tone skin/contrast seam.
  6. A **feathered convex-hull mask** of the face is produced — this is where the
     refinement happens; outside it the result is a single identity.
- **Caching** (`morphing/dataset.py`) — the aligned `warpedA`, `warpedB`, hull
  `mask`, and `mean_lm` tensors are written to one compressed `.npz` per split.
  The 68-channel landmark heatmaps are *deliberately not cached* (≈13 MB/sample);
  they are tiny to regenerate from `mean_lm` and are produced on the fly per
  batch.

**Output:** `cache/train.npz`, `cache/val.npz`.

### 3. Training the morph refiner — `main_train_model.py`

Trains the U-Net residual generator (`model/diffusion_model.py`). The name
"DiffusionModel" is historical — it is **not** a denoising-diffusion model but a
U-Net **residual refiner**.

- **Inputs (75 channels)** — `warped_A` (3) + `warped_B` (3) + 68 landmark
  heatmaps + hull mask (1). The heatmaps give the network explicit spatial
  awareness of facial structure.
- **Architecture** — a fully-convolutional 3-level encoder/decoder U-Net with
  skip connections, swish activations, and batch norm. It is *resolution-
  agnostic* (spatial dims are `None`), so the same weights train/infer at 224,
  256 or 512. The output head is a `tanh` **residual delta**, not a full image.
- **Image formation** (`_build_morph`) — the network never paints the whole face
  from scratch:
  ```
  blended = (1 − α)·warped_A + α·warped_B          # clean, because aligned
  base    = mask·blended + (1 − mask)·warped_A      # outside hull = identity A
  morph   = clip( base + delta·mask )               # refine only inside the hull
  ```
  Outside the hull (hair, ears, neck, background) the two identities have no
  landmark correspondence, so they are *not* blended — that region is taken from
  a single identity to avoid ghosting.
- **Loss** (`_balanced_loss`), a weighted sum of:
  - **Identity-balance loss (×25)** — the dominant term. FaceNet (`keras_facenet`)
    embeddings of the morph are pulled toward the α-weighted blend of A's and B's
    embeddings (cosine). This is what makes the morph match *both* identities —
    the actual attack objective. ImageNet features can't do this; a face-trained
    embedder is required.
  - **Perceptual loss (×6)** — VGG16 feature-space distance to A and B (α-weighted)
    for texture realism.
  - **Reconstruction / landmark anchor (×3)** — hull-masked pixel-L2 to A and B.
    Kept modest because L2 to two faces is minimized by their blurry average.
  - **Total-variation (×0.5)** — light smoothness regularizer.
  - **Adversarial (×1)** — a **PatchGAN** (LSGAN) critic classifies local patches
    as real face vs. morph, restoring high-frequency skin/eye/hair detail the
    pixel losses wash out. No paired ground-truth morph is needed: the "real"
    samples are the two aligned source faces.
- Generator and discriminator are trained alternately (`train_step`), with
  held-out validation each epoch and diagnostics for `|Δ|` (refiner activity),
  adversarial loss and discriminator loss.

**Output:** `cache/morph_model.weights.h5`.

### 4. Inference + analysis — `main_inference.py`

Generates the final morph for a pair of images and quantifies the identity blend.

1. Detect landmarks on both inputs and warp them onto the mean shape
   (`morph_pair`), exactly as in training.
2. Build the 75-channel input and run the trained refiner to get the morph.
3. **Poisson seam blend** (`poisson_seam_blend`) — `cv2.seamlessClone`
   re-integrates the morphed face over the single-identity background in the
   gradient domain, so the hull boundary becomes invisible.
4. **Face restoration** (`image_utils/face_restore.py`) — an optional GFPGAN pass
   re-synthesizes photographic high-frequency detail (and can upscale). This is a
   *soft dependency* (it pulls in PyTorch); if GFPGAN isn't installed the
   pipeline returns the un-restored image and continues end-to-end.
5. **Zone-embedding analysis** — the face is split into seven landmark zones
   (jaw, eyebrows, nose, eyes, mouth); each zone is cropped, embedded, and the
   cosine similarity of the morph to A and to B is reported per zone, plus a PCA
   projection and similarity heatmap. This verifies the morph genuinely sits
   between the two identities rather than collapsing onto one.

---

## Why hybrid (classical geometry + neural photometry)?

A pure cross-dissolve of two faces ghosts, because corresponding features sit at
different pixel locations. A pure generative model must learn both the large
geometric motion *and* the photometry, which is data-hungry and tends to blur or
to drift toward one identity. By solving geometry classically and exactly, the
network is left with a small, well-posed residual-correction problem — yielding a
morph that is simultaneously sharp, seamless, and balanced between both
identities.

## Repository layout

| Path | Role |
|------|------|
| `main_save_landmarks.py` | Stage 1 — detect & save 68 landmarks per image |
| `main_prepare_pairs.py`  | Stage 2 — pair identities, warp & cache |
| `main_train_model.py`    | Stage 3 — train the U-Net residual refiner |
| `main_inference.py`      | Stage 4 — generate a morph + zone analysis |
| `landmarks/`             | dlib landmark detection & heatmaps |
| `morphing/`              | warping (`warp.py`), pairing, dataset/caching |
| `model/`                 | U-Net refiner + PatchGAN, ArcFace/FaceNet embedders |
| `image_utils/`           | image loading, GFPGAN face restoration |
| `execution_parameters.yaml` | all run configuration |

## Configuration (`execution_parameters.yaml`)

Key knobs: `image_width/height` (model resolution, divisible by 8),
`number_of_landmarks` (68), `alpha` (blend ratio; 0.5 = equal contribution),
`morph_epochs` / `batch_size`, and the dataset parameters (`output_dir`,
`cache_dir`, `val_split`, `max_pairs`, `seed`).

## Quick start (full pipeline)

```bash
pip install -r requirements.txt

python main_save_landmarks.py          # 1. landmarks
python main_prepare_pairs.py           # 2. pair + warp + cache
python main_train_model.py             # 3. train refiner
python main_inference.py A.jpg B.jpg   # 4. morph two faces
```

## Inference only

If you just want to morph two faces with the already-trained model
(`cache/morph_model.weights.h5`), you don't need the training dependencies — use
the slimmer `requirements_inference.txt`, which also includes the GFPGAN
restoration pass.

```bash
# 1. create / activate the environment
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate

# 2. install the inference dependencies
pip install -r requirements_inference.txt

# 3. run the morph, passing the two input image paths
python main_inference.py path/to/A.jpg path/to/B.jpg
```

Notes:

- **Image paths** are positional: the first argument is identity A, the second is
  identity B. If you omit them, it falls back to the defaults in
  `main_inference.py` (`output_dataset/1000/image_1.jpg` and `image_8.jpg`).
- The blend ratio, model resolution, and `cache_dir` are read from
  `execution_parameters.yaml` (`alpha`, `image_width/height`, `cache_dir`).
- On the **first** run GFPGAN downloads its model weights automatically to
  `gfpgan/weights/`.
- GFPGAN is a *soft dependency*. If it isn't installed, `restore_face` returns the
  un-restored image and the pipeline still runs end-to-end.
