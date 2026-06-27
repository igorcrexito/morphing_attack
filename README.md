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

#### Is this a GAN? How the generator and discriminator work

**Yes — but a *conditional, partial* GAN, not a from-scratch image generator.**
The adversarial loss is only one of five terms in the training objective, and the
generator never synthesizes a whole face; it predicts a small correction on top of
an already-aligned classical blend. Concretely:

- **Generator** = the U-Net residual refiner (`build_morph_generator`). It takes
  the 75-channel conditioning input (`warped_A` + `warped_B` + 68 landmark
  heatmaps + hull mask) and outputs a `tanh`-bounded **residual delta** the same
  size as the image. That delta is added to the classical blend *only inside the
  feathered hull* (`_build_morph`):

  ```
  blended = (1 − α)·warped_A + α·warped_B     # geometry already solved → no ghosting
  base    = mask·blended + (1 − mask)·warped_A # outside the hull = identity A
  morph   = clip( base + delta·mask )          # generator refines inside the hull
  ```

  So this is a **conditional GAN** (cGAN): the generator is conditioned on the two
  source faces and the landmark structure, not sampling from noise. Its job is
  texture/photometry, not geometry.

- **Discriminator** = a **PatchGAN** critic (`build_discriminator`). Instead of
  emitting one real/fake score for the whole image, it is fully convolutional and
  outputs a *grid* of logits, each judging a local ~receptive-field patch as
  "real face" vs. "morph". This is what rewards locally realistic skin, eye and
  hair texture — the high-frequency detail the pixel-L2 losses blur away. It is
  hull-masked (`discriminator(x * mask)`) so it only critiques the face region.

- **Adversarial game (`train_step`), trained with the LSGAN least-squares loss:**
  1. *Discriminator step* — the **two aligned source photos** `warped_A`,
     `warped_B` are the "real" examples (target → 1); the generator's `morph` is
     the "fake" (target → 0). Crucially **no paired ground-truth morph is ever
     needed** — the real distribution is simply "a genuine aligned face".
  2. *Generator step* — the generator is pushed to make the critic score its
     `morph` as real (`(d_fake − 1)²`), *in addition to* the reconstruction,
     perceptual, identity-balance and TV terms. The adversarial weight
     (`ADV_WEIGHT = 1`) is kept modest so the identity-balance loss (×25) still
     dominates — the GAN sharpens texture but does not get to drift the morph
     toward one identity.

  The two networks have separate Adam optimizers and are updated alternately each
  batch. Training diagnostics print `|Δ|` (mean absolute delta → shows the refiner
  is actually correcting, not collapsing to zero), `adv` and `D` (the two sides of
  the game staying in tension).

In short: **classical warping does the geometry, a small conditional PatchGAN does
the photometry**, and the adversarial term is a texture-sharpener layered on top of
strong reconstruction/identity supervision — not the sole training signal.

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

### 5. Batch dataset generation — `main_generate_morphed_dataset.py`

Generates a whole morphed dataset (rather than a single pair) from a landmarked
dataset under `output_dataset/<dataset_name>`. For every face it picks a partner by
**FaceNet embedding distance**, runs the full pipeline (align → refiner → Poisson
seam blend → pose to A → GFPGAN), and writes only the final image.

- **Partner-selection strategy** (`morph_strategy` in
  `execution_parameters.yaml`) — among all shape-compatible candidates
  (`shape_distance ≤ SHAPE_THRESHOLD`), rank by FaceNet cosine similarity and pick:
  - `most_similar` — the closest identity (most realistic, hardest-to-detect morphs),
  - `most_distant` — the farthest identity (hardest morphs to make convincing),
  - `median` — the 50th-percentile candidate.
- **Output layout** — morphs are written to
  `morphed_dataset/<dataset_name>/<strategy>/morph_<a>_<b>.jpg`, so the three
  strategies land in their own sub-directories side by side and can be compared.

```bash
python main_generate_morphed_dataset.py    # uses dataset_name + morph_strategy from the yaml
```

### 6. Morphing your own images — `user_test_images.py`

Runs the same inference + analysis as `main_inference.py`, but **starting from raw
images** (it performs landmark detection itself, so no pre-computed `.csv` is
needed). Drop at least two face images into a `user_test/` folder:

```bash
mkdir -p user_test
# copy two face photos into user_test/, then:
python user_test_images.py                       # morphs the first two images found
python user_test_images.py user_test/alice.jpg user_test/bob.jpg   # or pick explicitly
```

It prints the per-zone cosine-similarity table and shows the same three plots
(morphing result, zone crops, zone-embedding PCA + similarity heatmap).

> **Use frontal face photos.** Landmark detection relies on dlib's *frontal*-face
> detector, and the warping assumes a reliable front-facing 68-point
> correspondence. Near-frontal, unobstructed, well-lit portraits (one face per
> image) give the best morphs; strongly rotated, tilted, profile or occluded faces
> may fail detection or produce poor results.

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
| `main_generate_morphed_dataset.py` | Stage 5 — batch-morph a whole dataset by embedding-distance strategy |
| `user_test_images.py`    | Stage 6 — morph your own raw images from `user_test/` |
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

## Getting the code (and the pre-trained weights)

The repository is **self-contained**: the trained morph refiner
(`cache/morph_model.weights.h5`, ≈8 MB) and the dlib landmark predictors
(`landmarks/shape_predictor_68_face_landmarks.dat`,
`landmarks/shape_predictor_5_face_landmarks.dat`) are committed directly to git, so
a plain clone gives you everything needed to run inference — **no separate model
download and no Git LFS required.**

```bash
# 1. clone
git clone git@github.com:igorcrexito/morphing_attack.git
#   (or over HTTPS:)
# git clone https://github.com/igorcrexito/morphing_attack.git
cd morphing_attack

# 2. confirm the trained weights came down with the checkout
ls -lh cache/morph_model.weights.h5            # ≈8 MB, ships in the repo

# 3. create an environment and install the inference deps
python -m venv .venv
source .venv/bin/activate                      # Windows: .venv\Scripts\activate
pip install -r requirements_inference.txt

# 4. morph two of your own faces straight away — no training needed
mkdir -p user_test                             # add two face photos here
python user_test_images.py
#   or morph an explicit pair:
# python main_inference.py path/to/A.jpg path/to/B.jpg
```

Because the weights are versioned alongside the code, you can reproduce results
immediately after cloning; only re-run stages 1–3 if you want to **retrain** on a
different dataset. GFPGAN's restoration weights are the one exception — they are
downloaded automatically to `gfpgan/weights/` on the first inference run.

## Reducing blend errors (ghosting / visible seams)

Occasional morphs show a doubled feature (e.g. two eyebrows) or a faint
rectangular hull seam. The cause is almost always a **non-frontal or pose-mismatched
pair**: when A and B don't share a near-frontal pose the 68-point correspondence is
unreliable, so after warping the features land a few pixels apart and ghost. The
`quality_parameters` block in `execution_parameters.yaml` controls the mitigations —
**none of these require retraining**, as they only affect pair selection, frontality
filtering and the classical compositing around the network:

| Key | Effect |
|-----|--------|
| `shape_threshold` | Max Procrustes shape divergence for a blendable pair (lower = stricter pairing; `0.09` is a good start, vs. the old loose `0.12`). |
| `max_yaw` | Rejects non-frontal faces by nose/eye horizontal asymmetry (`0` = perfectly frontal). `null` disables the gate. Applied in `Landmarks` and as a pre-pairing filter in `main_generate_morphed_dataset.py`. |
| `hull_feather` | Face-mask feather radius (px). Larger softens the hull boundary so residual ghosting fades instead of stepping. |
| `poisson_erode` | Erosion iterations before `cv2.seamlessClone`; more pulls the clone boundary inside the face, hiding ghosting that sits against the hull edge. |
| `align_faces` | Similarity-align & crop each face at landmark time (eyes → a canonical template) so faces are centered/scaled consistently and in-plane roll is removed. Improves facial detail and FaceNet identity/pairing. **See the retraining note below.** |

**About `align_faces`.** When on, `Landmarks.generate_landmarks` rotates/scales/
translates each face so its eye centers hit fixed template coordinates, then crops to
the model resolution — the standard FFHQ/ArcFace-style alignment. It returns that
aligned colour image, and every entry-point uses it (so the saved `.jpg` and its
`.csv` always match). This puts more pixels on the face and gives FaceNet cleaner,
consistently-scaled crops, which sharpens morphs and improves partner selection. It
removes in-plane *roll* but **cannot** fix out-of-plane head rotation (yaw/pitch) —
that's still the job of `max_yaw`. Because it changes the facial scale/position the
network sees, **enabling it for the dataset flow means re-running
`main_save_landmarks.py` and retraining (`main_train_model.py`)**; inference on the
existing weights still runs but with a mild train/inference mismatch until you
retrain. It's `false` by default for exactly this reason.

The `median` / `most_distant` strategies are more exposed to ghosting (they pick
less-similar — and often pose-mismatched — partners), so pair them with a tighter
`shape_threshold`.

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
  `main_inference.py` (`output_dataset/<dataset_name>/image_1.jpg` and
  `image_43.jpg`).
- The blend ratio, model resolution, and `cache_dir` are read from
  `execution_parameters.yaml` (`alpha`, `image_width/height`, `cache_dir`).
- On the **first** run GFPGAN downloads its model weights automatically to
  `gfpgan/weights/`.
- GFPGAN is a *soft dependency*. If it isn't installed, `restore_face` returns the
  un-restored image and the pipeline still runs end-to-end.
