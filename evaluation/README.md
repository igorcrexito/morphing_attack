# HyMorph Evaluation Suite

This folder is a self-contained toolkit that answers one question fairly:
**how good is this project's morphing attack, measured the same way the
published literature measures its own attacks?**

It does three things, in order:

1. Runs your trained morphing pipeline against **real face-recognition
   systems** and reports the standard attack-success metric (**MMPMR**).
2. Trains and cross-validates a **detection baseline** on your morphs and
   reports the standard detector metrics (**EER, APCER/BPCER**).
3. Places your measured numbers next to numbers reported in the literature
   (`literature_numbers.yaml`), clearly labeled and never mixed together.

Everything here is a normal Python script you can read top to bottom - nothing
is hidden in a framework. If a number in the final report looks surprising,
open the script that produced it; the docstring at the top explains exactly
what it computed and why.

> **Read this before trusting any number:** Section "How to interpret the
> numbers" near the end of this file explains a real, already-observed
> limitation of the default run (FaceNet is not the same face-recognition
> model the literature benchmarks against) and exactly what to do about it.

---

## 0. What "fair comparison" means here, in one paragraph

Two papers can both report "95% MMPMR" and still not be comparable, if one
used easier identity pairs, a different face-recognition backbone, or a looser
decision threshold. So this suite fixes all three of those to match a specific,
named, published protocol wherever possible:

- **Same identity pairs**: `main_syn_mad_pairs.py` (already in the project
  root, one level up from this folder) reproduces the **SYN-MAD 2022**
  pairing recipe on the **FRLL** dataset - the exact same recipe the DCMorph
  (CVPRW 2026) paper uses. This suite's step 0 runs it for you if you haven't
  already.
- **Same face-recognition systems, calibrated the same way**: thresholds are
  computed from real genuine/impostor score distributions at a target False
  Match Rate (FMR), never eyeballed - see step 1.
- **Same reporting structure**: MMPMR at FMR 1% and 0.1%, a detector EER,
  robustness under compression, and an ablation - the same axes the
  literature review (`/home/igor/morphing_literature_summaries.pdf`) uses to
  compare every other method in this field.

---

## 1. Folder layout

```
evaluation/
├── README.md                     <- you are here
├── literature_numbers.yaml       <- published numbers, for CONTEXT ONLY (see its header)
├── run_full_evaluation.sh        <- runs every step below, in order
├── common/                       <- shared code, imported by every numbered script
│   ├── paths.py                  <- every file path used anywhere in this suite
│   ├── fr_systems.py             <- pluggable face-recognition backends
│   ├── metrics.py                <- MMPMR, EER, APCER/BPCER, threshold calibration
│   ├── degrade.py                <- JPEG recompression + print-scan simulation
│   └── mad_features.py           <- the statistical detector's feature extractor
├── 01_calibrate_thresholds.py    <- Step 1
├── 02_vulnerability_eval.py      <- Step 2
├── 03_mad_baseline_eval.py       <- Step 3
├── 04_robustness_eval.py         <- Step 4
├── 05_ablation_generate.py       <- Step 5
├── 06_ablation_eval.py           <- Step 6
├── 07_build_report.py            <- Step 7
└── artifacts/                    <- everything the scripts produce (git-ignored)
    └── FINAL_REPORT.md           <- the end result, in one file
```

---

## 2. What each step does, in plain language

### Step 0 - Build the benchmark pairs and morphs (already-existing project scripts)

Not a new script - it's `main_syn_mad_pairs.py` and
`main_generate_morphed_dataset.py --pairs pairs_synmad.csv`, one directory up.
`run_full_evaluation.sh` calls them for you if `pairs_synmad.csv` or
`morphed_dataset/frll_synmad/` don't exist yet. This produces:

- `pairs_synmad.csv`: 1000 identity pairs (250 most-similar pairs per
  gender x expression group, from the 102-subject FRLL dataset) - the SYN-MAD
  2022 recipe.
- `morphed_dataset/frll_synmad/{neutral,smiling}/morph_<a>_<b>.jpg`: the
  finished attack images, produced by your full trained pipeline (warp ->
  neural refinement -> Poisson blend -> pose back to identity A -> GFPGAN).

### Step 1 - `01_calibrate_thresholds.py`: fix the decision thresholds

A face-recognition system says "same person" when a similarity score clears
some threshold. That threshold must be picked **without looking at the
morphs**, exactly like a real deployment would set it: from real photos only.

This script uses the 102 FRLL subjects, each with a neutral capture
(`output_dataset/frll/`) and a smiling capture (`output_dataset/frll_probe/`)
of the *same* person:

- **Genuine pairs** = (neutral_i, smiling_i) for every subject -> 102 scores.
- **Impostor pairs** = (neutral_i, smiling_j) for every *different* pair of
  subjects -> 10,302 scores.

For every available face-recognition system (see "Face-recognition systems"
below), it finds the threshold where only 1% (then 0.1%) of impostor pairs
would still be wrongly accepted, and also reports that system's own baseline
EER as a sanity check. Output: `artifacts/thresholds.json`.

### Step 2 - `02_vulnerability_eval.py`: does the morph fool face recognition?

For every morph `morph_<a>_<b>.jpg`, this compares it against a **genuine
probe photo of subject a and of subject b** - specifically, a photo from the
*other* expression directory, so it was never used to build the morph. This is
the realistic scenario: the morph is the submitted enrollment photo; the probe
is what the traveller's own face looks like at the gate later.

For each morph it computes two similarity scores (morph-vs-probe-A,
morph-vs-probe-B) and reports **MMPMR** (Mated Morph Presentation Match Rate):
the fraction of morphs where the *worse* of the two scores still clears the
threshold from step 1 - i.e., the morph fools the system for **both**
contributors at once. Reported at both FMR 1% and FMR 0.1%.

This script is written as a reusable function
(`evaluate_vulnerability(morph_root, thresholds, label=...)`) so steps 4 and 6
call it on other morph directories (degraded / ablation variants) without
duplicating any logic.

### Step 3 - `03_mad_baseline_eval.py`: can a detector catch it?

Attack success alone doesn't tell you if the morph is realistic - it might
fool a face-recognition system's cosine similarity while still having an
obvious visual tell. This step trains a **Morphing Attack Detector (MAD)**
baseline and reports how well it can tell your morphs apart from real photos.

The detector (`common/mad_features.py`) is a compression-error statistical
baseline, inspired by a published, strong, simple detector described in the
literature review (Batskos, IWBF 2025): it re-compresses the image at several
JPEG quality levels and measures how the compression error and a rough
spatial-frequency signature respond, then classifies with an SVM. **This is an
independent, simplified re-implementation, not a reproduction of that paper -
see the caveat at the top of `mad_features.py` before quoting its number as
"the field's detector performance."**

Evaluation is 5-fold stratified cross-validation (every image is scored by a
model that never saw it during training), reporting:

- **EER** (Equal Error Rate): where the detector's false-alarm rate and
  miss rate are equal - one summary number, lower is better *for the
  detector* (i.e. higher EER here means your morphs are HARDER to detect).
- **BPCER @ APCER 5%/10%**: how many real photos get wrongly flagged, if the
  detector is tuned to miss at most 5% (or 10%) of attacks.

### Step 4 - `04_robustness_eval.py`: does it survive the real world?

A real ID-issuance pipeline compresses photos, and sometimes prints and
re-scans them. This script (`common/degrade.py`):

- **JPEG recompression** at quality 50.
- **A print-scan simulation**: downscale+upscale, a touch of blur, sensor
  noise, then JPEG recompression. This is an approximation of a physical
  print+scan round trip, not a real one - treat these numbers as directional,
  not as a substitute for an actual printed-and-rescanned test set.

Only the **submitted morph** is degraded (not the live probe capture, which
would be a fresh, undegraded photo at a real gate); for the MAD baseline, the
bona fide comparison photos are degraded identically, since a detector has to
inspect whatever photo was actually submitted. It re-runs steps 2 and 3 on the
degraded copies and reports the same metrics side by side with the pristine
ones, under the `robust_jpeg50` / `robust_printscan` labels.

### Step 5 - `05_ablation_generate.py`: isolate what each stage contributes

Generates three matched versions of the same subset of pairs, so any
difference between them is attributable to exactly one design choice:

| Variant | What's included |
|---|---|
| `geo_only` | Classical landmark warp + straight alpha cross-dissolve. No neural network, no Poisson blend, no restoration. What a pure 2014-style landmark morph looks like. |
| `network_only` | + the trained residual-refinement network (Stage 3 of the paper). Still no Poisson blend, no pose-back-to-A warp, no GFPGAN. |
| `full_pipeline` | The complete pipeline (network + Poisson seam blend + pose back to identity A + GFPGAN restoration) - reused by copying the matching files already produced by `main_generate_morphed_dataset.py`, not recomputed. |

`--limit N` controls how many pairs per gender/expression group to include
(default 30, i.e. up to 120 pairs total); this is deliberately smaller than
the full 1000-pair benchmark since it recomputes two of the three variants
from scratch.

### Step 6 - `06_ablation_eval.py`: score and compare the three variants

Runs steps 2 and 3 on each of the three ablation directories and prints/saves
a single side-by-side comparison table (`artifacts/ablation/comparison.json`).

### Step 7 - `07_build_report.py`: assemble everything into one file

Reads every JSON artifact produced by steps 1-6 (skipping any that weren't
run) plus `literature_numbers.yaml`, and writes
**`artifacts/FINAL_REPORT.md`**: your measured numbers first, the literature
context afterward, always visually separated.

---

## 3. Face-recognition systems: what's built in, and how to add a real one

| System | Type | Coverage |
|---|---|---|
| `facenet` | Runs locally, always available (`keras-facenet`, already a project dependency). | Can embed **any** image, including freshly generated morphs. **Caveat learned the hard way (see below): FaceNet's threshold at FMR 1% is 0.574, much stricter in raw terms than a margin-loss model's, so MMPMR measured against it reads far lower than against the literature's usual backbones - it is NOT the number to report as "attack success."** |
| `adaface` | Precomputed embeddings loaded from `cache/frll_adaface.npz` (already present in this repo). | Only the 204 original FRLL/FRLL-probe photos - **not** the morphs. Calibration only. |
| **`arcface`** | Precomputed embeddings in `cache/arcface.npz`, produced by `export_arcface_embeddings.py` (see below). | **Covers everything**: FRLL, FRLL-probe, and every morph directory listed in that script's `IMAGE_DIRS`. This is the system to report MMPMR against - it's the standard ArcFace-family backbone (insightface's `buffalo_l` = ArcFace-R50) the SYN-MAD 2022 / DCMorph / Greedy-DiM literature actually uses. |

### `export_arcface_embeddings.py` - how the `arcface` system was actually added

This is a worked, already-run example of the general "add a published FR
system" recipe below, kept as its own script because ArcFace is important
enough to be a first-class, always-available system rather than a one-off.

**It runs in a SEPARATE virtualenv (`arcface_env`), never in `morphing_env`.**
This matters: `insightface` depends on `onnxruntime`, which pulls in
`numpy>=2.0` and `protobuf>=5` - both incompatible with this project's
TensorFlow 2.15 (`numpy<2.0`, `protobuf<4.21`). Installing it into
`morphing_env` breaks TensorFlow immediately (this was tried, and fixed by
reinstalling `numpy<2.0`, `ml-dtypes~=0.3.1`, `protobuf<4.21` - if you ever see
`AttributeError: np.complex_ was removed` or similar from `morphing_env`,
that's this exact mistake). Keep the two venvs strictly separate, the same way
the project already keeps GFPGAN's PyTorch dependency isolated
(`image_utils/face_restore.py`'s module docstring).

Setup (one-time):
```bash
python3 -m venv arcface_env
source arcface_env/bin/activate
pip install onnxruntime insightface numpy pillow
```

Run (any time you want the `arcface` system to cover newly generated morphs -
safe to re-run, it recomputes everything under `IMAGE_DIRS` and overwrites
`cache/arcface.npz`):
```bash
source arcface_env/bin/activate
python3 evaluation/export_arcface_embeddings.py
deactivate
# back to morphing_env for everything else:
source morphing_env/bin/activate
python3 evaluation/01_calibrate_thresholds.py     # picks up "arcface" automatically
python3 evaluation/02_vulnerability_eval.py ...   # ditto
```

**What this changed, concretely:** the original pipeline's MMPMR went from
6.2% (FaceNet, misleadingly low) to **77.2%** (ArcFace, the literature-correct
backbone) at FMR 1%. The identity-imbalance problem documented throughout this
README is still real (every morph still scores higher against A than B, by
about the same margin under either backbone) - but ArcFace's much lower,
correctly-calibrated threshold (0.172 vs FaceNet's 0.574) means the B-side
score usually clears the bar anyway. **Always report MMPMR against `arcface`,
never `facenet` - `facenet` is useful for training-loss consistency checks and
as an always-available fallback, not for reporting attack success.**

### Adding another published FR system (ElasticFace / AdaFace-for-morphs / CurricularFace)

Same idea as `export_arcface_embeddings.py`, for a model that particular
script doesn't cover. Export embeddings in whatever environment has the model
and hand them back as a `.npz`:

```python
# in that OTHER environment:
import numpy as np
paths = [...]              # absolute paths to every image you need scored:
                            # output_dataset/frll/*.jpg, output_dataset/frll_probe/*.jpg,
                            # AND every morphed_dataset*/**/*.jpg you want covered
embeddings = model.embed(paths)   # your model's (N, D) embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
np.savez("elasticface.npz", paths=np.array(paths), embeddings=embeddings)
```

Drop the resulting file in `cache/elasticface.npz` (or `cache/curricularface.npz`,
...). `common/fr_systems.available_systems()` picks up every `cache/*.npz`
automatically and names the system after the filename - no code changes
needed. Re-run steps 1 and 2 and the new system appears in the tables
automatically.

---

## 4. How to run the evaluation

From the project root (`/home/igor/PycharmProjects/morphing_attack`), with the
project's virtualenv active:

```bash
source morphing_env/bin/activate

# everything, in order (builds pairs/morphs first if they don't exist yet):
./evaluation/run_full_evaluation.sh

# a faster pass, skipping the robustness step:
SKIP_ROBUSTNESS=1 ./evaluation/run_full_evaluation.sh

# a smaller ablation subset (faster):
ABLATION_LIMIT=10 ./evaluation/run_full_evaluation.sh
```

Or run any step on its own (each is independently useful and safe to re-run):

```bash
python3 evaluation/01_calibrate_thresholds.py
python3 evaluation/02_vulnerability_eval.py --morph-root morphed_dataset/frll_synmad --label full_pipeline
python3 evaluation/03_mad_baseline_eval.py  --morph-root morphed_dataset/frll_synmad --label full_pipeline
python3 evaluation/04_robustness_eval.py                      # both jpeg50 + printscan
python3 evaluation/04_robustness_eval.py --transform jpeg50   # just one
python3 evaluation/05_ablation_generate.py --limit 30
python3 evaluation/06_ablation_eval.py
python3 evaluation/07_build_report.py
```

Runtime on a laptop CPU, full 1000-pair benchmark: threshold calibration is
seconds; vulnerability scoring with FaceNet is a couple of minutes; the MAD
baseline (feature extraction + 5-fold SVM) is under a minute; robustness
roughly doubles steps 2-3's time per transform; ablation generation is the
slowest part per-pair (it runs the actual neural network), so its `--limit`
is deliberately conservative by default.

---

## 5. Results from the runs already performed while building this suite

Re-run `07_build_report.py` any time to regenerate the full table from the
latest artifacts (`artifacts/FINAL_REPORT.md`). The headline numbers below are
reported against **ArcFace** (the literature-correct backbone, via
`export_arcface_embeddings.py` - see Section 3); FaceNet numbers are kept
alongside for context, since the difference between them is itself an
important finding of this investigation (Section 6 explains why).

**Vulnerability, full 1000-pair `full_pipeline` (original model):**

| Split | n | MMPMR @ FMR 1% (ArcFace) | MMPMR @ FMR 0.1% (ArcFace) | MMPMR @ FMR 1% (FaceNet) |
|---|---|---|---|---|
| neutral | 500 | **77.2%** | 47.8% | 6.2% |
| smiling | 500 | **74.4%** | 42.0% | 6.2% |

**Vulnerability, full 1000-pair runs, `official_pipeline/` and its ablations
(FINAL - all three pipelines complete, both expression splits):**

| Pipeline | n | MMPMR @ FMR 1% (ArcFace) | MAD EER |
|---|---|---|---|
| original (`model/`, historical baseline) | 1000 | 75.8% (weighted avg, both splits) | 7.33% |
| fixes #1-3 only, no pose-blend (ablation) | 1000 | 79.1% | 7.82% |
| **official (fixes #1-3 + pose-blend, `official_pipeline/`)** | 1000 | **86.5%** | 9.36% |

See `artifacts/FINAL_REPORT.md` section 7 for these same numbers placed
directly alongside the literature (Greedy-DiM*, DCMorph, MIPGAN, ...).

**Identity score gap (ArcFace embedding space), same three pipelines,
averaged across both splits:**

| Pipeline | sim→A | sim→B | gap | frac. of morphs favoring A |
|---|---|---|---|---|
| original | 0.650 | 0.223 | 0.427 | 100% |
| fixes #1-3 only | 0.646 | 0.230 | 0.417 | 100% |
| official (+ pose-blend) | 0.628 | 0.258 | 0.370 | 100% |

**Ablation (32-pair subset, FaceNet only - small-sample, directional):**

| Variant | MMPMR @ FMR 1% | MAD EER |
|---|---|---|
| geo_only (classical warp + blend) | 12.50% | 2.54% |
| network_only (+ trained refiner) | 12.50% | 2.79% |
| full_pipeline (+ Poisson + pose-back + GFPGAN) | 6.25% | 9.34% |

---

## 6. How to interpret the numbers (read this before citing anything)

**Update: the ~6% MMPMR that motivated most of this README's earlier analysis
turned out to be almost entirely a FaceNet measurement artifact, now
confirmed by actually exporting ArcFace embeddings (Section 3) and
re-measuring.** With ArcFace, the SAME original model scores **~75-77%
MMPMR** - a real, competitive number. Keep both findings, because both are
true and both matter:

1. **FaceNet is not the system the literature benchmarks against, and the gap
   was large.** FaceNet's FMR-1% threshold on this data is 0.574; ArcFace's
   is 0.172 - a huge difference in raw terms, because ArcFace's margin-loss
   training clusters genuine pairs far more tightly than FaceNet's triplet
   loss does. A morph whose similarity to B is ~0.23-0.26 (which is what this
   pipeline actually produces) clears ArcFace's bar easily but fails
   FaceNet's by a wide margin. **This was the single biggest lever in the
   whole investigation - bigger than any generator-side fix.** If you take
   only one thing from this README: always report MMPMR against `arcface`,
   never `facenet`.
2. **The pose-back-to-identity-A bias is real and confirmed, but matters far
   less than it first appeared.** Re-measuring the identity gap in ArcFace's
   own embedding space (Section 5's gap table) shows the SAME bias as under
   FaceNet - every single morph still scores higher against A than B, by
   almost the same margin (~0.37-0.43). It just doesn't fail the (much lower)
   ArcFace threshold as often. In other words: the generator-side bias this
   README spent most of its effort diagnosing is a real, fixable property of
   the pipeline. `official_pipeline/` (which supersedes the original `model/`
   pipeline as the project's recommended generator - see its own README)
   closes part of this gap: MMPMR rose from 75.8% to 86.5% and the identity
   gap shrank from 0.427 to 0.370, but the bias is still far from eliminated
   (100% of morphs still favor A in every variant tested). It was never the
   reason MMPMR looked catastrophic, though - the backbone choice was.

**The practical lesson for next time:** when a measured number looks wildly
out of line with the literature, check the measurement instrument (here: the
FR backbone and its threshold) before concluding the thing being measured is
broken. Both checks were worth doing, but they should be done in that order.

**The MAD baseline (EER ≈ 7%) is a homemade, simplified detector** - a real
paper should additionally submit morphs to a public platform (BOEP or FATE
MORPH) for an independently-run, citable detectability number, and/or compare
against a detector pretrained on SMDD (fully synthetic, never saw this attack
type) as the more rigorous zero-shot test - see
`literature_numbers.yaml` for pointers to both.

**The literature numbers in Section 6 of `FINAL_REPORT.md` are for
orientation only** - different papers, different protocols, copied from a
secondary review document. Never average them with your own measured numbers
or present them as directly equivalent in a table without repeating the
caveats in `literature_numbers.yaml`.
