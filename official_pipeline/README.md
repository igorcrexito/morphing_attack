# official_pipeline/ — the project's official morph generator

**This is now the recommended, official pipeline for this project**, replacing
the original (`model/diffusion_model.py`, `main_train_model.py`,
`main_generate_morphed_dataset.py`, still kept in place as the historical
baseline the `evaluation/` suite's ablations compare against). It's the
"balanced + pose-blend" configuration that came out of a full evaluation
investigation (see `evaluation/README.md` sections 5-6 for the complete story)
- three targeted loss/compositing fixes, plus a softened version of the final
pose-back warp, that together measurably improve identity balance without
touching the network architecture, the data pipeline, or the training loop.

## Why this replaced the original

The evaluation suite found that the original pipeline's morphs consistently
looked more like identity A than identity B. Two things turned out to matter,
found in this order:

1. **The measurement itself was misleading at first.** MMPMR looked very low
   (~6%) when scored against FaceNet, but that was mostly a property of
   FaceNet's threshold, not the generator - re-scoring against ArcFace (the
   backbone the literature actually uses) showed the *original* pipeline was
   already at ~77% MMPMR. See `evaluation/README.md` section 3/6 for the full
   explanation - always score against `arcface`, never `facenet`.
2. **The identity-A bias was real, and is what this folder fixes.** Even
   under ArcFace, every morph the original pipeline produces scores
   noticeably higher against A than B. Three causes were found in the code
   (see `model.py`'s module docstring for the full mechanism of each):
   - **FIX #1** - the identity loss was blind to the background (it zeroed
     it out before scoring), even though a real face-recognition system
     compares the full photo.
   - **FIX #2** - the loss only pulled toward the *average* of A's and B's
     embeddings, which doesn't guarantee equal similarity to both. A new term
     directly penalizes the gap.
   - **FIX #3** - the region outside the face mask (hair, ears, neck,
     background) was 100% identity A by construction; now a small, tunable
     fraction of B is blended in there too.
   - **FIX #4 (simple version)** - the final "pose the morph onto A's own
     head pose" warp, measured to roughly *triple* the identity gap on its
     own, is now only applied part-way (`POSE_BLEND`), landing on a geometry
     between the identity-neutral mean shape and A's exact posture.

## What was NOT fixed (on purpose)

A fully correct fix for #4 needs a *differentiable* version of the geometric
warp inside the training loop, so the network could actually learn against
its own final, posed geometry - a real project on its own, not a small patch.
`POSE_BLEND` is the practical, much simpler stand-in: it measurably helps
(see results below) without that larger undertaking. This is the next
highest-leverage improvement if you want to keep going.

## Files

| File | What it is |
|---|---|
| `model.py` | The official generator (`BalancedMorphGenerator`). Same architecture and training loop as the original `DiffusionModel`, only the loss and compositing formula changed (fixes #1-3). |
| `main_train.py` | Copy of `../main_train_model.py`, trains `BalancedMorphGenerator` on the same cached data (`cache/train.npz`/`val.npz`). Saves to `cache/morph_model_official.weights.h5`. |
| `main_generate.py` | Copy of `../main_generate_morphed_dataset.py`, generates morphs with the official model, including the softened pose-back warp (fix #4, `POSE_BLEND`). Writes to `morphed_dataset_official/`. |

## How to run

From the project root, with the same cached pairs the original pipeline uses
(`cache/train.npz` / `cache/val.npz` - run `main_prepare_pairs.py` first if
these don't exist yet, exactly as for the original pipeline):

```bash
source morphing_env/bin/activate

# 1. Train the official model (same epoch count etc. as execution_parameters.yaml)
python3 official_pipeline/main_train.py

# 2. Generate the SYN-MAD benchmark morphs with it
python3 official_pipeline/main_generate.py --pairs pairs_synmad.csv --out-name frll_synmad

# 3. IMPORTANT: score against ArcFace, not FaceNet - see evaluation/README.md
#    section 3/6. FaceNet's threshold reads misleadingly low MMPMR for ANY
#    pipeline; the numbers only mean something against the literature's
#    actual backbone. Make sure cache/arcface.npz covers these morphs first:
source arcface_env/bin/activate
python3 evaluation/export_arcface_embeddings.py   # re-run to pick up new morphs
deactivate
source morphing_env/bin/activate

# 4. Score it with the SAME evaluation suite used on the original pipeline
python3 evaluation/01_calibrate_thresholds.py   # picks up "arcface" automatically
python3 evaluation/02_vulnerability_eval.py \
    --morph-root morphed_dataset_official/frll_synmad --label official_pipeline
python3 evaluation/03_mad_baseline_eval.py \
    --morph-root morphed_dataset_official/frll_synmad --label official_pipeline

# 5. Rebuild the report - shows the official pipeline next to the original
#    baseline and the literature, side by side.
python3 evaluation/07_build_report.py
```

`BACKGROUND_BLEND` and `IDENTITY_BALANCE_WEIGHT` at the top of `model.py` are
the two knobs worth re-tuning if a future run under- or over-shoots: raise
`IDENTITY_BALANCE_WEIGHT` if the score gap is still large; lower
`BACKGROUND_BLEND` if ghosting returns in the hair/neck region. `POSE_BLEND`
in `main_generate.py` is the other knob - lower it for more identity balance
at the cost of the output looking less exactly like a normal photo of A.

## Measured results (FINAL - full 1000-pair SYN-MAD benchmark, both expression
splits complete, scored against ArcFace)

See `evaluation/artifacts/FINAL_REPORT.md` (section 7, "Head-to-head: this
work vs. the literature") for the always-current version of this table, and
`evaluation/README.md` section 5 for the same numbers in full detail.

| Pipeline | MMPMR @ FMR 1% (ArcFace) | MAD EER | Identity gap (sim→A − sim→B) |
|---|---|---|---|
| original (historical baseline, `model/`) | 75.8% | 7.33% | 0.427 |
| this folder, fixes #1-3 only (no pose-blend, ablation) | 79.1% | 7.82% | 0.417 |
| **this folder, official (+ `POSE_BLEND=0.5`)** | **86.5%** | 9.36% | **0.370** |

Fixes #1-3 alone move the identity gap and MMPMR only modestly - the pose-back
warp (fix #4) was always the dominant source of the bias, which is why
softening it (even with the simple `POSE_BLEND` compromise, not the full
differentiable fix) produces the largest jump in both numbers. The gap is
still far from closed (every morph, in every variant - 100% - still scores
higher against A than B) - lowering `POSE_BLEND` further, or eventually
implementing fix #4 properly, remain the clearest next steps.

One trade-off worth flagging honestly: the official pipeline's MAD EER also
rose (7.33% → 9.36%), meaning the statistical detector baseline
(`evaluation/common/mad_features.py`) finds these morphs somewhat *easier* to
catch, not harder. The detector's signal comes mostly from a high/low-frequency
spectral ratio (see evaluation chat history / README) - plausibly because
`POSE_BLEND` changes how much geometric re-warping the image goes through
before GFPGAN restoration. Better identity balance and lower detectability are
not automatically aligned; both axes should keep being tracked together, not
just MMPMR.
