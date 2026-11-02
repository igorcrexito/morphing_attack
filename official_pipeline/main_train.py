"""official_pipeline/main_train.py

*** Trains the official / recommended morph generator for this project. ***

Near-identical copy of ../main_train_model.py. Trains the SAME architecture on
the SAME cached data (cache/train.npz / cache/val.npz, built by the original
main_prepare_pairs.py - nothing about data preparation changes), the only
difference is which model class is trained:

    original (historical baseline) : model.diffusion_model.DiffusionModel
    here (official)                : official_pipeline.model.BalancedMorphGenerator
               (the 3 identity-balance fixes - see model.py's module docstring
               for exactly what changed and why)

Everything else - reading execution_parameters.yaml, loading the cached
dataset, the fit() call, the printed summary - is the same as the original.
A few imports that were unused in the original file (tensorflow_addons,
tensorflow_datasets, einops, keras.Sequential/Layer) were dropped here to keep
this version simple; they were not doing anything in the original either.

Run this from the PROJECT ROOT, exactly like the original:
    python official_pipeline/main_train.py
"""
import os
import sys

import yaml
import tensorflow as tf

# The project's other modules (morphing/, model/) live one level up from this
# folder, so make sure that directory is importable regardless of where this
# script is launched from - same convention used throughout evaluation/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from morphing.dataset import make_dataset
from official_pipeline.model import BalancedMorphGenerator

# Suppressing tf.hub warnings - same as the original.
tf.get_logger().setLevel("ERROR")

# Configure the GPU (TF2 API) - same as the original: let TF grow allocation
# as needed instead of pinning a fixed cap.
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # set_memory_growth must run before GPUs are initialized.
        print(f"GPU memory config skipped: {e}")


if __name__ == "__main__":
    print("Reading the configuration yaml that stores the execution variables")
    with open("execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    # Same parameters as the original training script - alpha (identity blend
    # ratio), image size, epoch count, batch size, cache directory.
    alpha = float(params["morphing_parameters"]["alpha"])
    width = int(params["image_parameters"]["image_width"])
    height = int(params["image_parameters"]["image_height"])
    morph_epochs = int(params["model_parameters"]["morph_epochs"])
    batch_size = int(params["model_parameters"]["batch_size"])
    cache_dir = str(params["dataset_parameters"]["cache_dir"])

    train_cache = os.path.join(cache_dir, "train.npz")
    val_cache = os.path.join(cache_dir, "val.npz")
    if not os.path.exists(train_cache):
        raise FileNotFoundError(
            f"{train_cache} not found. Run `python main_prepare_pairs.py` "
            "(from the project root) first to build the warped-pair cache - "
            "this balanced pipeline reuses the SAME cache format, nothing "
            "about data preparation changed.")

    # Loading the pre-warped, cached datasets (heatmaps generated on the fly)
    # - identical to the original.
    train_ds, n_train = make_dataset(train_cache, width, height, batch_size, shuffle=True)
    val_ds, n_val = (make_dataset(val_cache, width, height, batch_size, shuffle=False)
                     if os.path.exists(val_cache) else (None, 0))
    print(f"Train pairs: {n_train} | Val pairs: {n_val}")

    # === the only real change: train BalancedMorphGenerator instead of the
    # original DiffusionModel. Its .fit()/.save() API is identical. ===
    model = BalancedMorphGenerator()
    history = model.fit(train_ds, alpha=alpha, epochs=morph_epochs, val_dataset=val_ds)

    print("\n=== Training complete (balanced pipeline) ===")
    print(f"  Final train loss : {history['train'][-1]:.4f}")
    print(f"  Best  train loss : {min(history['train']):.4f}  "
          f"(epoch {history['train'].index(min(history['train'])) + 1})")
    if history["val"]:
        print(f"  Final val   loss : {history['val'][-1]:.4f}")
        print(f"  Best  val   loss : {min(history['val']):.4f}  "
              f"(epoch {history['val'].index(min(history['val'])) + 1})")

    # Saved under a DIFFERENT filename than the original
    # (morph_model_official.weights.h5, not morph_model.weights.h5) so training
    # this pipeline never overwrites your original model's weights - both can
    # exist side by side and be compared with evaluation/.
    weights_path = os.path.join(cache_dir, "morph_model_official.weights.h5")
    model.save(weights_path)
    print(f"Saved balanced model weights to {weights_path}")
