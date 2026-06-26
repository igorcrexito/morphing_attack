import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras, einsum
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from einops import rearrange
from einops.layers.tensorflow import Rearrange
import yaml
from landmarks.landmark import Landmarks
from PIL import Image
import os
from model.diffusion_model import DiffusionModel
from morphing.dataset import make_dataset
import matplotlib.pyplot as plt



# Suppressing tf.hub warnings
tf.get_logger().setLevel("ERROR")

# configure the GPU (TF2 API): cap usage at ~80% of the 8 GB card
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Let TF grow allocation as needed instead of pinning a fixed cap below
        # the card's real capacity; this gives the backward pass more headroom.
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # set_memory_growth must run before GPUs are initialized
        print(f"GPU memory config skipped: {e}")

if __name__ == '__main__':
    print("Reading the configuration yaml the stores the executation variables")
    with open("execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    ### retrieving alpha coefficient. It represents the proportion of each face to be considered
    alpha = float(params['morphing_parameters']['alpha'])
    width = int(params['image_parameters']['image_width'])
    height = int(params['image_parameters']['image_height'])
    train_morph_model = params['model_parameters'].get('train_morph_model', True)
    morph_epochs = int(params['model_parameters']['morph_epochs'])
    batch_size = int(params['model_parameters']['batch_size'])
    cache_dir = str(params['dataset_parameters']['cache_dir'])

    train_cache = os.path.join(cache_dir, "train.npz")
    val_cache = os.path.join(cache_dir, "val.npz")
    if not os.path.exists(train_cache):
        raise FileNotFoundError(
            f"{train_cache} not found. Run `python main_prepare_pairs.py` first "
            "to build the warped-pair cache.")

    ### loading pre-warped, cached datasets (heatmaps generated on the fly)
    train_ds, n_train = make_dataset(train_cache, width, height, batch_size, shuffle=True)
    val_ds, n_val = (make_dataset(val_cache, width, height, batch_size, shuffle=False)
                     if os.path.exists(val_cache) else (None, 0))
    print(f"Train pairs: {n_train} | Val pairs: {n_val}")

    if train_morph_model:
        ### instantiating the morph refinement model
        model = DiffusionModel()

        ### fitting the model (with held-out validation)
        history = model.fit(train_ds, alpha=alpha, epochs=morph_epochs, val_dataset=val_ds)

        print("\n=== Training complete ===")
        print(f"  Final train loss : {history['train'][-1]:.4f}")
        print(f"  Best  train loss : {min(history['train']):.4f}  (epoch {history['train'].index(min(history['train'])) + 1})")
        if history['val']:
            print(f"  Final val   loss : {history['val'][-1]:.4f}")
            print(f"  Best  val   loss : {min(history['val']):.4f}  (epoch {history['val'].index(min(history['val'])) + 1})")

        ### saving morph model
        weights_path = os.path.join(cache_dir, "morph_model.weights.h5")
        model.save(weights_path)
        print(f"Saved model weights to {weights_path}")
    else:
        print("Skipping morph model training (train_morph_model: false)")