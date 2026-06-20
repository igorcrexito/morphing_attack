import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from PIL import Image

import tensorflow as tf
from tensorflow import keras, einsum
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
import tensorflow.keras.layers as nn
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

from einops import rearrange
from einops.layers.tensorflow import Rearrange
from functools import partial
from inspect import isfunction
import yaml
from image_utils.image_loader import ImageLoader
# Suppressing tf.hub warnings
tf.get_logger().setLevel("ERROR")

# configure the GPU
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)



if __name__ == '__main__':

    print("Reading the configuration yaml the stores the executation variables")
    with open("execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    ### instantiating and loading the image dataset
    image_loader = ImageLoader(width = int(params['image_parameters']['image_width']),
                                height = int(params['image_parameters']['image_height']),
                                channels = int(params['image_parameters']['image_channels']),
                                base_path = str(params['image_parameters']['dataset_path']))

    image_dataset = image_loader.load_images_from_path()

    __import__("IPython").embed()