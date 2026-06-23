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


# Suppressing tf.hub warnings
tf.get_logger().setLevel("ERROR")

# configure the GPU
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)

output_dir = "output_dataset"
extension = '.jpg'

if __name__ == '__main__':
    print("Reading the configuration yaml the stores the executation variables")
    with open("execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    ### retrieving alpha coefficient. It represents the proportion of each face to be considered
    alpha = float(params['morphing_parameters']['alpha'])

    ### List all store images
    files = [f for f in os.listdir(output_dir) if f.endswith(extension)]

    ### randomizing two elements of the input list
    index_A, index_B = np.random.choice(files, size=2, replace=False)

    ### loading images A and B and landmarks
    image_A = np.array(Image.open(f"{output_dir}/{index_A}"))
    image_B = np.array(Image.open(f"{output_dir}/{index_B}"))

    ### generating landmarks heatmaps
    landmark_descriptor = Landmarks(number_of_landmarks=int(params['landmark_parameters']['number_of_landmarks']))
    landmarks_A = landmark_descriptor.generate_heatmaps(np.loadtxt(f"{output_dir}/{index_A.split('.')[0]}.csv", delimiter=",", skiprows=1),
                                                        int(params['image_parameters']['image_width']),
                                                        int(params['image_parameters']['image_height']))

    landmarks_B = landmark_descriptor.generate_heatmaps(np.loadtxt(f"{output_dir}/{index_B.split('.')[0]}.csv", delimiter=",", skiprows=1),
                                                        int(params['image_parameters']['image_width']),
                                                        int(params['image_parameters']['image_height']))

    ### generating an input tensor
    image_A = image_A.astype(np.float32) / 255.0
    image_B = image_B.astype(np.float32) / 255.0
    input_tensor = tf.concat([image_A, image_B, landmarks_A, landmarks_B], axis=-1)

    ### instantiating a morphing model
    diffusion_model = DiffusionModel()

    ### creating a target landmark and collapsing into a single channel
    maskA = tf.reduce_max(landmarks_A, axis=-1, keepdims=True)
    maskB = tf.reduce_max(landmarks_B, axis=-1, keepdims=True)

    ### generating a weak supervised signal for the morphing image
    predicted_morph = diffusion_model(input_tensor, training=True)

    ### defining a customized mse loss
    lossA = tf.reduce_mean(maskA * tf.square(morph - image_A))
    lossB = tf.reduce_mean(maskB * tf.square(morph - image_B))
    balanced_loss = (alpha * lossA + (1 - alpha) * lossB)

    __import__("IPython").embed()