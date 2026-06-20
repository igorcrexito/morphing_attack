import numpy as np
import matplotlib.pyplot as plt
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
from image_utils.image_loader import ImageLoader
from landmarks.landmark import Landmarks
from PIL import ImageDraw
from PIL import Image


# Suppressing tf.hub warnings
tf.get_logger().setLevel("ERROR")

# configure the GPU
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)



def plot_image(image, landmarks):
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)

    for x, y in landmarks:
        r = 2
        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill='red'
        )

    plt.imshow(image_draw)
    plt.axis('off')
    plt.show()



if __name__ == '__main__':

    print("Reading the configuration yaml the stores the executation variables")
    with open("execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    ### instantiating and loading the image dataset
    image_loader = ImageLoader(channels = int(params['image_parameters']['image_channels']),
                                base_path = str(params['image_parameters']['dataset_path']))

    image_dataset = image_loader.load_images_from_path()

    ### computing landmarks
    landmark_descriptor = Landmarks(number_of_landmarks=int(params['landmark_parameters']['number_of_landmarks']))

    for image in image_dataset:
        _, landmarks = landmark_descriptor.generate_landmarks(image=image,
                                                           channels=int(params['image_parameters']['image_channels']),
                                                           width=int(params['image_parameters']['image_width']),
                                                           height=int(params['image_parameters']['image_height']))

        if str(params['plotting_parameters']['plot_image']) == 'true':
            plot_image(Image.fromarray(image).resize((int(params['image_parameters']['image_width']),
                                                            int(params['image_parameters']['image_height']))), landmarks)