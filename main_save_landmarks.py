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
import os

output_dir = "output_dataset"
os.makedirs(output_dir, exist_ok=True)


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

    landmarks_list = []
    image_list = []
    for index, image in enumerate(image_dataset):
        _, landmarks = landmark_descriptor.generate_landmarks(image=image,
                                                           channels=int(params['image_parameters']['image_channels']),
                                                           width=int(params['image_parameters']['image_width']),
                                                           height=int(params['image_parameters']['image_height']))

        ### resizing image to be in accordance with model input
        resized_image = Image.fromarray(image).resize((int(params['image_parameters']['image_width']),
                                                       int(params['image_parameters']['image_height'])))

        ### storing resized image
        image_filename = os.path.join(output_dir, f"image_{index + 1}.jpg")
        resized_image.save(image_filename, format="JPEG")

        ### storing landmarks
        landmark_filename = os.path.join(output_dir, f"image_{index + 1}.csv")
        np.savetxt(landmark_filename,
                   landmarks,
                   delimiter=",",
                   header="x,y",
                   comments="",
                   fmt="%.4f")


        ### showing landmarks
        if str(params['plotting_parameters']['plot_image']) == 'true':
            image = np.array(Image.open(f"{output_dir}/image_{index + 1}.jpg"))

            landmarks = np.loadtxt(f"{output_dir}/image_{index + 1}.csv", delimiter=",", skiprows=1)

            plt.figure(figsize=(6, 6))
            plt.imshow(image)

            plt.scatter(landmarks[:, 0], landmarks[:, 1], color="red", s=20)

            plt.xlim(0, 224)
            plt.ylim(224, 0)

            plt.axis("off")
            plt.show()