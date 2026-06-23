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
import matplotlib.pyplot as plt


# Suppressing tf.hub warnings
tf.get_logger().setLevel("ERROR")

# configure the GPU
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)

output_dir = "output_dataset"
extension = '.jpg'


def load_training_pairs(output_dir, landmark_descriptor, width, height):

    files = sorted([f for f in os.listdir(output_dir) if f.endswith(".jpg")])

    imageA_list = []
    imageB_list = []

    heatmapA_list = []
    heatmapB_list = []

    for i in range(len(files)-1):
        imgA = np.array(Image.open(os.path.join(output_dir, files[i]))).astype(np.float32) / 255.0
        imgB = np.array(Image.open(os.path.join(output_dir, files[i+1]))).astype(np.float32) / 255.0

        lmA = np.loadtxt(os.path.join(output_dir, files[i].replace(".jpg",".csv")), delimiter=",", skiprows=1)
        lmB = np.loadtxt(os.path.join(output_dir, files[i+1].replace(".jpg",".csv")), delimiter=",", skiprows=1)

        hmA = landmark_descriptor.generate_heatmaps(lmA, width, height)
        hmB = landmark_descriptor.generate_heatmaps(lmB, width, height)

        imageA_list.append(imgA)
        imageB_list.append(imgB)

        heatmapA_list.append(hmA)
        heatmapB_list.append(hmB)

    return (np.array(imageA_list), np.array(imageB_list), np.array(heatmapA_list), np.array(heatmapB_list))



if __name__ == '__main__':
    print("Reading the configuration yaml the stores the executation variables")
    with open("execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    ### retrieving alpha coefficient. It represents the proportion of each face to be considered
    alpha = float(params['morphing_parameters']['alpha'])

    ### instantiating landmark descriptor
    landmark_descriptor = Landmarks(number_of_landmarks=68)

    ### reading dataset information
    imageA, imageB, heatmapA, heatmapB = load_training_pairs(output_dir=output_dir,
                                                             landmark_descriptor=landmark_descriptor,
                                                             width=int(params['image_parameters']['image_width']),
                                                             height=int(params['image_parameters']['image_height']))

    ### instantiating the diffusion model
    model = DiffusionModel()

    ### fitting the model
    model.fit(imageA, imageB, heatmapA, heatmapB, alpha=alpha, epochs=50, batch_size=1)

    ### predicting
    predicted_morph = model.predict(imageA, imageB, heatmapA, heatmapB)
    plt.imshow(np.uint8(predicted_morph[0]*255.0))
    plt.show()
    __import__("IPython").embed()