# model/arcface_backbone.py
import tensorflow as tf

class ArcFaceBackbone:

    def __init__(self, weights_path):

        self.model = tf.keras.models.load_model(weights_path, compile=False)
        self.model.trainable = False

    def __call__(self, images):
        images = tf.image.resize(images, (112,112))
        images = (images * 255.0)

        embeddings = self.model(images, training=False)
        embeddings = tf.math.l2_normalize(embeddings, axis=-1)

        return embeddings