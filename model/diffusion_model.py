import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Concatenate)


class DiffusionModel:

    def __init__(self):
        self.model = self.build_morph_generator()


    def build_morph_generator(self):

        inputs = Input(shape=(224, 224, 142))

        s1, p1 = self._encoder_block(inputs, 64)
        s2, p2 = self._encoder_block(p1, 128)
        s3, p3 = self._encoder_block(p2, 256)
        s4, p4 = self._encoder_block(p3, 512)

        b = self._conv_block(p4, 1024)

        d1 = self._decoder_block(b, s4, 512)
        d2 = self._decoder_block(d1, s3, 256)
        d3 = self._decoder_block(d2, s2, 128)
        d4 = self._decoder_block(d3, s1, 64)

        output = Conv2D(3, 1, activation="sigmoid")(d4)

        return Model(inputs, output)


    def _conv_block(self, x, filters):
        x = Conv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("swish")(x)
        x = Conv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("swish")(x)
        return x

    def _encoder_block(self, x, filters):
        f = self._conv_block(x, filters)
        p = MaxPooling2D()(f)
        return f, p

    def _decoder_block(self, x, skip, filters):
        x = UpSampling2D()(x)
        x = Concatenate()([x, skip])
        x = self._conv_block(x, filters)
        return x

    def _landmark_region_loss(self, morph, target_face, mask):
        return tf.reduce_mean(mask * tf.square(morph - target_face))

    def fit(self, image_A, image_B, heatmap_A, heatmap_B):
        X = tf.concat([image_A, image_B, heatmap_A, heatmap_B], axis=-1)
        morph = self.model(X, training=True)