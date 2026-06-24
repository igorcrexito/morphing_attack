import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Concatenate)
import numpy as np

class DiffusionModel:

    def __init__(self):
        self.model = self.build_morph_generator()


    def build_morph_generator(self):

        inputs = Input(shape=(224, 224, 142))

        s1, p1 = self._encoder_block(inputs, 32)
        s2, p2 = self._encoder_block(p1, 64)
        s3, p3 = self._encoder_block(p2, 128)
        #s4, p4 = self._encoder_block(p3, 256)

        b = self._conv_block(p3, 256)

        #d1 = self._decoder_block(b, s4, 256)
        d2 = self._decoder_block(b, s3, 128)
        d3 = self._decoder_block(d2, s2, 64)
        d4 = self._decoder_block(d3, s1, 32)

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

    @tf.function
    def train_step(self, image_A, image_B, heatmap_A, heatmap_B, alpha, optimizer):
        X = tf.concat([image_A, image_B, heatmap_A, heatmap_B], axis=-1)

        with tf.GradientTape() as tape:
            morph = self.model(X, training=True)

            loss = self._balanced_loss(morph, image_A, image_B, heatmap_A, heatmap_B, alpha)

        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss


    def _balanced_loss(self, morph, image_A, image_B, heatmap_A, heatmap_B, alpha):
        maskA = tf.reduce_max(heatmap_A, axis=-1, keepdims=True)
        maskB = tf.reduce_max(heatmap_B, axis=-1, keepdims=True)

        lossA = tf.reduce_mean(maskA * tf.square(morph - image_A))
        lossB = tf.reduce_mean(maskB * tf.square(morph - image_B))

        return (alpha * lossA + (1 - alpha) * lossB)


    def fit(self, image_A, image_B, heatmap_A, heatmap_B, alpha, epochs, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((image_A, image_B, heatmap_A, heatmap_B))
        dataset = dataset.shuffle(len(image_A))

        dataset = dataset.batch(batch_size)

        optimizer = tf.keras.optimizers.Adam(1e-4)

        for epoch in range(epochs):
            epoch_loss = []

            for imgA, imgB, hmA, hmB in dataset:
                loss = self.train_step(imgA, imgB, hmA, hmB, alpha, optimizer)

                epoch_loss.append(loss.numpy())

            print(f"Epoch {epoch + 1}: "f"{np.mean(epoch_loss):.4f}")

    def predict(self, image_A, image_B, heatmap_A, heatmap_B):
        X = tf.concat([image_A, image_B, heatmap_A, heatmap_B], axis=-1)
        morph = self.model(X, training=False)

        return morph