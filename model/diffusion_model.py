import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Concatenate)
import numpy as np
from tensorflow.keras.layers import Lambda

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

        #output = Conv2D(3, 1, activation="sigmoid")(d4)
        delta = Conv2D(3, 1, activation="tanh", name="residual_delta")(d4)

        return Model(inputs, delta)
        #return Model(inputs, output)


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
            #morph = self.model(X, training=True)
            delta = self.model(X, training=True)
            morph = self._build_morph(image_A, delta, heatmap_B)

            loss = self._balanced_loss(morph, image_A, image_B, heatmap_A, heatmap_B, alpha)

        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def _build_morph(self, image_A, delta, heatmap_B):
        maskB = tf.reduce_max(heatmap_B, axis=-1, keepdims=True)
        maskB = tf.nn.avg_pool2d(maskB, ksize=11, strides=1, padding="SAME")

        maskB = tf.clip_by_value(maskB, 0.0, 1.0)
        morph = image_A + (0.20 * delta * maskB)

        morph = tf.clip_by_value(morph, 0.0, 1.0)
        return morph


    def _balanced_loss(self, morph, image_A, image_B, heatmap_A, heatmap_B, alpha):

        maskB = tf.reduce_max(heatmap_B, axis=-1, keepdims=True)
        maskB = tf.nn.avg_pool2d(maskB, ksize=11, strides=1, padding="SAME")
        maskB = tf.clip_by_value(maskB, 0.0, 1.0)

        background_mask = 1.0 - maskB

        target_landmark = (alpha * image_A + (1.0 - alpha) * image_B)

        # global preservation of A
        loss_recon_A = tf.reduce_mean(tf.square(morph - image_A))

        # preserve outside landmarks
        loss_background = tf.reduce_mean(background_mask * tf.square(morph - image_A))

        # inject B only on landmarks
        loss_landmark = tf.reduce_mean(maskB * tf.square(morph - target_landmark))
        loss_shape = self._shape_loss(morph, image_A)
        loss_color = self._color_loss(morph,image_A)

        loss = (5.0 * loss_recon_A + 10.0 * loss_background + 3.0 * loss_landmark + 2.0 * loss_shape + 2.0 * loss_color)

        return loss


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
        delta = self.model(X, training=False)
        morph = self._build_morph(image_A, delta, heatmap_B)

        return morph

    def _shape_loss(self, morph, image_A):
        gx_m = morph[:, :, 1:, :] - morph[:, :, :-1, :]
        gy_m = morph[:, 1:, :, :] - morph[:, :-1, :, :]

        gx_a = image_A[:, :, 1:, :] - image_A[:, :, :-1, :]
        gy_a = image_A[:, 1:, :, :] - image_A[:, :-1, :, :]

        return (tf.reduce_mean(tf.square(gx_m - gx_a)) + tf.reduce_mean(tf.square(gy_m - gy_a)))


    def _color_loss(self, morph, image_A):
        mean_morph = tf.reduce_mean(morph, axis=[1, 2], keepdims=True)
        mean_A = tf.reduce_mean(image_A, axis=[1, 2], keepdims=True)

        return tf.reduce_mean(tf.square(mean_morph - mean_A))