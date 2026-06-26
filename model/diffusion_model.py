import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                                      MaxPooling2D, UpSampling2D, Concatenate,
                                      LeakyReLU)
import numpy as np
from tqdm import tqdm
from keras_facenet import FaceNet

# NOTE: this is a U-Net residual *refiner*, not a denoising-diffusion model.
# The heavy lifting of alignment is done classically (morphing/warp.py); the
# network only predicts a small correction `delta` on top of the aligned blend.
# Inputs are already warped to the mean shape, so the channel layout is:
#   warped_A (3) + warped_B (3) + mean heatmaps (68) + hull mask (1) = 75
INPUT_CHANNELS = 3 + 3 + 68 + 1


class DiffusionModel:

    # Relative weight of the adversarial (PatchGAN) term in the generator loss.
    # Kept modest so the reconstruction/identity anchors still dominate; the GAN
    # only sharpens texture the L2 terms would otherwise blur out.
    ADV_WEIGHT = 1.0

    def __init__(self):
        self.model = self.build_morph_generator()

        # PatchGAN discriminator + its own optimizer for the adversarial term.
        self.discriminator = self.build_discriminator()
        self.g_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

        # Face-identity embedder (FaceNet, trained on faces) for the biometric
        # identity-balance loss. ImageNet backbones do NOT encode face identity.
        self.facenet = FaceNet().model
        self.facenet.trainable = False

        # Perceptual (VGG) feature extractor for texture realism.
        vgg = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
        vgg.trainable = False
        self.perceptual_net = tf.keras.Model(
            inputs=vgg.input,
            outputs=[vgg.get_layer("block2_conv2").output,
                     vgg.get_layer("block3_conv3").output,
                     vgg.get_layer("block4_conv3").output])

    def build_morph_generator(self):
        # Resolution-agnostic: spatial dims are left as None so the same network
        # trains/infers at 224, 256 or 512 (anything divisible by 8 for the 3
        # pooling levels). Only the channel count is fixed.
        inputs = Input(shape=(None, None, INPUT_CHANNELS))

        s1, p1 = self._encoder_block(inputs, 32)
        s2, p2 = self._encoder_block(p1, 64)
        s3, p3 = self._encoder_block(p2, 128)

        b = self._conv_block(p3, 256)

        d2 = self._decoder_block(b, s3, 128)
        d3 = self._decoder_block(d2, s2, 64)
        d4 = self._decoder_block(d3, s1, 32)

        delta = Conv2D(3, 1, activation="tanh", name="residual_delta")(d4)
        return Model(inputs, delta)

    def build_discriminator(self):
        """PatchGAN: classifies overlapping local patches as real face / morph.

        A patch (rather than whole-image) critic rewards locally realistic skin,
        eye and hair texture - exactly the high-frequency detail the pixel-L2
        terms wash out - without needing a paired ground-truth morph."""
        def block(x, filters, bn=True):
            x = Conv2D(filters, 4, strides=2, padding="same")(x)
            if bn:
                x = BatchNormalization()(x)
            return LeakyReLU(0.2)(x)

        inputs = Input(shape=(None, None, 3))
        x = block(inputs, 64, bn=False)
        x = block(x, 128)
        x = block(x, 256)
        x = Conv2D(512, 4, strides=1, padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        logits = Conv2D(1, 4, strides=1, padding="same")(x)  # patch logits map
        return Model(inputs, logits)

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

    # ------------------------------------------------------------------ morph

    def _build_morph(self, warped_A, warped_B, delta, mask, alpha):
        # Faces are already aligned, so a straight cross-dissolve is clean *inside*
        # the face. Outside the hull (hair, ears, neck, background) the two
        # identities have no landmark correspondence, so blending them there only
        # produces ghosting/double-edges. Composite that region from a single
        # identity (A) instead, and refine only inside the feathered hull.
        blended = (1.0 - alpha) * warped_A + alpha * warped_B
        base = mask * blended + (1.0 - mask) * warped_A
        morph = base + delta * mask
        return tf.clip_by_value(morph, 0.0, 1.0)

    @tf.function
    def train_step(self, warped_A, warped_B, heatmaps, mask, alpha):
        X = tf.concat([warped_A, warped_B, heatmaps, mask], axis=-1)

        # ----- discriminator update -----------------------------------------
        # Real samples are the two aligned face photos; the fake is the morph.
        # Masking focuses the critic on the face region (consistent with the
        # other losses, which are all hull-masked).
        delta = self.model(X, training=True)
        morph = self._build_morph(warped_A, warped_B, delta, mask, alpha)
        with tf.GradientTape() as d_tape:
            d_real_A = self.discriminator(warped_A * mask, training=True)
            d_real_B = self.discriminator(warped_B * mask, training=True)
            d_fake = self.discriminator(morph * mask, training=True)
            # LSGAN targets: real -> 1, fake -> 0.
            d_loss = (tf.reduce_mean(tf.square(d_real_A - 1.0)) +
                      tf.reduce_mean(tf.square(d_real_B - 1.0)) +
                      2.0 * tf.reduce_mean(tf.square(d_fake))) * 0.25
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables))

        # ----- generator update ---------------------------------------------
        with tf.GradientTape() as g_tape:
            delta = self.model(X, training=True)
            morph = self._build_morph(warped_A, warped_B, delta, mask, alpha)
            recon = self._balanced_loss(morph, warped_A, warped_B, mask, alpha)
            # Adversarial: push the critic's score on the morph toward "real".
            d_fake = self.discriminator(morph * mask, training=False)
            adv = tf.reduce_mean(tf.square(d_fake - 1.0))
            g_loss = recon + self.ADV_WEIGHT * adv
        g_grads = g_tape.gradient(g_loss, self.model.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.model.trainable_variables))

        # Diagnostics: mean |delta| shows the refiner is actually correcting the
        # blend (not collapsing to 0); adv/d_loss show the GAN is in tension.
        delta_mag = tf.reduce_mean(tf.abs(delta))
        return g_loss, adv, d_loss, delta_mag

    # ------------------------------------------------------------------ losses

    def _balanced_loss(self, morph, warped_A, warped_B, mask, alpha):
        # Identity anchor in the face region: pull toward A and B by alpha weight.
        loss_landmark = (
            (1.0 - alpha) * tf.reduce_mean(mask * tf.square(morph - warped_A)) +
            alpha * tf.reduce_mean(mask * tf.square(morph - warped_B)))

        loss_perceptual = self._perceptual_loss(morph, warped_A, warped_B, mask, alpha)
        loss_tv = self._tv_loss(morph, mask)
        loss_identity = self._identity_loss(morph, warped_A, warped_B, mask, alpha)

        # Pixel-L2 to two different faces is minimized by their (blurry) average,
        # so keep it modest and lean on perceptual + adversarial for sharpness.
        return (3.0 * loss_landmark +
                6.0 * loss_perceptual +
                0.5 * loss_tv +
                25.0 * loss_identity)

    def _perceptual_loss(self, morph, warped_A, warped_B, mask, alpha):
        def preprocess(x):
            return tf.keras.applications.vgg16.preprocess_input(x * 255.0)

        fm = self.perceptual_net(preprocess(morph * mask))
        fa = self.perceptual_net(preprocess(warped_A * mask))
        fb = self.perceptual_net(preprocess(warped_B * mask))

        loss = 0.0
        for m, a, b in zip(fm, fa, fb):
            loss += (1.0 - alpha) * tf.reduce_mean(tf.square(m - a))
            loss += alpha * tf.reduce_mean(tf.square(m - b))
        return loss

    def _facenet_embed(self, images, mask):
        # FaceNet expects 160x160 RGB, per-image standardized ("prewhitening").
        x = tf.image.resize(images * mask, (160, 160)) * 255.0
        mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        std = tf.math.reduce_std(x, axis=[1, 2, 3], keepdims=True)
        std = tf.maximum(std, 1.0 / tf.sqrt(160.0 * 160.0 * 3.0))
        emb = self.facenet((x - mean) / std)
        return tf.math.l2_normalize(emb, axis=-1)

    def _identity_loss(self, morph, warped_A, warped_B, mask, alpha):
        emb_M = self._facenet_embed(morph, mask)
        emb_A = self._facenet_embed(warped_A, mask)
        emb_B = self._facenet_embed(warped_B, mask)

        target = tf.math.l2_normalize(
            (1.0 - alpha) * emb_A + alpha * emb_B, axis=-1)
        sim = tf.reduce_sum(emb_M * target, axis=-1)
        return tf.reduce_mean(1.0 - sim)

    def _tv_loss(self, morph, mask):
        dy = morph[:, 1:, :, :] - morph[:, :-1, :, :]
        dx = morph[:, :, 1:, :] - morph[:, :, :-1, :]
        return (tf.reduce_mean(mask[:, 1:, :, :] * tf.abs(dy)) +
                tf.reduce_mean(mask[:, :, 1:, :] * tf.abs(dx)))

    # ------------------------------------------------------------------ api

    @tf.function
    def eval_step(self, warped_A, warped_B, heatmaps, mask, alpha):
        X = tf.concat([warped_A, warped_B, heatmaps, mask], axis=-1)
        delta = self.model(X, training=False)
        morph = self._build_morph(warped_A, warped_B, delta, mask, alpha)
        return self._balanced_loss(morph, warped_A, warped_B, mask, alpha)

    def evaluate(self, dataset, alpha):
        alpha = tf.constant(alpha, dtype=tf.float32)
        losses = [self.eval_step(wA, wB, hm, mk, alpha).numpy()
                  for wA, wB, hm, mk in dataset]
        return float(np.mean(losses)) if losses else float("nan")

    def fit(self, train_dataset, alpha, epochs, val_dataset=None):
        alpha = tf.constant(alpha, dtype=tf.float32)

        history = {"train": [], "val": []}
        total_start = time.time()

        for epoch in range(epochs):
            epoch_loss, epoch_adv, epoch_d, epoch_delta = [], [], [], []
            t0 = time.time()

            pbar = tqdm(
                train_dataset,
                desc=f"Epoch {epoch + 1:>3}/{epochs}",
                unit="batch",
                leave=True,
                dynamic_ncols=True,
            )
            for wA, wB, hm, mk in pbar:
                loss, adv, d_loss, delta_mag = self.train_step(wA, wB, hm, mk, alpha)
                epoch_loss.append(loss.numpy())
                epoch_adv.append(adv.numpy())
                epoch_d.append(d_loss.numpy())
                epoch_delta.append(delta_mag.numpy())
                pbar.set_postfix({
                    "loss": f"{np.mean(epoch_loss):.4f}",
                    "adv": f"{np.mean(epoch_adv):.4f}",
                    "D": f"{np.mean(epoch_d):.4f}",
                    "|Δ|": f"{np.mean(epoch_delta):.4f}",
                })

            train_mean = float(np.mean(epoch_loss))
            adv_mean = float(np.mean(epoch_adv))
            d_mean = float(np.mean(epoch_d))
            delta_mean = float(np.mean(epoch_delta))
            history["train"].append(train_mean)
            elapsed = time.time() - t0

            diag = f"adv {adv_mean:.4f} | D {d_mean:.4f} | |Δ| {delta_mean:.4f}"
            if val_dataset is not None:
                val_mean = self.evaluate(val_dataset, alpha)
                history["val"].append(val_mean)
                print(f"  → train {train_mean:.4f} | val {val_mean:.4f} | "
                      f"{diag} | {elapsed:.1f}s")
            else:
                print(f"  → train {train_mean:.4f} | {diag} | {elapsed:.1f}s")

            # Show loss trend every 5 epochs (or at the end)
            if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
                best = min(history["train"])
                print(f"  [progress] best train so far: {best:.4f}  "
                      f"(total elapsed {(time.time() - total_start) / 60:.1f} min)")

        return history

    def save(self, weights_path):
        self.model.save_weights(weights_path)

    def load(self, weights_path):
        self.model.load_weights(weights_path)

    def predict(self, warped_A, warped_B, heatmaps, mask, alpha):
        alpha = tf.constant(alpha, dtype=tf.float32)
        X = tf.concat([warped_A, warped_B, heatmaps, mask], axis=-1)
        delta = self.model(X, training=False)
        morph = self._build_morph(warped_A, warped_B, delta, mask, alpha)
        return morph, delta
