"""official_pipeline/model.py

*** THIS IS NOW THE OFFICIAL / RECOMMENDED MORPH GENERATOR FOR THE PROJECT. ***
It supersedes ../model/diffusion_model.py, which is kept in place only as the
historical baseline the evaluation suite's ablations compare against (see
evaluation/README.md section 5-6 and this folder's README for the full story
of why and how it changed).

This file started as a near-copy of ../model/diffusion_model.py - almost every
line is still identical, by design, so the change is easy to review. Every
place that actually changed is marked with a comment block starting
"# === FIX #N ===" so you can find the 3 fixes by searching for that string.

WHY these 3 fixes exist (short version - see official_pipeline/README.md for
the full investigation): the evaluation suite in ../evaluation/ measured a
low attack success rate under an initial (misleading, FaceNet-calibrated)
face-recognition backbone, then, after switching to ArcFace (the literature's
actual backbone - see evaluation/README.md section 3), confirmed the
underlying pipeline was already competitive (~77% MMPMR) but still had a real,
measurable bias toward identity A. Two causes were found in the original code,
plus one contributing design choice:

  FIX #1 - The original identity loss embeds `morph * mask` (the background is
           ZEROED before being shown to FaceNet). But the image a real face
           recognition system actually compares against is the FULL photo,
           background included - and that background is, by construction,
           100% identity A (see FIX #3). So the original loss never "sees"
           the thing that is actually causing the imbalance. Fix: embed the
           full image, unmasked, so the loss reflects reality.

  FIX #2 - The original loss only pulls the morph's embedding toward the
           midpoint embedding (1-alpha)*embA + alpha*embB. That does not
           guarantee sim(morph, A) == sim(morph, B) - cosine similarity isn't
           linear along that average direction. Fix: add a second, explicit
           term that directly penalizes the GAP between the two similarities.

  FIX #3 - The original composition takes the region OUTSIDE the face mask
           (hair, ears, neck, background) 100% from identity A, on purpose,
           to avoid ghosting there (there's no reliable landmark
           correspondence outside the face). That is still mostly true here -
           we keep the same design - but we blend in a SMALL, controlled
           amount of B outside the mask too (see BACKGROUND_BLEND below),
           so the image isn't structurally 100% A before the network even
           runs. This is a deliberate trade-off: too much blending outside
           the mask reintroduces the ghosting the original design avoided, so
           the constant below is kept small and is easy to tune.

FIX #4 from the investigation (the fact that Stage 4's "pose the morph back
onto identity A's own landmarks" step happens AFTER training, so the network
never trains against its own final geometry) is DELIBERATELY NOT implemented
here, to keep this pipeline simple: doing it properly needs a differentiable
version of the geometric warp inside the training loop, which is a much
bigger change than the other three. FIX #1 and FIX #3 both reduce how much
that pose-back step can hurt (there is less asymmetric material for it to
work with), which is why they are still worth doing on their own. See
official_pipeline/README.md, section "What was NOT fixed (on purpose)".
"""
import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                                      MaxPooling2D, UpSampling2D, Concatenate,
                                      LeakyReLU)
import numpy as np
from tqdm import tqdm
from keras_facenet import FaceNet

# Same as the original: warped_A (3) + warped_B (3) + landmark heatmaps (68) +
# hull mask (1) = 75 input channels. Unchanged.
INPUT_CHANNELS = 3 + 3 + 68 + 1

# === FIX #3 (tunable constant) =============================================
# Fraction of identity B blended into the region OUTSIDE the face mask (hair,
# ears, neck, background), which the original code took 100% from identity A.
# 0.0 reproduces the original (fully-A) behaviour exactly. Keep this small:
# the region outside the mask has no reliable landmark correspondence between
# A and B, so a large value reintroduces the ghosting artifacts the original
# design was built to avoid. 0.25 was chosen as a conservative starting point
# - re-tune by regenerating morphs at a few values and checking both the
# identity-balance metrics (evaluation/02_vulnerability_eval.py) and visual
# ghosting around the hairline.
BACKGROUND_BLEND = 0.25

# === FIX #2 (tunable constant) ==============================================
# Weight of the new identity-BALANCE loss term (see BalancedMorphGenerator.
# _identity_balance_loss below), relative to the other loss terms in
# _balanced_loss. Kept modest relative to the 25.0 identity-anchor weight so
# it nudges the network toward symmetry without fighting the anchor term.
IDENTITY_BALANCE_WEIGHT = 10.0


class BalancedMorphGenerator:
    """Same residual U-Net + PatchGAN generator as the original DiffusionModel,
    with the 3 identity-balance fixes described at the top of this file.
    Everything else (architecture, training loop shape, save/load/predict API)
    is intentionally left unchanged so it's a drop-in replacement."""

    # Unchanged from the original.
    ADV_WEIGHT = 1.0

    def __init__(self):
        self.model = self.build_morph_generator()

        # PatchGAN discriminator + its own optimizer for the adversarial term.
        # Unchanged from the original.
        self.discriminator = self.build_discriminator()
        self.g_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

        # Face-identity embedder (FaceNet) for the biometric identity losses.
        # Unchanged from the original.
        self.facenet = FaceNet().model
        self.facenet.trainable = False

        # Perceptual (VGG) feature extractor for texture realism. Unchanged.
        vgg = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
        vgg.trainable = False
        self.perceptual_net = tf.keras.Model(
            inputs=vgg.input,
            outputs=[vgg.get_layer("block2_conv2").output,
                     vgg.get_layer("block3_conv3").output,
                     vgg.get_layer("block4_conv3").output])

    # ------------------------------------------------------------- network
    # Everything in this section (build_morph_generator, build_discriminator,
    # _conv_block, _encoder_block, _decoder_block) is IDENTICAL to the
    # original model/diffusion_model.py - no fix touches the network
    # architecture, only the losses and the compositing formula.

    def build_morph_generator(self):
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
        Unchanged from the original."""
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
        logits = Conv2D(1, 4, strides=1, padding="same")(x)
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
        """Compose the final morph from the network's residual correction.

        Same idea as the original: cross-dissolve A and B inside the face
        mask, add the network's small correction there, and fill everything
        OUTSIDE the mask (hair, ears, neck, background) mostly from A because
        there's no reliable A<->B correspondence there.

        === FIX #3 =========================================================
        The original filled the outside-mask region 100% from `warped_A`.
        Here we instead fill it with a small blend of A and the (A,B)
        cross-dissolve, controlled by BACKGROUND_BLEND. At BACKGROUND_BLEND=0
        this is byte-for-byte the same as the original formula.
        """
        blended = (1.0 - alpha) * warped_A + alpha * warped_B

        # Original: outside_mask_fill = warped_A (100% identity A).
        # Fixed:    outside_mask_fill = a small blend toward `blended` instead,
        #           so the background isn't structurally 100% A. BACKGROUND_BLEND
        #           is small (default 0.25) so most of the region is still A,
        #           keeping the original design's anti-ghosting intent.
        outside_mask_fill = ((1.0 - BACKGROUND_BLEND) * warped_A +
                             BACKGROUND_BLEND * blended)

        base = mask * blended + (1.0 - mask) * outside_mask_fill
        morph = base + delta * mask
        return tf.clip_by_value(morph, 0.0, 1.0)

    @tf.function
    def train_step(self, warped_A, warped_B, heatmaps, mask, alpha):
        # This whole method is structurally identical to the original -
        # discriminator update, then generator update. Only the loss
        # computed inside (`_balanced_loss`) changed - see below.
        X = tf.concat([warped_A, warped_B, heatmaps, mask], axis=-1)

        delta = self.model(X, training=True)
        morph = self._build_morph(warped_A, warped_B, delta, mask, alpha)
        with tf.GradientTape() as d_tape:
            d_real_A = self.discriminator(warped_A * mask, training=True)
            d_real_B = self.discriminator(warped_B * mask, training=True)
            d_fake = self.discriminator(morph * mask, training=True)
            d_loss = (tf.reduce_mean(tf.square(d_real_A - 1.0)) +
                      tf.reduce_mean(tf.square(d_real_B - 1.0)) +
                      2.0 * tf.reduce_mean(tf.square(d_fake))) * 0.25
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables))

        with tf.GradientTape() as g_tape:
            delta = self.model(X, training=True)
            morph = self._build_morph(warped_A, warped_B, delta, mask, alpha)
            recon = self._balanced_loss(morph, warped_A, warped_B, mask, alpha)
            d_fake = self.discriminator(morph * mask, training=False)
            adv = tf.reduce_mean(tf.square(d_fake - 1.0))
            g_loss = recon + self.ADV_WEIGHT * adv
        g_grads = g_tape.gradient(g_loss, self.model.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.model.trainable_variables))

        delta_mag = tf.reduce_mean(tf.abs(delta))
        return g_loss, adv, d_loss, delta_mag

    # ------------------------------------------------------------------ losses

    def _balanced_loss(self, morph, warped_A, warped_B, mask, alpha):
        # Unchanged: pixel/perceptual/TV terms, same weights as the original.
        loss_landmark = (
            (1.0 - alpha) * tf.reduce_mean(mask * tf.square(morph - warped_A)) +
            alpha * tf.reduce_mean(mask * tf.square(morph - warped_B)))

        loss_perceptual = self._perceptual_loss(morph, warped_A, warped_B, mask, alpha)
        loss_tv = self._tv_loss(morph, mask)

        # === FIX #1 + FIX #2 =================================================
        # _identity_loss (the anchor term, weight 25) now embeds the FULL,
        # unmasked image - see _facenet_embed below. _identity_balance_loss is
        # brand new (FIX #2): it directly penalizes the gap between the
        # morph's similarity to A and to B, which is exactly the metric
        # evaluation/02_vulnerability_eval.py's MMPMR depends on.
        loss_identity = self._identity_loss(morph, warped_A, warped_B, alpha)
        loss_identity_balance = self._identity_balance_loss(morph, warped_A, warped_B)

        return (3.0 * loss_landmark +
                6.0 * loss_perceptual +
                0.5 * loss_tv +
                25.0 * loss_identity +
                IDENTITY_BALANCE_WEIGHT * loss_identity_balance)

    def _perceptual_loss(self, morph, warped_A, warped_B, mask, alpha):
        # Unchanged from the original: perceptual quality still only cares
        # about the face region (mask), since that's the only region the
        # network actually modifies.
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

    def _facenet_embed(self, images):
        """FaceNet embedding of a FULL image (160x160, per-image standardized).

        === FIX #1 ==========================================================
        The original signature was `_facenet_embed(self, images, mask)` and
        computed `images * mask` here - zeroing the background before
        embedding. That made the identity loss blind to the fact that the
        background is (mostly) identity A, so training never had a reason to
        compensate for it. This version embeds the image AS IS, matching what
        a real face-recognition system sees when it scores the final photo.
        """
        x = tf.image.resize(images, (160, 160)) * 255.0
        mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        std = tf.math.reduce_std(x, axis=[1, 2, 3], keepdims=True)
        std = tf.maximum(std, 1.0 / tf.sqrt(160.0 * 160.0 * 3.0))
        emb = self.facenet((x - mean) / std)
        return tf.math.l2_normalize(emb, axis=-1)

    def _identity_loss(self, morph, warped_A, warped_B, alpha):
        """Identity-ANCHOR loss (same idea as the original, weight 25 in
        _balanced_loss): pull the morph's embedding toward the alpha-weighted
        midpoint of A's and B's embeddings. Only change: embeddings are now
        computed on the full, unmasked image (FIX #1)."""
        emb_M = self._facenet_embed(morph)
        emb_A = self._facenet_embed(warped_A)
        emb_B = self._facenet_embed(warped_B)

        target = tf.math.l2_normalize(
            (1.0 - alpha) * emb_A + alpha * emb_B, axis=-1)
        sim = tf.reduce_sum(emb_M * target, axis=-1)
        return tf.reduce_mean(1.0 - sim)

    def _identity_balance_loss(self, morph, warped_A, warped_B):
        """=== FIX #2 (new) ==================================================
        Identity-BALANCE loss: directly penalizes the gap between how similar
        the morph looks to A versus to B. This is the loss term the original
        model was missing - anchoring to the midpoint embedding does not by
        itself guarantee sim(morph, A) == sim(morph, B), since cosine
        similarity is not linear along the averaging direction. This term
        targets that gap directly, which is exactly what MMPMR measures
        (evaluation/02_vulnerability_eval.py: a morph only "succeeds" if its
        WORSE similarity, i.e. the smaller of the two, clears the threshold -
        so shrinking the gap between them raises MMPMR almost by definition).
        """
        emb_M = self._facenet_embed(morph)
        emb_A = self._facenet_embed(warped_A)
        emb_B = self._facenet_embed(warped_B)

        sim_A = tf.reduce_sum(emb_M * emb_A, axis=-1)
        sim_B = tf.reduce_sum(emb_M * emb_B, axis=-1)
        return tf.reduce_mean(tf.abs(sim_A - sim_B))

    def _tv_loss(self, morph, mask):
        # Unchanged from the original.
        dy = morph[:, 1:, :, :] - morph[:, :-1, :, :]
        dx = morph[:, :, 1:, :] - morph[:, :, :-1, :]
        return (tf.reduce_mean(mask[:, 1:, :, :] * tf.abs(dy)) +
                tf.reduce_mean(mask[:, :, 1:, :] * tf.abs(dx)))

    # ------------------------------------------------------------------ api
    # Everything below (eval_step, evaluate, fit, save, load, predict) is
    # IDENTICAL to the original - same training loop, same public API, so
    # main_train_balanced.py and main_generate_balanced.py can use this class
    # as a drop-in replacement for the original DiffusionModel.

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
