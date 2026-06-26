"""Warped-pair caching + tf.data input pipeline.

Caching: warping with OpenCV is CPU work we don't want to repeat every run, so a
one-off preprocessing pass (main_prepare_pairs.py) writes the aligned tensors to
a single .npz per split. We deliberately do NOT cache the 68-channel heatmaps -
at 224x224x68 float32 (~13 MB/sample) they would dwarf everything else. Instead
we cache the (tiny) mean landmarks and regenerate heatmaps on the fly per batch.
"""
import os
import numpy as np
import tensorflow as tf

from morphing.warp import morph_pair


def landmarks_to_heatmaps(landmarks, width, height, sigma=3):
    """Vectorized Gaussian heatmaps, HxWxN, from Nx2 landmarks."""
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    xx = xx[..., None].astype(np.float32)
    yy = yy[..., None].astype(np.float32)
    lx = landmarks[:, 0].astype(np.float32)
    ly = landmarks[:, 1].astype(np.float32)
    hm = np.exp(-((xx - lx) ** 2 + (yy - ly) ** 2) / (2.0 * sigma ** 2))
    return hm.astype(np.float32)



def build_cache(output_dir, pairs, alpha, width, height, cache_path):
    """Warp every pair once and persist the aligned tensors to `cache_path`."""
    warpedA, warpedB, mask, mean_lm = [], [], [], []

    from PIL import Image
    for a, b in pairs:
        imgA = np.array(Image.open(os.path.join(output_dir, a))).astype(np.float32) / 255.0
        imgB = np.array(Image.open(os.path.join(output_dir, b))).astype(np.float32) / 255.0
        lmA = np.loadtxt(os.path.join(output_dir, a.replace(".jpg", ".csv")), delimiter=",", skiprows=1)
        lmB = np.loadtxt(os.path.join(output_dir, b.replace(".jpg", ".csv")), delimiter=",", skiprows=1)

        wA, wB, mlm, mk = morph_pair(imgA, imgB, lmA, lmB, alpha, width, height)
        warpedA.append(wA)
        warpedB.append(wB)
        mask.append(mk)
        mean_lm.append(mlm)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez_compressed(
        cache_path,
        warpedA=np.asarray(warpedA, np.float32),
        warpedB=np.asarray(warpedB, np.float32),
        mask=np.asarray(mask, np.float32),
        mean_lm=np.asarray(mean_lm, np.float32),
        alpha=np.float32(alpha))
    return len(pairs)


def make_dataset(cache_path, width, height, batch_size, shuffle, sigma=3):
    """Load a cached split and build a batched tf.data pipeline that generates
    the heatmap channels on the fly."""
    data = np.load(cache_path)
    warpedA = data["warpedA"]
    warpedB = data["warpedB"]
    mask = data["mask"]
    mean_lm = data["mean_lm"]
    n = len(warpedA)

    ds = tf.data.Dataset.from_tensor_slices((warpedA, warpedB, mean_lm, mask))
    if shuffle:
        ds = ds.shuffle(n)

    def _add_heatmaps(wA, wB, lm, mk):
        hm = tf.py_function(
            lambda l: landmarks_to_heatmaps(l.numpy(), width, height, sigma),
            [lm], tf.float32)
        hm.set_shape((height, width, mean_lm.shape[1]))
        return wA, wB, hm, mk

    ds = ds.map(_add_heatmaps, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, n
