"""FaceNet identity embeddings for partner selection.

Used to pair each face with the most identity-similar other face: morphing two
look-alike faces produces a far more realistic result than morphing two very
different identities. FaceNet is the same identity model the training loss uses
(keras_facenet), so the embeddings here are consistent with that anchor.
"""
import numpy as np
from PIL import Image
from tqdm import tqdm

_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from keras_facenet import FaceNet
        _EMBEDDER = FaceNet()
    return _EMBEDDER


def embed_images(images_uint8, batch_size=128):
    """Embed a list of HxWx3 uint8 RGB images. Returns Nx512 L2-normalized."""
    embedder = _get_embedder()
    out = []
    for i in range(0, len(images_uint8), batch_size):
        emb = embedder.embeddings(images_uint8[i:i + batch_size])
        out.append(emb)
    emb = np.concatenate(out, axis=0).astype(np.float32)
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)


def embed_paths(paths, size, batch_size=128, desc="Embedding faces"):
    """Load each image path (resized to `size`x`size`) and embed it. Nx512."""
    embeddings = []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc, unit="batch"):
        batch = [np.asarray(Image.open(p).convert("RGB").resize((size, size)),
                            dtype=np.uint8)
                 for p in paths[i:i + batch_size]]
        embeddings.append(embed_images(batch, batch_size=batch_size))
    return np.concatenate(embeddings, axis=0)
