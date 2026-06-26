"""Optional face-restoration pass (GFPGAN).

The morph pipeline produces a geometrically correct but slightly soft face -
the source images are CelebA (178x218), upscaled, and the L2 losses smooth out
pore/eyelash/hair detail. GFPGAN re-synthesizes plausible high-frequency facial
texture, which both removes residual blend mush and makes the result look like a
genuine photograph.

This is intentionally a soft dependency: GFPGAN pulls in PyTorch, which the rest
of this (TensorFlow) project does not need. If it isn't installed the caller gets
the un-restored image back plus a one-time install hint, so the pipeline still
runs end-to-end.

    pip install gfpgan basicsr facexlib

The model weights download automatically on first use to ~/.cache (or pass an
explicit `model_path`).
"""
import numpy as np

_RESTORER = None          # cached GFPGANer instance
_UNAVAILABLE = False      # set once if import/init fails, to avoid retry spam


def _patch_torchvision_functional_tensor():
    """Make basicsr importable on modern torchvision.

    basicsr (a GFPGAN dependency) does `from torchvision.transforms.functional_tensor
    import rgb_to_grayscale`, but that module was removed in torchvision >=0.17 - the
    function now lives in torchvision.transforms.functional. Register an alias so the
    old import path resolves, instead of patching the installed package by hand.
    """
    import sys
    import importlib
    name = "torchvision.transforms.functional_tensor"
    if name in sys.modules:
        return
    try:                                  # old path still present - nothing to do
        importlib.import_module(name)
    except ModuleNotFoundError:
        try:                              # new path - alias it under the old name
            sys.modules[name] = importlib.import_module(
                "torchvision.transforms.functional")
        except ModuleNotFoundError:
            pass                          # no torchvision at all; GFPGAN import fails next


def _get_restorer(upscale, model_path):
    global _RESTORER, _UNAVAILABLE
    if _RESTORER is not None or _UNAVAILABLE:
        return _RESTORER
    try:
        _patch_torchvision_functional_tensor()
        from gfpgan import GFPGANer
        # v1.4 is the latest general-purpose model; URL is used if no local path.
        url = ("https://github.com/TencentARC/GFPGAN/releases/download/"
               "v1.3.4/GFPGANv1.4.pth")
        _RESTORER = GFPGANer(
            model_path=model_path or url,
            upscale=upscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
    except Exception as e:                       # noqa: BLE001 - want broad guard
        _UNAVAILABLE = True
        print(f"[face_restore] GFPGAN unavailable ({e}); returning the "
              f"un-restored image. Install with: pip install gfpgan basicsr facexlib")
    return _RESTORER


def restore_face(image_rgb01, upscale=1, model_path=None):
    """Restore a single face image.

    image_rgb01 : HxWx3 float in [0, 1], RGB.
    Returns the restored image in the same format. If GFPGAN is not installed,
    returns the input unchanged.
    """
    restorer = _get_restorer(upscale, model_path)
    if restorer is None:
        return np.clip(image_rgb01, 0.0, 1.0).astype(np.float32)

    # GFPGAN works on BGR uint8 whole images and detects faces internally.
    bgr = np.clip(image_rgb01[:, :, ::-1], 0.0, 1.0)
    bgr = (bgr * 255.0).astype(np.uint8)

    _, _, restored_bgr = restorer.enhance(
        bgr, has_aligned=False, only_center_face=True, paste_back=True)

    if restored_bgr is None:                     # no face found - keep original
        return np.clip(image_rgb01, 0.0, 1.0).astype(np.float32)

    restored_rgb = restored_bgr[:, :, ::-1].astype(np.float32) / 255.0
    return np.clip(restored_rgb, 0.0, 1.0)
