"""Robustness transforms: JPEG re-compression and a simple print-scan simulation.

These approximate what a real document-issuance pipeline does to a submitted
photo. Neither is a substitute for an actual printer + scanner round trip - the
print-scan function is explicitly a simulation (blur + mild resampling + noise),
not a physical reproduction - see README.md "Robustness pass" for the caveat.
"""
import cv2
import numpy as np


def jpeg_recompress(img_rgb01, quality=50):
    """Re-encode as JPEG at `quality` and decode back. img_rgb01: HxWx3 float [0,1]."""
    bgr = cv2.cvtColor((np.clip(img_rgb01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return img_rgb01
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb


def print_scan_simulate(img_rgb01, downscale=0.6, blur_ksize=3, noise_std=3.0,
                         jpeg_quality=70, seed=None):
    """Approximate a print + scan round trip.

    Steps (each a documented, simplified stand-in for a physical effect):
      1. Downscale then upscale back - print/scan resampling loses high frequencies.
      2. Slight Gaussian blur - optical softening from the printer/scanner.
      3. Additive Gaussian sensor noise - scanner sensor noise.
      4. JPEG re-compression - most scan pipelines re-encode the digitized image.
    """
    rng = np.random.default_rng(seed)
    h, w = img_rgb01.shape[:2]
    small = cv2.resize(img_rgb01, (max(1, int(w * downscale)), max(1, int(h * downscale))),
                       interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if blur_ksize > 0:
        k = blur_ksize * 2 + 1
        back = cv2.GaussianBlur(back, (k, k), 0)
    noise = rng.normal(0, noise_std / 255.0, back.shape).astype(np.float32)
    noisy = np.clip(back + noise, 0.0, 1.0)
    return jpeg_recompress(noisy, quality=jpeg_quality)
