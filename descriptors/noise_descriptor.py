from descriptors.image_descriptor import ImageDescriptor
from skimage.restoration import estimate_sigma, denoise_wavelet
import numpy as np


class NoiseDescriptor(ImageDescriptor):

    def compute_feature(self, image: np.ndarray):
        sigma_est = estimate_sigma(image, channel_axis=-1, average_sigmas=True)
        img_visushrink = denoise_wavelet(image, method='VisuShrink', mode='hard',
                                         sigma=sigma_est, wavelet_levels=5,
                                         wavelet='db8', channel_axis=-1, convert2ycbcr=False,
                                         rescale_sigma=True)

        filtered = img_visushrink * 255
        filtered = filtered.astype(np.uint8)
        noise_visu = image - filtered

        return noise_visu
