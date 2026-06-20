from descriptors.image_descriptor import ImageDescriptor
from skimage.restoration import estimate_sigma, denoise_wavelet
import numpy as np
import cv2


class ReflectanceDescriptor(ImageDescriptor):

    def __init__(self, descriptor_name: str, sigma: float):
        super().__init__(descriptor_name)
        self.sigma = sigma

    def compute_feature(self, image: np.ndarray, num_channels=3):

        if num_channels == 3:
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        channels = cv2.split(image)
        retinex_channels = []

        for channel in channels:
            channel = channel.astype(np.float32)
            log_channel = np.log1p(channel)
            blurred_log_channel = cv2.GaussianBlur(log_channel, (0, 0), self.sigma)
            retinex_channel = log_channel - blurred_log_channel

            retinex_channels.append(retinex_channel)

        return np.array(cv2.merge(retinex_channels))