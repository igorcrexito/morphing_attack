import pandas as pd

from descriptors.image_descriptor import ImageDescriptor
import numpy as np
import scipy
import math
from scipy import signal
import tqdm


class BSIFDescriptor(ImageDescriptor):

    def __init__(self, descriptor_name: str, base_path: str = '../filters/texturefilters/', extension: str = ".mat",
                 compute_histogram: bool = False, image_size: tuple = (224, 224), filter_size: str = '3x3_5'):
        super().__init__(descriptor_name)
        self.base_path = base_path
        self.extension = extension
        self.compute_histogram = compute_histogram
        self.image_size = image_size
        self.filter_size = filter_size

    def compute_feature(self, image: np.ndarray):

        ## retrieving all filters from the filter folder
        filter = f"{self.base_path}ICAtextureFilters_{self.filter_size}bit{self.extension}"

        f = scipy.io.loadmat(filter)
        texturefilters = f.get('ICAtextureFilters')

        # Initialize
        numScl = np.shape(texturefilters)[2]
        codeImg = np.ones(np.shape(image))
        output_image = np.ones((np.shape(image)[0], np.shape(image)[1], 3))

        # Make spatial coordinates for sliding window
        r = int(math.floor(np.shape(texturefilters)[0] / 2))

        # Wrap image (increase image size according to maximum filter radius by wrapping around)
        upimg = image[0:r, :]
        btimg = image[-r:, :]
        lfimg = image[:, 0:r]
        rtimg = image[:, -r:]
        cr11 = image[0:r, 0:r]
        cr12 = image[0:r, -r:]
        cr21 = image[-r:, 0:r]
        cr22 = image[-r:, -r:]

        imgWrap = np.vstack(
            (np.hstack((cr22, btimg, cr21)), np.hstack((rtimg, image, lfimg)), np.hstack((cr12, upimg, cr11))))

        # Loop over scales
        for i in range(numScl):
            tmp = texturefilters[:, :, numScl - i - 1]
            ci = signal.convolve2d(imgWrap, np.rot90(tmp, 2), mode='valid')
            t = np.multiply(np.double(ci > 0), 2 ** i)
            codeImg = codeImg + t

        hist_bsif = None
        if self.compute_histogram:
            hist_bsif = np.histogram(codeImg.ravel(), bins=np.arange(1, (2 ** numScl) + 2))
            hist_bsif = hist_bsif[0]
            # normalize the histogram
            hist_bsif = hist_bsif / (hist_bsif.sum() + 1e-7)

        output_image[:, :, 0] = codeImg
        output_image[:, :, 1] = codeImg
        output_image[:, :, 2] = codeImg

        return output_image, hist_bsif
