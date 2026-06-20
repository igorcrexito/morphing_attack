from descriptors.image_descriptor import ImageDescriptor
import numpy as np
import cv2
from skimage.feature import local_binary_pattern


class ELBPDescriptor(ImageDescriptor):

    def __init__(self, descriptor_name: str, method: str = 'uniform', radius: int = 1, neighbors: int = 8):
        super().__init__(descriptor_name)
        self.radius = radius
        self.neighbors = neighbors
        self.method = method

    def compute_feature(self, image: np.ndarray):
        height, width = image.shape

        elbp_image_final = np.ones((height, width, 1), dtype=np.uint8)

        lbp = local_binary_pattern(image, self.neighbors, self.radius, self.method)*255
        elbp_image_final[:, :, 0] = lbp
        return elbp_image_final

    def _compute_feature(self, image: np.ndarray):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        height, width = image.shape
        elbp_image = np.zeros((height, width), dtype=np.uint8)
        elbp_image_final = np.ones((height, width, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                center = image[y, x]
                binary_pattern = 0

                for n in range(self.neighbors):
                    angle = 2 * np.pi * n / self.neighbors
                    neighbor_x = x + self.radius * np.cos(angle)
                    neighbor_y = y - self.radius * np.sin(angle)

                    if 0 <= neighbor_x < width and 0 <= neighbor_y < height:
                        neighbor_value = image[int(neighbor_y), int(neighbor_x)]
                        binary_pattern |= (neighbor_value >= center) << n

                elbp_image[y, x] = binary_pattern

        elbp_image_final[:, :, 0] = elbp_image
        elbp_image_final[:, :, 1] = elbp_image
        elbp_image_final[:, :, 2] = elbp_image

        return elbp_image_final
