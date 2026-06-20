from abc import abstractmethod, ABC
import numpy as np


class ImageDescriptor(ABC):
    """
    Abstract class that defines image descriptors
    """

    def __init__(self, descriptor_name: str):
        """
        Defining the constructor of the descriptor
        """
        self.descriptor_name = descriptor_name

    @abstractmethod
    def compute_feature(self, image: np.ndarray):
        """
        Abstract method that defines the behavior of the feature extractor
        """
        pass
