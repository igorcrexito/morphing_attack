import glob
from PIL import Image, ImageOps

class ImageLoader:

    def __init__(self, width: int, height: int, channels: 3):
        self.width = width
        self.height = height
        self.channels = channels


    def load_images_from_path(self, path: str):
        pass