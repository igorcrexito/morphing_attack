from pathlib import Path
from PIL import Image
import numpy as np

class ImageLoader:

    def __init__(self, channels: int, base_path: str):
        self.channels = channels
        self.folder = Path(base_path)

    def load_images_from_path(self):

        image_dataset = []

        ### image extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

        for file_path in self.folder.rglob("*"):
            if file_path.suffix.lower() in image_extensions:
                try:
                    ### reading imaged
                    image = Image.open(file_path)

                    ### converting channels
                    if self.channels == 3:
                        image = image.convert("RGB")
                    elif self.channels == 1:
                        image = image.convert("L")

                    ## converting to numpy array
                    image_dataset.append(np.array(image))

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        return image_dataset