from pathlib import Path
from PIL import Image
import numpy as np

class ImageLoader:

    def __init__(self, channels: int, base_path: str):
        self.channels = channels
        self.folder = Path(base_path)

    def load_images_from_path(self):
        return [img for _, img in self.load_images_with_paths()]

    def load_images_with_paths(self):
        """Return a list of (file_path, image_array) for every image found.

        Keeping the path lets callers recover each image's identity (e.g. for a
        leakage-free train/val split)."""
        image_dataset = []

        ### image extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

        for file_path in sorted(self.folder.rglob("*")):
            if file_path.suffix.lower() in image_extensions:
                try:
                    ### reading image
                    image = Image.open(file_path)

                    ### converting channels
                    if self.channels == 3:
                        image = image.convert("RGB")
                    elif self.channels == 1:
                        image = image.convert("L")

                    ## converting to numpy array
                    image_dataset.append((file_path, np.array(image)))

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        return image_dataset