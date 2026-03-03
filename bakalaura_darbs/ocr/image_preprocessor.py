from pathlib import Path
import numpy as np
from PIL import Image
from skimage.color import rgb2gray


class ImagePreprocessor:
    def __init__(self, threshold=0.85, save_debug=False):
        """
        pixels brighter than this become white.
        saves the preprocessed images to output_dir/preprocessed/
        """
        self.threshold = threshold
        self.save_debug = save_debug

    def process(self, image_paths: list[Path], output_dir: Path) -> list[Path]:
        debug_dir = output_dir / "preprocessed"
        if self.save_debug:
            debug_dir.mkdir(parents=True, exist_ok=True)

        processed_paths = []

        for image_path in image_paths:
            print(f"  Preprocessing: {image_path.name}")

            image = Image.open(image_path).convert("RGB")
            gray = rgb2gray(np.array(image))

            bw = (gray > self.threshold).astype(np.uint8) * 255
            result = Image.fromarray(bw).convert("RGB")

            if self.save_debug:
                save_path = debug_dir / f"{image_path.stem}_preprocessed.png"
                result.save(save_path)
                processed_paths.append(save_path)
                print(f"Saved preprocessed → {save_path}")
            else:
                save_path = image_path.parent / f"{image_path.stem}_preprocessed.png"
                result.save(save_path)
                processed_paths.append(save_path)

        return processed_paths

    def cleanup(self, processed_paths: list[Path]):
        for path in processed_paths:
            if "_preprocessed" in path.name and path.exists():
                path.unlink()




