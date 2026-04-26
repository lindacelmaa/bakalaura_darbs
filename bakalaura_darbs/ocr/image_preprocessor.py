from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.morphology import remove_small_objects
from skimage.transform import hough_line, hough_line_peaks, rotate
from skimage.feature import canny

"""
https://scikit-image.org/docs/stable/auto_examples/edges/plot_line_hough_transform.html
https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.rotate
https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.remove_small_holes
https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.rgb2gray
https://stackoverflow.com/questions/47864265/how-would-i-detect-the-angle-of-these-two-lines
https://github.com/stephanefschwarz/Hough-Transform/blob/master/HoughTransform.ipynb
"""

class ImagePreprocessor:
    def __init__(self, threshold=0.8, save_debug=False, min_object_size=50,
                 deskew=True, deskew_min_line_length=200):

        self.threshold = threshold
        self.save_debug = save_debug
        self.min_object_size = min_object_size
        self.deskew = deskew
        self.deskew_min_line_length = deskew_min_line_length

    def _remove_noise(self, bw: np.ndarray) -> np.ndarray:
        dark = bw == 0
        cleaned = remove_small_objects(dark, min_size=self.min_object_size)
        result = bw.copy()
        result[dark & ~cleaned] = 255   # restore small blobs → white
        return result

    def _detect_skew_angle(self, gray: np.ndarray) -> float:

        edges = canny(gray, sigma=2)

        angle_range = np.deg2rad(10)
        tested_angles = np.linspace(np.pi / 2 - angle_range, np.pi / 2 + angle_range, 100)
        h, theta, d = hough_line(edges, theta=tested_angles)

        peaks = hough_line_peaks(h, theta, d, num_peaks=20, threshold=0.3 * h.max())

        if not peaks or len(peaks[0]) == 0:
            print("    Deskew: no Hough peaks found, skipping.")
            return 0.0

        height, width = gray.shape
        diag = np.sqrt(height ** 2 + width ** 2)

        long_line_angles = []
        for _, angle, dist in zip(*peaks):
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            endpoints = []
            for x in [0, width - 1]:
                if abs(sin_a) > 1e-6:
                    y = (dist - x * cos_a) / sin_a
                    if 0 <= y < height:
                        endpoints.append((x, y))
            for y in [0, height - 1]:
                if abs(cos_a) > 1e-6:
                    x = (dist - y * sin_a) / cos_a
                    if 0 <= x < width:
                        endpoints.append((x, y))

            if len(endpoints) >= 2:
                p1, p2 = endpoints[0], endpoints[1]
                length = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                if length >= self.deskew_min_line_length:
                    long_line_angles.append(np.rad2deg(angle))

        if not long_line_angles:
            print(f"    Deskew: no lines ≥ {self.deskew_min_line_length}px found, skipping.")
            return 0.0

        median_angle = float(np.median(long_line_angles))

        rotation_angle = median_angle - 90.0

        rotation_angle = float(np.clip(rotation_angle, -45, 45))
        print(f"    Deskew: median line angle = {median_angle:.2f}°  →  rotating {rotation_angle:.2f}°")
        return rotation_angle

    def _deskew(self, image: np.ndarray, gray: np.ndarray, cval=1.0) -> np.ndarray:
        angle = self._detect_skew_angle(gray)
        if abs(angle) < 0.1:
            return image

        rotated = rotate(
            image.astype(np.float32) / 255.0,
            angle=angle,
            resize=False,
            cval=cval,
            preserve_range=True,
        )
        return (rotated * 255).astype(np.uint8)


    def process(self, image_paths: list[Path], output_dir: Path) -> list[Path]:
        debug_dir = output_dir / "preprocessed"
        if self.save_debug:
            debug_dir.mkdir(parents=True, exist_ok=True)

        processed_paths = []

        for image_path in image_paths:
            print(f"    Preprocessing: {image_path.name}")

            image = Image.open(image_path).convert("RGB")
            gray = rgb2gray(np.array(image))

            bw = (gray > self.threshold).astype(np.uint8) * 255

            bw = self._remove_noise(bw)
            print(f"    Noise removal done (min_size = {self.min_object_size}px)")

            if self.deskew:
                bw = self._deskew(bw, gray, cval=1.0)
                gray_uint8 = (gray * 255).astype(np.uint8)
                gray_uint8 = self._deskew(gray_uint8, gray, cval=255)

            else:
                gray_uint8 = (gray * 255).astype(np.uint8)

            result = Image.fromarray(bw).convert("RGB")

            if self.save_debug:
                save_path = debug_dir / f"{image_path.stem}_preprocessed.png"
            else:
                save_path = image_path.parent / f"{image_path.stem}_preprocessed.png"

            result.save(save_path)
            processed_paths.append(save_path)
            print(f"    Saved → {save_path}")

            result_gray = Image.fromarray(gray_uint8).convert("RGB")
            if self.save_debug:
                gray_path = debug_dir / f"{image_path.stem}_gray.png"
            else:
                gray_path = image_path.parent / f"{image_path.stem}_gray.png"
            result_gray.save(gray_path)
            print(f"    Saved gray → {gray_path}")

        return processed_paths

    def cleanup(self, processed_paths: list[Path]):
        for path in processed_paths:
            if "_preprocessed" in path.name and path.exists():
                path.unlink()