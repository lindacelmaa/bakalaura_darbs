from pathlib import Path
from PIL import Image, ImageDraw

from kraken import blla, rpred
from kraken.lib import models


class KrakenOCREngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None

    def _load_model(self):
        if self._model is None:
            print(f"  Loading Kraken model: {self.model_path}")
            self._model = models.load_any(self.model_path)
        return self._model

    def run(self, image_paths: list[Path], output_dir: Path) -> dict[Path, list[dict]]:
        rec_model = self._load_model()
        results = {}

        for image_path in image_paths:
            print(f"  OCR processing: {image_path.name}")
            image = Image.open(image_path).convert("RGB")

            seg = blla.segment(image)

            words = []
            for record in rpred.rpred(rec_model, image, seg):
                text = record.prediction.strip()
                if not text:
                    continue

                if record.line:
                    xs = [p[0] for p in record.line]
                    ys = [p[1] for p in record.line]
                    x, y = int(min(xs)), int(min(ys))
                    w = int(max(xs) - min(xs))
                    h = max(int(max(ys) - min(ys)), 20)
                else:
                    x, y, w, h = 0, 0, 0, 20

                words.append({
                    "text": text,
                    "left": x,
                    "top": y,
                    "width": w,
                    "height": h,
                    "conf": 80,
                })

            results[image_path] = words
            print(f"  Found {len(words)} lines")

        print("OCR ready")
        return results