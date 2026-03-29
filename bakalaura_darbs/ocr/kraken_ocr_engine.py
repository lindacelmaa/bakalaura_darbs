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
            print(f"  Segmentation found {len(seg.lines)} lines")

            debug_img = image.copy()
            draw = ImageDraw.Draw(debug_img)
            for line in seg.lines:
                pts = line.baseline
                if pts:
                    draw.line(pts, fill="red", width=3)
                poly = line.boundary
                if poly:
                    draw.polygon(poly, outline="blue")

            seg_path = output_dir / f"{image_path.stem}_segmentation.png"
            debug_img.save(seg_path)
            print(f"  Saved segmentation: {seg_path}")

            words = []
            for record in rpred.rpred(rec_model, image, seg):
                text = record.prediction.strip()
                if not text:
                    continue

                try:
                    x, y, w, h = record.bbox
                    w = w - x
                    h = h - y
                except Exception:
                    x, y, w, h = 0, 0, 100, 20

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
