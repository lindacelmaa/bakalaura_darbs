import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class TextLocalizer:
    def __init__(self, lang="lav+eng", ocr_engine="tesseract", kraken_model=None):
        self.lang = lang
        self.ocr_engine = ocr_engine
        self.kraken_model = kraken_model

    def _get_engine(self):
        if self.ocr_engine == "transformer":
            from ocr.transformer_ocr_engine import TransformerOCREngine
            return TransformerOCREngine(lang=self.lang)
        elif self.ocr_engine == "kraken":
            from ocr.kraken_ocr_engine import KrakenOCREngine
            if not self.kraken_model:
                raise ValueError("--kraken-model path is required when using --ocr kraken")
            return KrakenOCREngine(model_path=self.kraken_model)
        else:
            from ocr.tesseract_ocr_engine import OCREngine
            return OCREngine(lang=self.lang)

    def _extract_objects_as_arrays(self, image_path: Path, words: list[dict], min_width: int = 0, min_height: int = 0) -> list[np.ndarray]:

        gray_path = image_path.parent / image_path.name.replace("_preprocessed", "_gray")

        if gray_path.exists():
            image = Image.open(gray_path).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        objects = []

        for word in words:
            x, y = word["left"], word["top"]
            w, h = word["width"], word["height"]

            if w < min_width or h < min_height:
                continue


            cropped = image_np[y:y + h, x:x + w]

            objects.append({
                "image": cropped,
                "bbox": (x, y, w, h),
                "text": word["text"]
            })

        return objects

    def _save_objects(self, objects, output_dir: Path, prefix: str):
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = []

        for i, obj in enumerate(objects):
            img = Image.fromarray(obj["image"])
            filename = f"{prefix}_obj_{i:03d}.png"
            img.save(output_dir / filename)

            metadata.append({
                "file": filename,
                "bbox": obj["bbox"],
                "text": obj["text"]
            })

        with open(output_dir / f"{prefix}_meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def run(self, image_paths: list[Path], output_dir: Path) -> dict[Path, list[dict]]:
        engine = self._get_engine()

        print(f"    {len(image_paths)} images ready for localization ({self.ocr_engine})")
        ocr_results = engine.run(image_paths, output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        objects_dir = output_dir / "objects"
        all_objects = {}

        for image_path, words in ocr_results.items():
            print(f"  - {image_path.name}")

            # boxes
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)

            for word in words:
                x, y, w, h = word["left"], word["top"], word["width"], word["height"]
                draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
                draw.text((x, max(0, y - 12)), word["text"], fill="red")

            annotated_path = output_dir / f"{image_path.stem}_localized.png"
            image.save(annotated_path)
            print(f"    Saved image: {annotated_path}")

            objects = self._extract_objects_as_arrays(
                image_path,
                words,
                min_width=20,
                min_height=10
            )

            all_objects[image_path] = objects
            self._save_objects(objects, objects_dir, image_path.stem)

            print(f"    Extracted {len(objects)} objects")

            text_lines = " ".join(w["text"] for w in words)
            text_path = output_dir / f"{image_path.stem}_text.txt"
            text_path.write_text(text_lines, encoding="utf-8")
            print(f"    Saved text: {text_path}")
            print(f"    Preview: {text_lines[:200]}")

        return ocr_results