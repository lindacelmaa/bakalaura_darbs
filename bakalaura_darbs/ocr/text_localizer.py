from pathlib import Path
from PIL import Image, ImageDraw


class TextLocalizer:
    def __init__(self, lang="lav+eng", ocr_engine="tesseract"):
        self.lang = lang
        self.ocr_engine = ocr_engine

    def _get_engine(self):
        if self.ocr_engine == "transformer":
            from ocr.transformer_ocr_engine import TransformerOCREngine
            return TransformerOCREngine(lang=self.lang)
        else:
            from ocr.tesseract_ocr_engine import OCREngine
            return OCREngine(lang=self.lang)

    def run(self, image_paths: list[Path], output_dir: Path) -> dict[Path, list[dict]]:
        engine = self._get_engine()

        print(f"{len(image_paths)} images ready for localization ({self.ocr_engine})")
        ocr_results = engine.run(image_paths, output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

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
            print(f"Saved image: {annotated_path}")

            text_lines = " ".join(w["text"] for w in words)
            text_path = output_dir / f"{image_path.stem}_text.txt"
            text_path.write_text(text_lines, encoding="utf-8")
            print(f"Saved text: {text_path}")
            print(f"Preview: {text_lines[:200]}")

        return ocr_results