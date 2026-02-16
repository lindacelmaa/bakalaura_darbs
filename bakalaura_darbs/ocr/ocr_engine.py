from pathlib import Path


class OCREngine:
    def __init__(self, lang="lav+eng"):
        self.lang = lang

    def run(self, image_paths: list[Path], output_dir: Path):
        # TODO: implement OCR (e.g. Tesseract) on each page image
        print(f"OCR ready")