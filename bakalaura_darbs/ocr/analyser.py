import argparse
from pathlib import Path

from ocr.pdf_loader import PDFLoader
from ocr.text_localizer import TextLocalizer

class AnalyseText:
    def __init__(self):
        self.args = self.set_args()

        self.pdf_path = Path(self.args.pdf_path)
        self.out_dir = Path(self.args.output_dir)

        self.pdf_loader = PDFLoader(dpi=300)
        self.localizer = TextLocalizer(lang="lav+eng")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "images").mkdir(exist_ok=True)
        (self.out_dir / "localization").mkdir(exist_ok=True)

    def set_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("pdf_path", help="Path to input PDF")
        parser.add_argument("output_dir", help="Output directory")
        return parser.parse_args()

    def run(self):
        #Load PDF and split into pages
        images = self.pdf_loader.load(
            self.pdf_path,
            self.out_dir / "images"
        )
        print(f"{len(images)} pages saved to {self.out_dir / 'images'}")

        #Text localization
        self.localizer.run(
            images,
            self.out_dir / "localization"
        )