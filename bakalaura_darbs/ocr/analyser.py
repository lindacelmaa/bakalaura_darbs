import argparse
from pathlib import Path

from ocr.pdf_loader import PDFLoader
from ocr.text_localizer import TextLocalizer
from ocr.image_preprocessor import ImagePreprocessor


class AnalyseText:
    def __init__(self):
        self.args = self.set_args()

        self.pdf_path = Path(self.args.pdf_path)
        self.out_dir = Path(self.args.output_dir)
        self.ocr_engine = self.args.ocr  # tesseract or transformer
        self.threshold = self.args.threshold

        self.pdf_loader = PDFLoader(dpi=300)
        self.preprocessor = ImagePreprocessor(
            threshold=self.threshold,
            save_debug=self.args.save_preprocessed
        )
        self.localizer = TextLocalizer(lang="lav+eng", ocr_engine=self.ocr_engine, kraken_model=self.args.kraken_model)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "images").mkdir(exist_ok=True)
        (self.out_dir / f"localization_{self.ocr_engine}").mkdir(exist_ok=True)

    def set_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("pdf_path", help="Path to input PDF")
        parser.add_argument("output_dir", help="Output directory")
        parser.add_argument(
            "--ocr",
            choices=["tesseract", "transformer", "kraken"],
            default="tesseract",
            help="OCR engine to use: 'tesseract' (default), 'transformer' (TrOCR) or 'kraken' (Kraken)"
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.85,
            help="Grid removal intensity threshold"
        )
        parser.add_argument(
            "--save-preprocessed",
            action="store_true",
            help="Save preprocessed images to output/preprocessed"
        )
        parser.add_argument(
            "--pages",
            type=int,
            nargs="+",
            help="Only process specific pages"
        )
        parser.add_argument(
            "--no-ocr",
            action="store_true",
            help="Skip OCR, only run preprocessing"
        )
        parser.add_argument(
            "--use-preprocessed",
            action="store_true",
            help="Use already preprocessed images from output/preprocessed/ folder"
        )
        parser.add_argument(
            "--kraken-model",
            default=None,
            help="Path to .mlmodel file (required when --ocr kraken)"
        )
        parser.add_argument(
            "--use-pdf",
            action="store_true",
            help="Use PDF as input (default behavior)"
        )
        return parser.parse_args()

    def run(self):
        # Load PDF and split into pages
        if self.args.use_preprocessed:
            images_dir = self.out_dir / "preprocessed"
            images = sorted(images_dir.glob("*.png"))
            print(f"    Using preprocessed images from {images_dir}")

        elif self.args.use_pdf:
            images = self.pdf_loader.load(
                self.pdf_path,
                self.out_dir / "images"
            )
            print(f"    Loaded images from PDF: {len(images)} pages")

        if self.args.pages:
            images = [
                p for p in images
                if any(f"page_{n:04d}_" in p.name for n in self.args.pages)
            ]
            print(f"    Filtering to pages: {self.args.pages} → {len(images)} image(s)")

        # Skip preprocessing if using already preprocessed images
        if self.args.use_preprocessed:
            processed_images = images
            print(f"    Using preprocessed images from {images_dir}")
        else:
            print(f"    \nPreprocessing images (threshold={self.threshold})...")
            processed_images = self.preprocessor.process(images, self.out_dir)

        # Skip OCR
        if self.args.no_ocr:
            print("     \n--no-ocr flag set, skipping OCR. Done!")
            return

        # Text localization
        self.localizer.run(
            processed_images,
            self.out_dir / f"localization{self.ocr_engine}"
        )

        # Cleanup temp files
        if not self.args.save_preprocessed:
            self.preprocessor.cleanup(processed_images)
            print("     \nCleaned up temp files.")