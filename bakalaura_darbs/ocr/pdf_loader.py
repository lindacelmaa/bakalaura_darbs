from pathlib import Path
from pdf2image import convert_from_path


class PDFLoader:
    def __init__(self, dpi=300):
        self.dpi = dpi

    def load(self, pdf_path: Path, output_dir: Path) -> list[Path]:
        print(f"Loading PDF: {pdf_path}")
        pages = convert_from_path(pdf_path, dpi=self.dpi)

        image_paths = []
        for i, page in enumerate(pages, start=1):
            path = output_dir / f"page_{i:04d}.png"
            page.save(path, "PNG")
            image_paths.append(path)
            print(f"Saved page {i}/{len(pages)} -> {path.name}")

        return image_paths