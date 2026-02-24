from pathlib import Path
from PIL import Image

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class OCREngine:
    def __init__(self, lang="lav+eng"):
        self.lang = lang

    def run(self, image_paths: list[Path], output_dir: Path) -> dict[Path, list[dict]]:

        results = {}
        for image_path in image_paths:
            print(f"  OCR processing: {image_path.name}")
            image = Image.open(image_path)

            data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                output_type=pytesseract.Output.DICT
            )

            words = []
            for i in range(len(data["text"])):
                word = data["text"][i].strip()
                if word and int(data["conf"][i]) > 30:  # filter low-confidence noise
                    words.append({
                        "text": word,
                        "left": data["left"][i],
                        "top": data["top"][i],
                        "width": data["width"][i],
                        "height": data["height"][i],
                        "conf": int(data["conf"][i]),
                    })

            results[image_path] = words
            print(f"Found {len(words)} words")

        print("OCR ready")
        return results
