from pathlib import Path
from PIL import Image

"""
https://learnopencv.com/trocr-getting-started-with-transformer-based-ocr/
https://huggingface.co/microsoft/trocr-base-handwritten
https://huggingface.co/docs/transformers/model_doc/trocr
"""

class TransformerOCREngine:
    def __init__(self, lang="lav+eng"):
        self.lang = lang
        self._load_model()

    def _load_model(self):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        print("     Loading TrOCR model...")
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = self.model.to("cpu")
        self.model.eval()
        print("     TrOCR model loaded.")

    def _get_word_boxes(self, image: Image.Image) -> list[dict]:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        import torch

        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT
        )

        words = []
        for i in range(len(data["text"])):
            if int(data["conf"][i]) < 10:
                continue

            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            if w <= 0 or h <= 0:
                continue

            crop = image.crop((x, y, x + w, y + h)).convert("RGB")
            pixel_values = self.processor(images=crop, return_tensors="pt").pixel_values

            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            if text:
                words.append({
                    "text": text,
                    "left": x,
                    "top": y,
                    "width": w,
                    "height": h,
                    "conf": int(data["conf"][i]),
                })

        return words

    def run(self, image_paths: list[Path], output_dir: Path) -> dict[Path, list[dict]]:
        results = {}
        for image_path in image_paths:
            print(f"    TrOCR processing: {image_path.name}")
            image = Image.open(image_path).convert("RGB")
            words = self._get_word_boxes(image)
            results[image_path] = words
            print(f"    Found {len(words)} words")

        print("     TrOCR ready")
        return results