from pathlib import Path


class TextLocalizer:
    def __init__(self, lang="lav+eng"):
        self.lang = lang

    def run(self, image_paths: list[Path], output_dir: Path):
        #TODO: implement text localization on each page
        print(f"{len(image_paths)} images ready for localization")
        for image_path in image_paths:
            print(f"  - {image_path.name}")