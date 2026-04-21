import os
import shutil
import time

from PIL import Image

IMAGE_FOLDER = "C:/Users/linda/Desktop/bakalaura_darbs/bakalaura_darbs/localizationkraken (2)/localizationkraken/objects"
TRASH_FOLDER = "trash"
OUTPUT_MAPPING = "labels.txt"

def show_image(path):
    img = Image.open(path)
    img.show()

def reject_handle(path):
    os.makedirs(TRASH_FOLDER, exist_ok=True)

    base = os.path.basename(path)
    name, ext = os.path.splitext(base)

    timestamp = int(time.time())
    new_name = f"{name}_{timestamp}{ext}"

    destination = os.path.join(TRASH_FOLDER, new_name)
    shutil.move(path, destination)

    print(f"    Location changed to {destination}")


def save_results(results):
    with open(OUTPUT_MAPPING, "w", encoding="utf-8") as f:
        for name, text in results:
            f.write(f"{name}\t{text}\n")

def create_gt_files(results):
    for name, text in results:
        base, _ = os.path.splitext(name)
        gt_name = base + ".gt.txt"
        gt_path = os.path.join(IMAGE_FOLDER, gt_name)

        with open(gt_path, "w", encoding="utf-8") as f:
            f.write(text)

def main():
    images = sorted([
        img_format for img_format in os.listdir(IMAGE_FOLDER)
        if img_format.lower().endswith(".png")
    ])

    results = []

    for img_name in images:
        full_path = os.path.join(IMAGE_FOLDER, img_name)
        if not os.path.exists(full_path):
            continue

        base, _ = os.path.splitext(img_name)
        gt_path = os.path.join(IMAGE_FOLDER, base + ".gt.txt")

        if os.path.exists(gt_path):
            print(f"    Skip already annotated: {img_name}")
            continue

        print(f"\n------- {img_name} ------")
        show_image(full_path)

        while True:
            decision = input("  Usable? (y/n/q): ").strip().lower()

            if decision == "q":
                save_results(results)
                create_gt_files(results)
                print("Canceled.")
                return

            if decision == "n":
                reject_handle(full_path)
                break

            if decision == "y":
                text = input("  Input text: ").strip()
                results.append((img_name, text))
                break

    save_results(results)
    create_gt_files(results)
    print("     Done!")


if __name__ == "__main__":
    main()