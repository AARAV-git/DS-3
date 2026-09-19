"""
Balance PNG vs JPEG Formats on Disk (50/50 ratio per class) for DS-2 Training Set
"""
import os, random
from PIL import Image

DATASET_ROOT = os.path.join("dataset", "training")

ORIGIN_DIRS = {
    "ai": "AI Images",
    "ai_edited": "AI-edited",
    "real": "Non-AI Images"
}

def convert_png_to_jpg(png_path, quality=92):
    try:
        jpg_path = os.path.splitext(png_path)[0] + ".jpg"
        with Image.open(png_path) as im:
            rgb_im = im.convert("RGB")
            rgb_im.save(jpg_path, "JPEG", quality=quality)
        os.remove(png_path)
        return True
    except Exception as e:
        print(f"Error converting {png_path}: {e}")
        return False

def convert_jpg_to_png(jpg_path):
    try:
        png_path = os.path.splitext(jpg_path)[0] + ".png"
        with Image.open(jpg_path) as im:
            rgb_im = im.convert("RGB")
            rgb_im.save(png_path, "PNG")
        os.remove(jpg_path)
        return True
    except Exception as e:
        print(f"Error converting {jpg_path}: {e}")
        return False

def balance_class_formats(class_name, orig_folder):
    base_p = os.path.join(DATASET_ROOT, orig_folder)
    if not os.path.isdir(base_p): return

    png_files = []
    jpg_files = []

    for root, dirs, files in os.walk(base_p):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            fp = os.path.join(root, f)
            if ext == ".png":
                png_files.append(fp)
            elif ext in [".jpg", ".jpeg"]:
                jpg_files.append(fp)

    total = len(png_files) + len(jpg_files)
    target_png = total // 2
    target_jpg = total - target_png

    print(f"\n--- Class: {class_name.upper()} ---")
    print(f"  Current: {len(png_files)} PNGs | {len(jpg_files)} JPGs (Total: {total})")
    print(f"  Target : {target_png} PNGs | {target_jpg} JPGs")

    random.seed(42)

    if len(png_files) > target_png:
        num_to_convert = len(png_files) - target_png
        to_convert = random.sample(png_files, num_to_convert)
        converted = 0
        for fp in to_convert:
            if convert_png_to_jpg(fp, quality=random.randint(90, 96)):
                converted += 1
        print(f"  Converted {converted} PNGs -> JPGs")

    elif len(jpg_files) > target_jpg:
        num_to_convert = len(jpg_files) - target_jpg
        to_convert = random.sample(jpg_files, num_to_convert)
        converted = 0
        for fp in to_convert:
            if convert_jpg_to_png(fp):
                converted += 1
        print(f"  Converted {converted} JPGs -> PNGs")

def main():
    print("=" * 70)
    print("  Balancing PNG/JPG Formats (50/50 Ratio) in DS-2 Training Set")
    print("=" * 70)

    for cname, folder in ORIGIN_DIRS.items():
        balance_class_formats(cname, folder)

    print("\n" + "=" * 70)
    print("  FORMAT BALANCING COMPLETE!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
