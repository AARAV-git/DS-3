"""
DS-MTFNet ─ Dataset Reorganizer
=================================
Copies images from the original folder naming convention into the
standard DS-MTFNet structure that src/dataset.py expects.

Run ONCE from the DS-MTFNet root directory:
    python reorganize_dataset.py

Source layout (what you have):
    dataset/training/AI Images/AI_faces_images/
    dataset/training/AI Images/ai_generated_Human/
    dataset/training/AI Images/AI_animal_images/CAT/ DOG/ …
    dataset/training/Non-AI Images/real_faces_images/
    dataset/training/Non-AI Images/real_human_images/
    dataset/training/Non-AI Images/real_animal_images/CAT/ DOG/ …
    dataset/training/AI-edited/AI_face_edited/
    dataset/training/AI-edited/AI_Human_edited/
    dataset/training/AI-edited/AI_animal_edited/Cat/ Dog/ …
    (same pattern under validation/ and testing/)

Target layout (what DS-MTFNet needs):
    dataset/train/ai/face/
    dataset/train/ai/human/
    dataset/train/ai/animal/cat/ dog/ elephant/ horse/ lion/
    dataset/train/real/face/
    dataset/train/real/human/
    dataset/train/real/animal/cat/ dog/ elephant/ horse/ lion/
    dataset/train/ai_edited/face/
    dataset/train/ai_edited/human/
    dataset/train/ai_edited/animal/cat/ dog/ elephant/ horse/ lion/
    (same pattern under val/ and test/)
"""

import os
import shutil

# ──────────────────────────────────────────────────────────────────────────────
# Mapping: (source_split, source_origin_folder, source_content_path)
#       -> (target_split, target_origin,        target_content_path)
# ──────────────────────────────────────────────────────────────────────────────
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Source root = one level up from DS-MTFNet (i.e. Image_Classifier/)
SOURCE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Split name mapping:  source folder -> target folder
SPLIT_MAP = {
    "training":   "train",
    "validation": "val",
    "testing":    "test",
}

# For each origin folder in source, map to target origin name
ORIGIN_MAP = {
    "AI Images":      "ai",
    "Non-AI Images":  "real",
    "AI-edited":      "ai_edited",
}

# For each (origin_key, source_content_subfolder) -> (target_content, target_animal_sub or None)
#   target_animal_sub is the sub-subfolder name under animal/ (lowercase), or None if not animal
CONTENT_MAP = {
    # AI Images subcategories
    "AI Images": {
        "AI_faces_images":    ("face",   None),
        "ai_generated_Human": ("human",  None),
        "AI_faces_images":    ("face",   None),
        # Animal mapped per sub-sub folder below
        "AI_animal_images":   ("animal", "SUBDIR"),  # SUBDIR = iterate sub-subfolders
    },
    # Non-AI Images subcategories
    "Non-AI Images": {
        "real_faces_images":  ("face",   None),
        "real_human_images":  ("human",  None),
        "real_animal_images": ("animal", "SUBDIR"),
    },
    # AI-edited subcategories
    "AI-edited": {
        "AI_face_edited":     ("face",   None),
        "AI_Human_edited":    ("human",  None),
        "AI_animal_edited":   ("animal", "SUBDIR"),
    },
}

DATASET_ROOT = "dataset"   # destination inside DS-MTFNet/


def copy_images(src_dir: str, dst_dir: str, moved: list):
    """Copy all valid images from src_dir into dst_dir (flat, no sub-dirs)."""
    if not os.path.isdir(src_dir):
        return
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if os.path.splitext(fname)[1].lower() in VALID_EXTS:
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            # Avoid overwriting: append _1, _2 … if needed
            stem, ext = os.path.splitext(fname)
            counter = 1
            while os.path.exists(dst):
                dst = os.path.join(dst_dir, f"{stem}_{counter}{ext}")
                counter += 1
            shutil.copy2(src, dst)
            moved.append((src, dst))


def reorganize():
    moved   = []
    skipped = []

    for src_split, tgt_split in SPLIT_MAP.items():
        # Source: Image_Classifier/training|validation|testing
        split_path = os.path.join(SOURCE_ROOT, src_split)
        if not os.path.isdir(split_path):
            print(f"  [SKIP] Split folder not found: {split_path}")
            skipped.append(split_path)
            continue

        for src_origin, tgt_origin in ORIGIN_MAP.items():
            origin_path = os.path.join(split_path, src_origin)
            if not os.path.isdir(origin_path):
                continue

            content_rules = CONTENT_MAP.get(src_origin, {})

            for src_content_folder, (tgt_content, animal_mode) in content_rules.items():
                src_content_path = os.path.join(origin_path, src_content_folder)

                if animal_mode == "SUBDIR":
                    # Iterate sub-subfolders (CAT, DOG, ELEPHANT, HORSE, LION …)
                    if not os.path.isdir(src_content_path):
                        continue
                    for animal_sub in os.listdir(src_content_path):
                        animal_src = os.path.join(src_content_path, animal_sub)
                        if not os.path.isdir(animal_src):
                            continue
                        animal_name = animal_sub.lower()  # normalise case
                        dst = os.path.join(DATASET_ROOT, tgt_split,
                                           tgt_origin, tgt_content, animal_name)
                        copy_images(animal_src, dst, moved)
                else:
                    # Flat content folder (human / face)
                    dst = os.path.join(DATASET_ROOT, tgt_split, tgt_origin, tgt_content)
                    copy_images(src_content_path, dst, moved)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*54}")
    print(f"  Reorganization complete!")
    print(f"  Images copied  : {len(moved):,}")
    print(f"  Splits skipped : {len(skipped)}")
    print(f"{'='*54}")

    # Count per split/origin
    counts: dict[str, int] = {}
    for _, dst in moved:
        parts = dst.replace("\\", "/").split("/")
        try:
            key = f"{parts[-4]}/{parts[-3]}/{parts[-2]}"
        except IndexError:
            key = dst
        counts[key] = counts.get(key, 0) + 1

    print("\n  Images per category:")
    for k in sorted(counts):
        print(f"    {k:<40} {counts[k]:>4}")
    print()


if __name__ == "__main__":
    print("\nDS-MTFNet Dataset Reorganizer")
    print("Working directory:", os.getcwd())
    print("Dataset root     :", os.path.abspath(DATASET_ROOT))
    print()
    reorganize()
    print("Done! You can now run:  python train.py\n")
