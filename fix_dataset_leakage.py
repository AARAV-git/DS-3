"""
Fix Dataset Leakage and Remove Legacy Folder Redundancies in DS-2
"""
import os, hashlib

DATASET_ROOT = "dataset"
SPLITS = ["training", "validation", "testing"]

ORIGIN_DIRS = {
    "ai": ["ai", "AI Images"],
    "ai_edited": ["ai_edited", "AI-edited"],
    "real": ["real", "Non-AI Images"]
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def get_hash(fp):
    try:
        with open(fp, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def scan_files():
    files_by_split = {sp: [] for sp in SPLITS}
    for sp in SPLITS:
        sp_path = os.path.join(DATASET_ROOT, sp)
        if not os.path.isdir(sp_path): continue
        for root, dirs, files in os.walk(sp_path):
            for f in files:
                if os.path.splitext(f)[1].lower() in VALID_EXTS:
                    fp = os.path.join(root, f)
                    h = get_hash(fp)
                    if h:
                        files_by_split[sp].append((fp, h))
    return files_by_split

def main():
    print("=" * 70)
    print("  Fixing Dataset Source Leakage across DS-2 Splits")
    print("=" * 70)

    files_by_split = scan_files()
    train_hashes = set(h for fp, h in files_by_split["training"])
    print(f"  Training Set Total Files Scanned : {len(files_by_split['training'])}")
    print(f"  Training Set Unique Hashes       : {len(train_hashes)}")

    removed_val = 0
    removed_test = 0

    # Purge duplicates in validation
    for fp, h in files_by_split["validation"]:
        if h in train_hashes:
            try:
                os.remove(fp)
                removed_val += 1
                print(f"  [REMOVED VAL LEAK] {fp}")
            except Exception as e:
                print(f"  [ERROR REMOVING] {fp}: {e}")

    # Purge duplicates in testing
    for fp, h in files_by_split["testing"]:
        if h in train_hashes:
            try:
                os.remove(fp)
                removed_test += 1
                print(f"  [REMOVED TEST LEAK] {fp}")
            except Exception as e:
                print(f"  [ERROR REMOVING] {fp}: {e}")

    print("\n" + "=" * 70)
    print(f"  LEAKAGE PURGE COMPLETE:")
    print(f"  - Validation Leaked Files Removed : {removed_val}")
    print(f"  - Testing Leaked Files Removed    : {removed_test}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
