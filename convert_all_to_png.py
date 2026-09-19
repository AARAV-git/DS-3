"""
DS-MTFNet — Convert all dataset images to .png while preserving exact aspect ratio.

This script:
1. Recursively scans dataset/ (training, validation, testing).
2. Converts every non-PNG image (.jpg, .jpeg, .webp, .bmp, .tiff, etc.) to .png in the SAME folder.
3. Preserves original image dimensions (W, H) exactly, so the aspect ratio is 100% identical.
4. Converts color mode cleanly to RGB (handling RGBA, CMYK, Palette).
5. Verifies the new .png file with Image.verify() before unlinking the original file.
6. Files already .png are left untouched.
7. Logs comprehensive before/after statistics.
"""

import os
import sys
import time
from pathlib import Path
from collections import Counter
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor

CONVERTIBLE_EXTS = {".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".jfif"}

def convert_single_image(src_path: Path, keep_originals: bool = False):
    target_path = src_path.with_suffix(".png")
    if target_path.exists() and target_path != src_path:
        # If target already exists, give it a unique suffix or skip if identical
        base = src_path.stem
        target_path = src_path.parent / f"{base}_converted.png"
    
    try:
        with Image.open(src_path) as im:
            orig_size = im.size  # (W, H)
            orig_ratio = orig_size[0] / orig_size[1]
            rgb_im = im.convert("RGB")
            rgb_im.save(target_path, "PNG", optimize=False)
            new_size = rgb_im.size
            new_ratio = new_size[0] / new_size[1]
            assert orig_size == new_size, f"Size changed: {orig_size} -> {new_size}"
        
        # Verify the saved PNG is valid
        with Image.open(target_path) as check:
            check.verify()

        if not keep_originals and src_path != target_path:
            src_path.unlink()
        
        return True, str(src_path), None
    except Exception as e:
        if target_path.exists() and src_path != target_path:
            try:
                target_path.unlink()
            except Exception:
                pass
        return False, str(src_path), str(e)


def process_dataset(root_dir: str = "dataset", keep_originals: bool = False, max_workers: int = 8):
    root = Path(root_dir)
    if not root.is_dir():
        print(f"[ERROR] Directory not found: {root_dir}")
        return

    print("=" * 80)
    print(f"  Standardizing Dataset Images to .PNG (Aspect-Ratio Preserving)")
    print(f"  Target Root Directory: {root.resolve()}")
    print("=" * 80)

    # Scan all files
    all_files = [p for p in root.rglob("*") if p.is_file()]
    ext_counter = Counter(p.suffix.lower() for p in all_files)
    
    print("\nCurrent Extension Distribution:")
    for ext, count in sorted(ext_counter.items(), key=lambda x: -x[1]):
        print(f"  {ext:10s} : {count:6d} files")
    
    to_convert = [p for p in all_files if p.suffix.lower() in CONVERTIBLE_EXTS]
    already_png = [p for p in all_files if p.suffix.lower() == ".png"]
    
    print(f"\nTotal files found       : {len(all_files):,}")
    print(f"Already in .png format  : {len(already_png):,}")
    print(f"Files needing conversion: {len(to_convert):,}")
    
    if not to_convert:
        print("\nAll dataset files are already in .png format! Nothing to convert.")
        return

    print(f"\nStarting conversion using {max_workers} worker threads...")
    start_time = time.time()
    
    converted_count = 0
    failed_count = 0
    failed_items = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(convert_single_image, p, keep_originals) for p in to_convert]
        for i, fut in enumerate(futures):
            success, path, err = fut.result()
            if success:
                converted_count += 1
            else:
                failed_count += 1
                failed_items.append((path, err))
            
            if (i + 1) % 500 == 0 or (i + 1) == len(to_convert):
                elapsed = time.time() - start_time
                print(f"  Progress: {i+1:5d}/{len(to_convert)} converted ({converted_count} success, {failed_count} failed) | Elapsed: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"  Conversion Finished in {elapsed:.2f} seconds")
    print(f"  Successfully converted : {converted_count}")
    print(f"  Failed                 : {failed_count}")
    if failed_items:
        print("  Errors:")
        for fp, err in failed_items[:10]:
            print(f"    {fp}: {err}")
    print("=" * 80)

    # Post-conversion verification
    post_files = [p for p in root.rglob("*") if p.is_file()]
    post_exts = Counter(p.suffix.lower() for p in post_files)
    print("\nPost-Conversion Extension Distribution:")
    for ext, count in sorted(post_exts.items(), key=lambda x: -x[1]):
        print(f"  {ext:10s} : {count:6d} files")
    print("=" * 80)


if __name__ == "__main__":
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    process_dataset(dataset_dir)
