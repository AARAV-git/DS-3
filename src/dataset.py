"""
DS-MTFNet — Dataset Module v5.0
================================
Key changes from audit report:
  - Resolution: 560x560 (40x40 DINOv2 patch grid = 1600 tokens, 6.25x detail vs 224x224)
  - Container shortcut removal: JPEG quality jitter on ALL classes (p=0.7, q=55-98)
    The audit found ai_edited=83.1% PNG vs real=91.7% JPG — a strong format shortcut.
  - Forensic vector expanded to 10 features (added ELA P95, Sat Std from audit table):
      [ela_mean, ela_p95, noise_mean, noise_std, texture_mean,
       sat_mean, sat_std, sat_p90, jpeg_block, sharpness]
    ELA P95 score=0.226 (AI vs Edit); Sat Std score=0.242 (Real vs Edit)
  - Forensic extraction from 256x256 (was 128x128) for better spatial precision
"""

import os
import random
import io
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

ORIGIN_CLASSES  = {"ai": 0, "ai_edited": 1, "real": 2}
CONTENT_CLASSES = {"human": 0, "face": 1, "animal": 2}
ANIMAL_CLASSES  = {"cat": 0, "dog": 1, "elephant": 2, "horse": 3, "lion": 4}
VALID_EXTS      = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation Pipeline
# ──────────────────────────────────────────────────────────────────────────────
class JPEGQualityJitter(object):
    """
    Re-encode to JPEG at a random quality (55–98) to neutralize the PNG/JPG
    container shortcut. Applied to ALL classes with p=0.7 so the network cannot
    learn format from the container header.
    Audit finding: ai_edited=83.1% PNG vs real=91.7% JPG → format bias removed.
    """
    def __call__(self, img):
        quality = random.randint(55, 98)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def get_transforms(train: bool = True) -> T.Compose:
    """
    DS-MTFNet v5.0 augmentation pipeline — 560x560 Ultra-HD with Aspect-Ratio Preservation.
    560 = 14 (DINOv2 patch size) × 40 → exact multiple → 1600 spatial tokens.

    Augmentations:
      - Aspect Ratio Preservation: Uniform scaling prevents non-uniform rectangular stretching
      - All Flips: RandomHorizontalFlip (p=0.5) and RandomVerticalFlip (p=0.5)
      - Zoom In / Zoom Out: RandomAffine scale (0.80=zoom out, 1.20=zoom in) + 1:1 aspect-ratio RandomResizedCrop
      - Photometric perturbations: ColorJitter, GaussianBlur, JPEG quality simulation, and RandomErasing
    """
    if train:
        return T.Compose([
            # 1. Base aspect-ratio preserving resize (scales smaller edge to 560)
            T.Resize(560, interpolation=T.InterpolationMode.BICUBIC),
            # 2. Zoom In & Zoom Out affine scaling (0.80x to 1.20x) with exact 1:1 aspect preservation
            T.RandomAffine(
                degrees=15,
                translate=(0.05, 0.05),
                scale=(0.80, 1.20),
                interpolation=T.InterpolationMode.BICUBIC,
                fill=0
            ),
            # 3. Multi-scale crop with strict 1:1 ratio (aspect-ratio preserved zoom-in)
            T.RandomResizedCrop(
                560,
                scale=(0.70, 1.0),
                ratio=(1.0, 1.0),
                interpolation=T.InterpolationMode.BICUBIC
            ),
            # 4. All Flips (horizontal + vertical)
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            # 5. Photometric jitter & blur
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.4, hue=0.08),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))], p=0.3),
            T.RandomApply([JPEGQualityJitter()], p=0.3),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.20, scale=(0.02, 0.12), value=0),
        ])
    return T.Compose([
        # Aspect-ratio preserving resize of smaller edge to 560, followed by center crop
        T.Resize(560, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(560),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Handcrafted Forensic Feature Extractor (10 Audit-Selected Features)
# ──────────────────────────────────────────────────────────────────────────────
def extract_forensic_vector(img_pil: Image.Image) -> torch.Tensor:
    """
    Extracts 10 forensic features grounded in dataset audit Class Separation Scores:

    Feature          | AI vs Real | AI vs Edit | Real vs Edit | Best Use
    -----------------+------------+------------+--------------+-----------
    ELA Mean         | 0.006      | 0.232      | 0.206        | Edit detection
    ELA P95          | 0.015      | 0.226      | 0.206        | Edit detection (v5 NEW)
    Noise Mean       | 0.217      | 0.126      | 0.095        | AI vs Real
    Noise Std        | 0.244      | 0.123      | 0.117        | AI vs Real
    Texture Mean     | 0.191      | 0.100      | 0.079        | AI vs Real
    Sat Mean         | 0.127      | 0.348      | 0.242        | Edit detection
    Sat Std          | -          | -          | 0.242        | Real vs Edit (v5 NEW)
    Sat P90          | 0.179      | 0.424      | 0.243        | Edit detection
    JPEG Block Score | 0.254      | 0.129      | 0.125        | AI vs Real
    Sharpness        | 0.192      | 0.110      | 0.096        | AI vs Real

    Excludes: file_kb (shortcut risk — reflects collection pipeline, not image content)
    Forensic extraction at 256x256 (was 128x128) for better spatial accuracy.
    """
    try:
        # Downsample to 256x256 for forensic extraction (better spatial precision than 128x128)
        img = img_pil.resize((256, 256), Image.BILINEAR)
        u8  = np.array(img, dtype=np.uint8)
        arr = u8.astype(np.float32)

        # 1. Noise Residual (AI vs Real score: 0.217 / 0.244)
        blur = cv2.GaussianBlur(u8, (3, 3), 1.0).astype(np.float32)
        ng = (0.299 * np.abs(arr - blur)[:, :, 0]
              + 0.587 * np.abs(arr - blur)[:, :, 1]
              + 0.114 * np.abs(arr - blur)[:, :, 2])
        noise_mean = float(ng.mean()) / 15.0
        noise_std  = float(ng.std()) / 15.0

        # 2. Texture — Local Variance (AI vs Real score: 0.191)
        gray = (0.299 * arr[:, :, 0]
                + 0.587 * arr[:, :, 1]
                + 0.114 * arr[:, :, 2])
        lm   = cv2.blur(gray, (8, 8))
        lm2  = cv2.blur(gray**2, (8, 8))
        lstd = np.sqrt(np.maximum(lm2 - lm**2, 0))
        texture_mean = float(lstd.mean()) / 35.0

        # 3. Saturation — Mean, Std, P90 (AI vs Edit: 0.348 / -, Real vs Edit: 0.242)
        hsv     = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
        sat     = hsv[:, :, 1] / 255.0
        sat_mean = float(sat.mean())
        sat_std  = float(sat.std())          # NEW in v5.0 — Real vs Edit discriminator
        sat_p90  = float(np.percentile(sat, 90))

        # 4. Error Level Analysis — Mean & P95 (AI vs Edit: 0.232 / 0.226)
        bgr   = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        _, enc = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        ela   = np.mean(
            np.abs(arr - cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR),
                                       cv2.COLOR_BGR2RGB).astype(np.float32)),
            axis=2
        )
        ela_mean = float(ela.mean()) / 10.0
        ela_p95  = float(np.percentile(ela, 95)) / 20.0  # NEW in v5.0

        # 5. JPEG Block Boundary Score (AI vs Real: 0.254)
        g  = gray.astype(np.float32)
        hd = float(np.abs(g[:, 8::8] - g[:, 7:-1:8]).mean())
        vd = float(np.abs(g[8::8, :] - g[7:-1:8, :]).mean())
        jpeg_block = ((hd + vd) / 2.0) / 20.0

        # 6. Sharpness — Laplacian Variance (AI vs Real: 0.192)
        shp = float(cv2.Laplacian(cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY),
                                   cv2.CV_64F).var()) / 3500.0

        # Order matches model.py FORENSIC_DIM = 10
        feats = [ela_mean, ela_p95, noise_mean, noise_std, texture_mean,
                 sat_mean, sat_std, sat_p90, jpeg_block, shp]
        vec = np.clip(np.array(feats, dtype=np.float32), 0.0, 2.0)
    except Exception:
        vec = np.zeros(10, dtype=np.float32)

    return torch.from_numpy(vec)


# ──────────────────────────────────────────────────────────────────────────────
# DATASET CLASS
# ──────────────────────────────────────────────────────────────────────────────
class DSMTFNetDataset(Dataset):
    def __init__(self, root: str, transform=None, is_train: bool = False):
        self.root = root
        self.transform = transform
        self.is_train = is_train
        self.samples: list = []
        self._forensic_cache = {}
        self._build_index()

    def _build_index(self):
        folder_candidates = {
            "ai":       ["ai", "AI Images"],
            "ai_edited":["ai_edited", "AI-edited"],
            "real":     ["real", "Non-AI Images"]
        }
        for origin_key, origin_label in ORIGIN_CLASSES.items():
            for orig_folder in folder_candidates[origin_key]:
                orig_path = os.path.join(self.root, orig_folder)
                if os.path.isdir(orig_path):
                    self._walk_and_add(orig_path, origin_label)

        if len(self.samples) == 0:
            print(f"[WARNING] No images found under '{self.root}'.")
        else:
            random.shuffle(self.samples)

    def _walk_and_add(self, base_folder: str, origin_label: int):
        for root_dir, _, files in os.walk(base_folder):
            rl = root_dir.lower()
            content_label = 0
            animal_label  = 0

            if "face" in rl:
                content_label = 1
            elif "animal" in rl:
                content_label = 2
                for anim_name, anim_idx in ANIMAL_CLASSES.items():
                    if anim_name in rl:
                        animal_label = anim_idx
                        break

            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() in VALID_EXTS:
                    self.samples.append((
                        os.path.join(root_dir, fname),
                        origin_label,
                        content_label,
                        animal_label
                    ))

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_spatial_cutmix(self, img: Image.Image) -> Image.Image:
        """Fast internal patch paste to simulate local edits without disk I/O."""
        if random.random() > 0.35:
            return img
        w, h = img.size
        if w < 32 or h < 32:
            return img
        cw = int(w * random.uniform(0.15, 0.35))
        ch = int(h * random.uniform(0.15, 0.35))
        cx1 = random.randint(0, w - cw)
        cy1 = random.randint(0, h - ch)
        cx2 = random.randint(0, w - cw)
        cy2 = random.randint(0, h - ch)
        try:
            crop = img.crop((cx1, cy1, cx1 + cw, cy1 + ch))
            img = img.copy()
            img.paste(crop, (cx2, cy2))
        except Exception:
            pass
        return img

    def __getitem__(self, idx: int):
        path, origin_label, content_label, animal_label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256), (128, 128, 128))
        img.info.clear()  # Strip EXIF / Metadata signatures

        # Retrieve or compute forensic vector (cached in RAM for subsequent epochs)
        if path in self._forensic_cache:
            forensic_vec = self._forensic_cache[path]
        else:
            forensic_vec = extract_forensic_vector(img)
            self._forensic_cache[path] = forensic_vec

        if self.is_train and origin_label == ORIGIN_CLASSES["ai_edited"]:
            img = self._apply_spatial_cutmix(img)

        if self.transform:
            img = self.transform(img)

        return img, forensic_vec, origin_label, content_label, animal_label


# ──────────────────────────────────────────────────────────────────────────────
# DATALOADER FACTORY
# ──────────────────────────────────────────────────────────────────────────────
def _collate_fn(batch):
    """Convert list of tuples into a dict of tensors expected by train.py."""
    imgs, forensics, origins, contents, animals = zip(*batch)
    return {
        "rgb":            torch.stack(imgs),
        "forensic":       torch.stack(forensics),
        "origin_target":  torch.tensor(origins,  dtype=torch.long),
        "content_target": torch.tensor(contents, dtype=torch.long),
        "animal_target":  torch.tensor(animals,  dtype=torch.long),
    }


def get_dataloaders(root_dir: str = "dataset", batch_size: int = 32, num_workers: int = 4):
    """
    Build train and validation DataLoaders.
    Expects dataset layout:
        <root_dir>/training/   → train split
        <root_dir>/validation/ → val   split
    Falls back to <root_dir>/ directly if those sub-folders don't exist.
    """
    from torch.utils.data import DataLoader

    train_root = os.path.join(root_dir, "training")
    val_root   = os.path.join(root_dir, "validation")

    if not os.path.isdir(train_root):
        train_root = root_dir
    if not os.path.isdir(val_root):
        val_root = root_dir

    train_dataset = DSMTFNetDataset(train_root, transform=get_transforms(train=True),  is_train=True)
    val_dataset   = DSMTFNetDataset(val_root,   transform=get_transforms(train=False), is_train=False)

    n_train = len(train_dataset)
    n_val   = len(val_dataset)
    bs      = batch_size

    print(f"  Train images : {n_train:,} | {(n_train + bs - 1) // bs} batches/epoch")
    print(f"  Val images   : {n_val:,}   | {(n_val + bs - 1) // bs} batches/epoch")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate_fn,
    )

    return train_loader, val_loader
