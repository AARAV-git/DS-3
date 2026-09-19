"""
DS-MTFNet ─ Single-Image Inference CLI
========================================
Run from the DS-MTFNet root directory:
    python predict.py path/to/image.jpg
    python predict.py path/to/image.jpg --gradcam
"""

import argparse
import os
import sys

import torch
from PIL import Image

from src.dataset import get_fft, get_transforms
from src.model import DSMTFNet

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT    = "checkpoints/best_model.pth"
ORIGIN_NAMES  = ["ai", "real", "ai_edited"]
CONTENT_NAMES = ["human", "face", "animal"]


def predict(image_path: str, gradcam: bool = False):
    # ── Sanity checks ─────────────────────────────────────
    if not os.path.isfile(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)
    if not os.path.isfile(CHECKPOINT):
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT}\n"
              "        Please train the model first: python train.py")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────
    model = DSMTFNet().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
    model.eval()

    # ── Preprocess ────────────────────────────────────────
    img       = Image.open(image_path).convert("RGB")
    transform = get_transforms(train=False)
    rgb = transform(img).unsqueeze(0).to(DEVICE)
    fft = get_fft(img).unsqueeze(0).to(DEVICE)

    # ── Inference ─────────────────────────────────────────
    with torch.no_grad():
        o_out, c_out = model(rgb, fft)

    origin_probs  = torch.softmax(o_out, dim=1)[0]
    content_probs = torch.softmax(c_out, dim=1)[0]

    origin_pred   = ORIGIN_NAMES[origin_probs.argmax().item()]
    content_pred  = CONTENT_NAMES[content_probs.argmax().item()]

    # ── Results ───────────────────────────────────────────
    print(f"\n{'='*48}")
    print(f"  Image  : {os.path.basename(image_path)}")
    print(f"{'='*48}")
    print(f"  Authenticity  : {origin_pred.upper()}")
    for name, prob in zip(ORIGIN_NAMES, origin_probs.tolist()):
        bar = "█" * int(prob * 30)
        print(f"    {name:<12} {prob:.4f}  {bar}")

    print(f"\n  Content       : {content_pred.upper()}")
    for name, prob in zip(CONTENT_NAMES, content_probs.tolist()):
        bar = "█" * int(prob * 30)
        print(f"    {name:<12} {prob:.4f}  {bar}")
    print(f"{'='*48}\n")

    # ── Optional Grad-CAM ─────────────────────────────────
    if gradcam:
        from gradcam import run_gradcam
        out_path = os.path.join("outputs", os.path.splitext(
                   os.path.basename(image_path))[0] + "_heatmap.jpg")
        print("Generating Grad-CAM heatmap …")
        run_gradcam(image_path, head="origin", output_path=out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DS-MTFNet Single-Image Predictor")
    parser.add_argument("image",     type=str, help="Path to input image")
    parser.add_argument("--gradcam", action="store_true",
                        help="Also save a Grad-CAM heatmap to outputs/")
    args = parser.parse_args()
    predict(args.image, gradcam=args.gradcam)
