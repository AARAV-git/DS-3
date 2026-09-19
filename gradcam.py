"""
DS-MTFNet ─ Grad-CAM Explainability
======================================
Generates a heatmap overlay showing which regions influenced
the model's authenticity (origin) prediction.

Usage:
    python gradcam.py path/to/image.jpg
    python gradcam.py path/to/image.jpg --head content --output outputs/cam.jpg
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import get_fft, get_transforms
from src.model import DSMTFNet

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "checkpoints/best_model.pth"

ORIGIN_NAMES  = ["ai", "real", "ai_edited"]
CONTENT_NAMES = ["human", "face", "animal"]


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    Hooks into the last convolutional block of the RGB stream (EfficientNet-B3).
    """

    def __init__(self, model: DSMTFNet):
        self.model       = model
        self.activations = None
        self.gradients   = None

        # Target: last MBConv block inside EfficientNet features
        target_layer = list(model.rgb_stream.children())[0][-1]

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(
        self,
        rgb: torch.Tensor,
        fft: torch.Tensor,
        head: str = "origin",
        class_idx: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Returns:
            cam       : H×W numpy array in [0, 1]
            class_idx : predicted (or forced) class index
            confidence: softmax confidence for that class
        """
        self.model.zero_grad()
        o_out, c_out = self.model(rgb, fft)
        logits = o_out if head == "origin" else c_out

        probs = torch.softmax(logits, dim=1)[0]
        if class_idx is None:
            class_idx = probs.argmax().item()
        confidence = probs[class_idx].item()

        logits[0, class_idx].backward()

        # Global average-pool gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)   # (B, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, h, w)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, confidence


def overlay_heatmap(image_path: str, cam: np.ndarray) -> np.ndarray:
    """Blend original image with Jet heatmap and return BGR overlay."""
    img_cv  = cv2.imread(image_path)
    img_cv  = cv2.resize(img_cv, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_cv, 0.55, heatmap, 0.45, 0)


def run_gradcam(image_path: str, head: str = "origin", output_path: str = "outputs/heatmap.jpg"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    transform = get_transforms(train=False)

    rgb = transform(img).unsqueeze(0).to(DEVICE)
    fft = get_fft(img).unsqueeze(0).to(DEVICE)

    model = DSMTFNet().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
    model.eval()

    gcam = GradCAM(model)
    cam, cls_idx, conf = gcam.generate(rgb, fft, head=head)

    label_names = ORIGIN_NAMES if head == "origin" else CONTENT_NAMES
    print(f"  Predicted class : {label_names[cls_idx]}  (confidence: {conf:.2%})")

    overlay = overlay_heatmap(image_path, cam)
    cv2.imwrite(output_path, overlay)
    print(f"  Heatmap saved   : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DS-MTFNet Grad-CAM Visualizer")
    parser.add_argument("image",  type=str, help="Path to input image")
    parser.add_argument("--head", type=str, default="origin",
                        choices=["origin", "content"],
                        help="Which task head to visualize (default: origin)")
    parser.add_argument("--output", type=str, default="outputs/heatmap.jpg",
                        help="Output path for heatmap image")
    args = parser.parse_args()
    run_gradcam(args.image, head=args.head, output_path=args.output)
