"""
DS-MTFNet v5.0 — FastAPI Inference Server
===========================================
Connects DS-MTFNet v5.0 (Hierarchical Head + 560x560 + 10-Feature Forensics)
with demo.html for interactive testing.

Start the server:
    uvicorn api:app --reload --port 8000

Open in browser:
    http://127.0.0.1:8000/
"""

import base64
import io
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

# Ensure local imports work properly
sys.path.insert(0, os.path.abspath("."))

from src.dataset import get_transforms, extract_forensic_vector
from src.model import DSMTFNetV4, HierarchicalOriginHead

# ──────────────────────────────────────────────────────────────────────────────
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
ORIGIN_CLASSES  = ["ai", "ai_edited", "real"]       # 0: ai, 1: ai_edited, 2: real
CONTENT_CLASSES = ["human", "face", "animal"]       # 0: human, 1: face, 2: animal
ANIMAL_CLASSES  = ["cat", "dog", "elephant", "horse", "lion"]
MAX_FILE_MB     = 10

CHECKPOINT_CANDIDATES = [
    "checkpoints/best_v5_model.pth",
    "checkpoints/latest_v5_model.pth",
    "checkpoints/best_v4_model.pth",
    "checkpoints/best_model.pth",
]

def find_checkpoint() -> str:
    for path in CHECKPOINT_CANDIDATES:
        if os.path.exists(path):
            return path
    if os.path.isdir("checkpoints"):
        pths = [os.path.join("checkpoints", f) for f in os.listdir("checkpoints") if f.endswith(".pth")]
        if pths:
            pths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return pths[0]
    raise FileNotFoundError("No model checkpoint found in checkpoints/")

CHECKPOINT_PATH = find_checkpoint()

# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DS-MTFNet v5.0 Inference API",
    description="Dual-Stream Multi-Task Network v5.0 for AI Image Forensics",
    version="5.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model singleton
_model     = None
_transform = get_transforms(train=False)

def get_model() -> nn.Module:
    global _model, CHECKPOINT_PATH
    if _model is None:
        CHECKPOINT_PATH = find_checkpoint()
        print(f"[API] Loading DS-MTFNet v5.0 checkpoint: {CHECKPOINT_PATH} on {DEVICE}...")
        model = DSMTFNetV4(
            dinov2_model_name="dinov2_vitb14",
            num_unfreeze_blocks=4,
            dropout=0.45
        ).to(DEVICE)
        
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        _model = model
        print(f"[API] Successfully loaded checkpoint '{CHECKPOINT_PATH}'")
    return _model


def compute_feature_heatmap(model: nn.Module, x_rgb: torch.Tensor, x_forensic: torch.Tensor) -> np.ndarray:
    """Extract spatial patch feature heatmap from DINOv2 last transformer block."""
    activations = {}
    target_layer = model.rgb_branch.blocks[-1]

    def fwd_hook(m, i, o):
        activations['a'] = o

    h = target_layer.register_forward_hook(fwd_hook)
    with torch.no_grad():
        _ = model(x_rgb, x_forensic)
    h.remove()

    # Spatial patch tokens (skip CLS token at index 0)
    act = activations['a'][:, 1:, :]           # (1, 1600, 768)
    cam = act.norm(dim=-1).squeeze(0)          # L2 norm across channels -> (1600,)

    grid_size = int(np.sqrt(cam.shape[0]))    # 40
    cam_2d    = cam.reshape(1, 1, grid_size, grid_size)
    cam_2d    = F.interpolate(cam_2d, size=(224, 224), mode="bilinear", align_corners=False)
    cam_np    = cam_2d.squeeze().cpu().numpy()
    
    # Normalize [0, 1]
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
    return cam_np


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["UI"])
def root():
    demo_path = os.path.join(os.path.dirname(__file__), "demo.html")
    if os.path.exists(demo_path):
        return FileResponse(demo_path, media_type="text/html")
    return {"status": "ok", "device": DEVICE, "checkpoint": CHECKPOINT_PATH}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "device": DEVICE,
        "checkpoint": CHECKPOINT_PATH,
        "model": "DS-MTFNet v5.0 (Hierarchical BCE Head)"
    }


@app.get("/demo", tags=["UI"])
def demo():
    demo_path = os.path.join(os.path.dirname(__file__), "demo.html")
    if not os.path.exists(demo_path):
        raise HTTPException(status_code=404, detail="demo.html not found.")
    return FileResponse(demo_path, media_type="text/html")


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_FILE_MB} MB).")
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    t_start = time.perf_counter()
    x_rgb      = _transform(img).unsqueeze(0).to(DEVICE)
    x_forensic = extract_forensic_vector(img).unsqueeze(0).to(DEVICE)

    model = get_model()
    with torch.no_grad():
        (z_ai, z_real), c_logits, anim_logits = model(x_rgb, x_forensic)
        origin_probs_tensor  = HierarchicalOriginHead.to_probs(z_ai, z_real)[0]
        content_probs_tensor = torch.softmax(c_logits, dim=1)[0]
        animal_probs_tensor  = torch.softmax(anim_logits, dim=1)[0]

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    origin_probs  = {name: round(p.item(), 4) for name, p in zip(ORIGIN_CLASSES, origin_probs_tensor)}
    content_probs = {name: round(p.item(), 4) for name, p in zip(CONTENT_CLASSES, content_probs_tensor)}

    origin_pred  = ORIGIN_CLASSES[int(origin_probs_tensor.argmax().item())]
    content_pred = CONTENT_CLASSES[int(content_probs_tensor.argmax().item())]
    animal_pred  = ANIMAL_CLASSES[int(animal_probs_tensor.argmax().item())]

    res = {
        "prediction":   {"origin": origin_pred, "content": content_pred},
        "origin_probs": origin_probs,
        "content_probs":content_probs,
        "inference_ms": round(elapsed_ms, 2),
        "filename":     file.filename,
    }
    if content_pred == "animal":
        res["prediction"]["animal_subclass"] = animal_pred
    return JSONResponse(res)


@app.post("/predict_with_cam", tags=["Inference"])
async def predict_with_cam(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_FILE_MB} MB).")
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    t_start = time.perf_counter()
    x_rgb      = _transform(img).unsqueeze(0).to(DEVICE)
    x_forensic = extract_forensic_vector(img).unsqueeze(0).to(DEVICE)

    model = get_model()
    cam_heatmap = compute_feature_heatmap(model, x_rgb, x_forensic)

    with torch.no_grad():
        (z_ai, z_real), c_logits, anim_logits = model(x_rgb, x_forensic)
        origin_probs_tensor  = HierarchicalOriginHead.to_probs(z_ai, z_real)[0]
        content_probs_tensor = torch.softmax(c_logits, dim=1)[0]
        animal_probs_tensor  = torch.softmax(anim_logits, dim=1)[0]

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    origin_probs  = {name: round(p.item(), 4) for name, p in zip(ORIGIN_CLASSES, origin_probs_tensor)}
    content_probs = {name: round(p.item(), 4) for name, p in zip(CONTENT_CLASSES, content_probs_tensor)}

    origin_pred  = ORIGIN_CLASSES[int(origin_probs_tensor.argmax().item())]
    content_pred = CONTENT_CLASSES[int(content_probs_tensor.argmax().item())]
    animal_pred  = ANIMAL_CLASSES[int(animal_probs_tensor.argmax().item())]

    # Overlay heatmap on image resized to 224x224 for UI display
    img_cv  = np.array(img.resize((224, 224)))[:, :, ::-1].copy()
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.55, heatmap, 0.45, 0)
    _, buf  = cv2.imencode(".jpg", overlay)
    cam_b64 = base64.b64encode(buf.tobytes()).decode()

    # Original base64
    orig_buf = io.BytesIO()
    img.resize((224, 224)).save(orig_buf, format="JPEG")
    orig_b64 = base64.b64encode(orig_buf.getvalue()).decode()

    res = {
        "prediction":     {"origin": origin_pred, "content": content_pred},
        "origin_probs":   origin_probs,
        "content_probs":  content_probs,
        "inference_ms":   round(elapsed_ms, 2),
        "filename":       file.filename,
        "heatmap_base64": cam_b64,
        "image_base64":   orig_b64,
    }
    if content_pred == "animal":
        res["prediction"]["animal_subclass"] = animal_pred
    return JSONResponse(res)
