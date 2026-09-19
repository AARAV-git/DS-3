"""
DS-MTFNet v5.0 — Video Inference Script
=========================================
Analyzes video files frame-by-frame using the trained DS-MTFNet v5.0 image model
and aggregates per-frame predictions to produce a video-level verdict:
    Origin  : AI Generated | AI Edited | Real Video
    Content : Human | Face | Animal

Strategy: Uniform frame sampling + forensic-weighted majority vote.
Each sampled frame is processed identically to a still image (same transforms,
same forensic extractor), making this a clean extension of the image model.

Usage:
    python video_infer.py --video path/to/clip.mp4
    python video_infer.py --video clip.mp4 --num_frames 32 --checkpoint checkpoints/best_v5_model.pth
    python video_infer.py --video clip.mp4 --output_json outputs/video_result.json --save_frames
"""

import sys
import os
import json
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from collections import Counter
import torchvision.transforms as T

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── DS-MTFNet v5.0 imports ───────────────────────────────────────────────────
from src.dataset import get_transforms, extract_forensic_vector
from src.model import DSMTFNetV4, HierarchicalOriginHead

# ── Labels ────────────────────────────────────────────────────────────────────
ORIGIN_NAMES  = {0: "AI Generated", 1: "AI Edited", 2: "Real Video"}
CONTENT_NAMES = {0: "Human", 1: "Face", 2: "Animal"}
ANIMAL_NAMES  = {0: "Cat", 1: "Dog", 2: "Elephant", 3: "Horse", 4: "Lion"}

# Confidence emoji for visual clarity in terminal
def conf_bar(p: float, width: int = 20) -> str:
    filled = int(p * width)
    return "[" + "#" * filled + "." * (width - filled) + f"] {p*100:.1f}%"


# ── Frame sampler ─────────────────────────────────────────────────────────────
def sample_frames(video_path: str, num_frames: int, strategy: str = "uniform"):
    """
    Samples frames from a video file using OpenCV.

    strategy:
      'uniform'   - evenly spaced across the full clip duration
      'dense'     - every 0.5s (audio-agnostic)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_s = total_frames / fps

    if strategy == "uniform":
        sample_idxs = [int(i * total_frames / num_frames) for i in range(num_frames)]
    else:  # dense: every 0.5s
        sample_idxs = [int(t * fps) for t in np.arange(0, duration_s, 0.5)]
        sample_idxs = sample_idxs[:num_frames]

    frames_pil = []
    frame_timestamps = []
    for idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if ok:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames_pil.append(Image.fromarray(frame_rgb))
            frame_timestamps.append(idx / fps)
    cap.release()

    if not frames_pil:
        raise RuntimeError(f"No readable frames from {video_path}")
    return frames_pil, frame_timestamps, {"total_frames": total_frames, "fps": fps, "duration_s": duration_s}


# ── Model loader ─────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device) -> DSMTFNetV4:
    print(f"\n[INFO] Loading DS-MTFNet v5.0 checkpoint: {checkpoint_path}")
    model = DSMTFNetV4(
        dinov2_model_name="dinov2_vitb14",
        num_unfreeze_blocks=4,
        dropout=0.45
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] Model loaded on {device}")
    return model


# ── Per-frame inference ───────────────────────────────────────────────────────
def infer_frame(frame_pil: Image.Image, model: DSMTFNetV4,
                transform: T.Compose, device: torch.device) -> dict:
    """Runs a single PIL frame through the full DS-MTFNet v5.0 pipeline."""
    # Extract forensic vector from raw frame (before visual transform)
    forensic_vec = extract_forensic_vector(frame_pil).unsqueeze(0).to(device)

    # Apply test-time transform (aspect-ratio preserving → 560x560)
    rgb_tensor = transform(frame_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        (z_ai, z_real), content_logits, animal_logits = model(rgb_tensor, forensic_vec)

    # Origin probabilities via hierarchical head
    origin_probs = HierarchicalOriginHead.to_probs(z_ai, z_real)[0].cpu()  # (3,)
    content_probs = torch.softmax(content_logits, dim=1)[0].cpu()           # (3,)
    animal_probs  = torch.softmax(animal_logits,  dim=1)[0].cpu()           # (5,)

    origin_pred  = origin_probs.argmax().item()
    content_pred = content_probs.argmax().item()
    animal_pred  = animal_probs.argmax().item()

    return {
        "origin_pred":    origin_pred,
        "origin_label":   ORIGIN_NAMES[origin_pred],
        "origin_conf":    float(origin_probs[origin_pred]),
        "origin_probs":   origin_probs.tolist(),
        "content_pred":   content_pred,
        "content_label":  CONTENT_NAMES[content_pred],
        "content_conf":   float(content_probs[content_pred]),
        "animal_pred":    animal_pred,
        "animal_label":   ANIMAL_NAMES[animal_pred],
        "animal_conf":    float(animal_probs[animal_pred]),
        "forensic_vec":   forensic_vec.cpu().squeeze(0).tolist(),
    }


# ── Aggregation ───────────────────────────────────────────────────────────────
def aggregate_predictions(frame_results: list) -> dict:
    """
    Aggregates per-frame results into a video-level verdict.
    Uses confidence-weighted voting for origin and majority vote for content.
    """
    # Confidence-weighted origin vote
    class_weights = [0.0, 0.0, 0.0]
    for r in frame_results:
        for c in range(3):
            class_weights[c] += r["origin_probs"][c]
    class_weights_norm = [w / len(frame_results) for w in class_weights]
    video_origin_pred = int(np.argmax(class_weights_norm))

    # Majority vote for content
    content_votes = Counter(r["content_pred"] for r in frame_results)
    video_content_pred = content_votes.most_common(1)[0][0]

    # Per-frame vote distribution (for transparency)
    origin_votes = Counter(r["origin_label"] for r in frame_results)
    content_vote_dist = Counter(r["content_label"] for r in frame_results)

    # Confidence stats
    origin_confs = [r["origin_conf"] for r in frame_results]

    return {
        "verdict": {
            "origin_class":   video_origin_pred,
            "origin_label":   ORIGIN_NAMES[video_origin_pred],
            "origin_avg_prob": class_weights_norm,
            "origin_confidence": class_weights_norm[video_origin_pred],
            "content_class":  video_content_pred,
            "content_label":  CONTENT_NAMES[video_content_pred],
        },
        "stats": {
            "frames_analyzed":    len(frame_results),
            "mean_frame_conf":    float(np.mean(origin_confs)),
            "min_frame_conf":     float(np.min(origin_confs)),
            "max_frame_conf":     float(np.max(origin_confs)),
            "std_frame_conf":     float(np.std(origin_confs)),
            "origin_vote_dist":   dict(origin_votes),
            "content_vote_dist":  dict(content_vote_dist),
        }
    }


# ── Terminal report printer ───────────────────────────────────────────────────
def print_report(video_path: str, video_info: dict, agg: dict, frame_results: list):
    sep = "=" * 70
    verdict  = agg["verdict"]
    stats    = agg["stats"]

    print(f"\n{sep}")
    print(f"  DS-MTFNet v5.0 -- Video Authenticity Analysis")
    print(f"{sep}")
    print(f"  Video File  : {os.path.basename(video_path)}")
    print(f"  Duration    : {video_info['duration_s']:.1f}s  |  FPS: {video_info['fps']:.1f}  |  Total Frames: {video_info['total_frames']}")
    print(f"  Frames Analyzed: {stats['frames_analyzed']}")
    print(f"{sep}")

    print(f"\n  VERDICT")
    print(f"  Origin  : [{verdict['origin_label']}]")
    probs = verdict["origin_avg_prob"]
    for i, name in ORIGIN_NAMES.items():
        marker = " <<" if i == verdict["origin_class"] else ""
        print(f"    {name:<16} {conf_bar(probs[i])}{marker}")

    print(f"\n  Content Type: [{verdict['content_label']}]")
    print(f"\n  Mean Frame Confidence: {stats['mean_frame_conf']*100:.1f}%  "
          f"(min {stats['min_frame_conf']*100:.1f}%  max {stats['max_frame_conf']*100:.1f}%)")

    print(f"\n  Per-Frame Vote Distribution (Origin):")
    for label, count in sorted(stats["origin_vote_dist"].items(), key=lambda x: -x[1]):
        pct = count / stats["frames_analyzed"] * 100
        print(f"    {label:<18}: {count:3d}/{stats['frames_analyzed']} frames  ({pct:.0f}%)")

    print(f"\n  Per-Frame Details:")
    print(f"  {'Frame':>6}  {'Origin Prediction':<18}  {'Conf':>7}  {'Content':<8}")
    print(f"  {'-'*6}  {'-'*18}  {'-'*7}  {'-'*8}")
    for i, r in enumerate(frame_results):
        print(f"  {i+1:>6}  {r['origin_label']:<18}  {r['origin_conf']*100:>6.1f}%  {r['content_label']:<8}")

    print(f"\n{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DS-MTFNet v5.0 Video Inference")
    p.add_argument("--video",       type=str, required=True,  help="Path to input video file (.mp4, .avi, .mov, etc.)")
    p.add_argument("--checkpoint",  type=str, default=None,   help="Path to model checkpoint (.pth). Auto-detected if omitted.")
    p.add_argument("--num_frames",  type=int, default=16,     help="Number of frames to sample (default: 16)")
    p.add_argument("--strategy",    type=str, default="uniform", choices=["uniform", "dense"],
                   help="Frame sampling strategy: 'uniform' (evenly spaced) or 'dense' (every 0.5s)")
    p.add_argument("--output_json", type=str, default=None,   help="Save full results to a JSON file")
    p.add_argument("--save_frames", action="store_true",      help="Save sampled frames as PNG files to outputs/video_frames/")
    return p.parse_args()


def find_checkpoint() -> str:
    candidates = [
        "checkpoints/best_v5_model.pth",
        "checkpoints/latest_v5_model.pth",
        "checkpoints/best_model.pth",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    if os.path.isdir("checkpoints"):
        pths = [os.path.join("checkpoints", f) for f in os.listdir("checkpoints") if f.endswith(".pth")]
        if pths:
            pths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return pths[0]
    raise FileNotFoundError("No checkpoint found in checkpoints/. Run training first: python train.py")


def main():
    args = parse_args()

    if not os.path.isfile(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    checkpoint = args.checkpoint or find_checkpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Sampling {args.num_frames} frames ({args.strategy}) from: {args.video}")

    # Load model
    model = load_model(checkpoint, device)
    transform = get_transforms(train=False)  # Val/test transform: aspect-ratio preserving 560x560

    # Sample frames
    frames_pil, timestamps, video_info = sample_frames(args.video, args.num_frames, args.strategy)
    print(f"[INFO] Sampled {len(frames_pil)} frames from {video_info['duration_s']:.1f}s video")

    # Save sampled frames if requested
    if args.save_frames:
        frame_dir = os.path.join("outputs", "video_frames")
        os.makedirs(frame_dir, exist_ok=True)
        for i, (f, ts) in enumerate(zip(frames_pil, timestamps)):
            fname = os.path.join(frame_dir, f"frame_{i+1:03d}_t{ts:.2f}s.png")
            f.save(fname)
        print(f"[INFO] Saved {len(frames_pil)} frames to {frame_dir}/")

    # Per-frame inference
    print(f"[INFO] Running inference on {len(frames_pil)} frames...")
    frame_results = []
    for i, (frame, ts) in enumerate(zip(frames_pil, timestamps)):
        result = infer_frame(frame, model, transform, device)
        result["frame_index"] = i + 1
        result["timestamp_s"] = ts
        frame_results.append(result)
        print(f"  Frame {i+1:3d}/{len(frames_pil)} @ {ts:.2f}s  ->  "
              f"{result['origin_label']} ({result['origin_conf']*100:.1f}%)")

    # Aggregate
    agg = aggregate_predictions(frame_results)

    # Print terminal report
    print_report(args.video, video_info, agg, frame_results)

    # Save JSON output
    output_path = args.output_json or os.path.join("outputs", "video_prediction.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    full_result = {
        "video_file":    args.video,
        "checkpoint":    checkpoint,
        "num_frames":    len(frame_results),
        "video_info":    video_info,
        "verdict":       agg["verdict"],
        "stats":         agg["stats"],
        "frame_details": frame_results,
    }
    with open(output_path, "w") as f:
        json.dump(full_result, f, indent=2)
    print(f"[INFO] Full results saved to: {output_path}")


if __name__ == "__main__":
    main()
