"""
DS-MTFNet v5.0 — Training Script
==================================
Key changes from v4.x:
  - HIERARCHICAL ORIGIN LOSS: two BCEWithLogitsLoss terms replace the 3-way FocalLoss
      loss_ai   : BCEWithLogitsLoss on P(is_AI)          (pos_weight from class counts)
      loss_real : BCEWithLogitsLoss on P(is_Real|Not-AI) (only on non-AI samples)
    Eliminates 3-way denominator competition that caused edit/real oscillation.
  - Forensic branch now uses 10 features (model.FORENSIC_DIM)
  - Resolution: 560x560 (1600 DINOv2 spatial tokens vs 256 at 224x224)
  - BATCH_SIZE=8 for 560x560 VRAM safety
  - Compute_metrics updated to use HierarchicalOriginHead.to_probs()
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.dataset import get_dataloaders
from src.model import DSMTFNetV4, HierarchicalOriginHead

import argparse

parser = argparse.ArgumentParser(description="DS-MTFNet v5.0 Training")
parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=21, help="Physical batch size (default: 21 -> fits in 8GB VRAM with zero paging)")
parser.add_argument("--accum_steps", type=int, default=2, help="Gradient accumulation steps (default: 2 -> effective batch 42, exactly 120 updates/epoch)")
parser.add_argument("--fresh", action="store_true", help="Start fresh training ignoring existing checkpoints")
parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
train_args, _ = parser.parse_known_args()

# ── Class indices (must match ORIGIN_CLASSES = {ai:0, ai_edited:1, real:2}) ─
AI_IDX, EDIT_IDX, REAL_IDX = 0, 1, 2

# ------------------------------------------------------------------------------
# HYPERPARAMETERS (DS-MTFNet v5.0 — 560x560 + Hierarchical Head)
# ------------------------------------------------------------------------------
BATCH_SIZE            = train_args.batch_size
ACCUM_STEPS           = max(1, train_args.accum_steps)
LR_HEAD               = 2e-4                   # Heads, FFT CNN, Forensic MLP, Fusion
LR_BACKBONE_PARTIAL   = 5e-6                   # Unfrozen DINOv2 blocks (forensic adaptation)
WEIGHT_DECAY          = 0.02                   # Strong regularization to combat overfitting
NUM_UNFREEZE_BLOCKS   = 4                      # Last 4 blocks of ViT-B/14
EPOCHS                = train_args.epochs
WARMUP_EPOCHS         = 3
PATIENCE              = train_args.patience
DEVICE                = "cuda" if torch.cuda.is_available() else "cpu"
DROPOUT               = 0.45                   # Fusion dropout
NUM_WORKERS           = 0                      # Windows: must be 0 (CUDA + multiprocessing)
CHECKPOINT_PATH       = "checkpoints/best_v5_model.pth"
HISTORY_PATH          = "logs/history_v5.json"

# Auxiliary loss weights
LAMBDA_ORIGIN  = 1.0
LAMBDA_CONTENT = 0.3
LAMBDA_ANIMAL  = 0.2


# ------------------------------------------------------------------------------
# HIERARCHICAL ORIGIN LOSS
# Based on class counts: ~1:1:1 ratio so pos_weights are symmetric
# ------------------------------------------------------------------------------
def compute_hierarchical_pos_weights(device):
    """
    Inverse-frequency positive weights for BCEWithLogitsLoss.
    With balanced classes (1680 each):
      - pos_weight_ai   = (n_edit + n_real) / n_ai = 2.0
      - pos_weight_real = n_edit / n_real           = 1.0
    Hardcoded from audit table; adjust if class counts change.
    """
    pos_weight_ai   = torch.tensor(2.0, device=device)   # 2 non-AI per 1 AI
    pos_weight_real = torch.tensor(1.0, device=device)   # 1 edit per 1 real
    return pos_weight_ai, pos_weight_real


def hierarchical_origin_loss(z_ai, z_real, y_origin, pos_weight_ai, pos_weight_real):
    """
    Two binary cross-entropy terms:
      1. loss_ai:   all samples — is this AI-generated from scratch?
      2. loss_real: only non-AI samples — is this untouched real vs AI-edited?
    Separating these two decisions eliminates the zero-sum 3-way softmax competition.
    """
    is_ai  = (y_origin == AI_IDX).float()
    loss_ai = F.binary_cross_entropy_with_logits(z_ai, is_ai, pos_weight=pos_weight_ai)

    not_ai_mask = (y_origin != AI_IDX)
    if not_ai_mask.any():
        is_real = (y_origin[not_ai_mask] == REAL_IDX).float()
        loss_real = F.binary_cross_entropy_with_logits(
            z_real[not_ai_mask], is_real, pos_weight=pos_weight_real
        )
    else:
        loss_real = torch.tensor(0.0, device=z_ai.device)

    return loss_ai + loss_real


# ------------------------------------------------------------------------------
# ACCURACY CALCULATOR (Hierarchical Head version)
# ------------------------------------------------------------------------------
def compute_metrics(z_ai, z_real, content_logits, animal_logits,
                    origin_targets, content_targets, animal_targets):
    with torch.no_grad():
        # Convert binary logits → 3-class probs → argmax for accuracy
        origin_probs = HierarchicalOriginHead.to_probs(z_ai, z_real)
        origin_preds  = origin_probs.argmax(dim=1)
        content_preds = content_logits.argmax(dim=1)

        origin_correct  = (origin_preds == origin_targets).sum().item()
        content_correct = (content_preds == content_targets).sum().item()

        # Animal accuracy only for animal-content rows
        animal_mask = (content_targets == 1)
        if animal_mask.sum().item() > 0:
            animal_preds   = animal_logits.argmax(dim=1)[animal_mask]
            animal_targ    = animal_targets[animal_mask]
            animal_correct = (animal_preds == animal_targ).sum().item()
            animal_total   = animal_mask.sum().item()
        else:
            animal_correct = 0
            animal_total   = 0

        # Per-class origin accuracy
        per_class_correct = [0, 0, 0]
        per_class_total   = [0, 0, 0]
        for c in range(3):
            mask = (origin_targets == c)
            per_class_correct[c] = (origin_preds[mask] == c).sum().item()
            per_class_total[c]   = mask.sum().item()

        # Coarse accuracy: AI vs Non-AI
        coarse_preds  = (origin_preds == AI_IDX).long()
        coarse_labels = (origin_targets == AI_IDX).long()
        coarse_correct = (coarse_preds == coarse_labels).sum().item()

    return {
        "origin_correct":    origin_correct,
        "content_correct":   content_correct,
        "animal_correct":    animal_correct,
        "animal_total":      animal_total,
        "coarse_correct":    coarse_correct,
        "total":             len(origin_targets),
        "per_class_correct": per_class_correct,
        "per_class_total":   per_class_total
    }


# ------------------------------------------------------------------------------
# MAIN TRAINING LOGIC
# ------------------------------------------------------------------------------
def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Speed: cuDNN benchmark + TF32 on Ampere Tensor Cores (RTX 3070 Ti)
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 105)
    print("  DS-MTFNet v5.0 — Hierarchical BCE Head + 560x560 Ultra-HD + 10-Feature Forensics")
    print("=" * 105)

    # Dataloaders
    train_loader, val_loader = get_dataloaders(
        root_dir="dataset",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Model
    print(f"\n[MODEL] Loading DINOv2 ViT-B/14 (dinov2_vitb14)...")
    model = DSMTFNetV4(
        dinov2_model_name="dinov2_vitb14",
        num_unfreeze_blocks=NUM_UNFREEZE_BLOCKS,
        dropout=DROPOUT
    ).to(DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}")
    print(f"  RGB Stream       : DINOv2 ViT-B/14 (560x560 -> 1600 tokens) [last 4 blocks @ LR=5e-6]")
    print(f"  FFT Stream       : 2D Magnitude Spectrum -> CNN (128-d)")
    print(f"  Forensic Stream  : 10-Feature Audit-Grounded MLP (64-d)")
    print(f"  Fusion           : MLP(576->512) shared embedding (Dropout=0.45)")
    print(f"  Origin Head      : Hierarchical BCE [AI vs Rest | Real vs Edit]")
    print(f"  Loss             : HierarchicalOriginLoss + FocalLoss(Content) + FocalLoss(Animal)")
    print(f"  Epochs / Patience: {EPOCHS} / {PATIENCE}")
    print(f"  Device           : {DEVICE.upper()}\n")

    # Optimizer: two groups (backbone at lower LR)
    head_params, backbone_params = model.get_param_groups()
    optimizer = torch.optim.AdamW([
        {"params": head_params,     "lr": LR_HEAD,             "weight_decay": WEIGHT_DECAY},
        {"params": backbone_params, "lr": LR_BACKBONE_PARTIAL, "weight_decay": WEIGHT_DECAY * 0.1},
    ])

    def lr_lambda(ep: int) -> float:
        if ep < WARMUP_EPOCHS:
            return 0.10 + 0.90 * ((ep + 1) / WARMUP_EPOCHS)
        progress = (ep - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss functions
    pos_weight_ai, pos_weight_real = compute_hierarchical_pos_weights(DEVICE)
    print(f"[LOSS] Hierarchical pos_weights: ai={pos_weight_ai.item():.2f}, real={pos_weight_real.item():.2f}")

    content_crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    animal_crit  = nn.CrossEntropyLoss(label_smoothing=0.05)

    # AMP — Automatic Mixed Precision (fp16 math on tensor cores, fp32 master weights)
    # Gives 2-3x speedup on RTX 3070 Ti with no accuracy loss
    use_amp = (DEVICE == "cuda")
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[SPEED] AMP (mixed precision): {'ENABLED (fp16)' if use_amp else 'DISABLED (cpu)'}")
    print(f"[SPEED] Physical batch: {BATCH_SIZE} | Accum: {ACCUM_STEPS} -> Effective batch: {BATCH_SIZE * ACCUM_STEPS}")
    print(f"[SPEED] Optimizer updates: {(len(train_loader) + ACCUM_STEPS - 1) // ACCUM_STEPS} steps/epoch")

    # Checkpoints
    LATEST_CHECKPOINT_PATH = "checkpoints/latest_v5_model.pth"
    
    start_epoch    = 1
    best_val_acc   = 0.0
    patience_count = 0
    history        = []

    # Auto-resume if latest checkpoint exists and not --fresh
    resume_path = None
    if not train_args.fresh:
        resume_path = LATEST_CHECKPOINT_PATH if os.path.exists(LATEST_CHECKPOINT_PATH) else (CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else None)
    if resume_path:
        print(f"[RESUME] Loading checkpoint from '{resume_path}'...")
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("val_origin_acc", ckpt.get("best_val_acc", 0.0))
        patience_count = ckpt.get("patience_count", 0)
        print(f"[RESUME] Resuming from Epoch {start_epoch} (Best Val Acc so far: {best_val_acc*100:.2f}%, Patience: {patience_count}/{PATIENCE})")
        
        # Advance LR scheduler to match resumed epoch
        for _ in range(1, start_epoch):
            scheduler.step()

    header = (
        f"{'Ep':>4} | {'TrLoss':>7} {'VlLoss':>7} | {'TrOAcc':>7} {'VlOAcc':>7} | "
        f"{'TrCAcc':>7} {'VlCAcc':>7} | {'ai':>5} {'edit':>5} {'real':>5} | Status"
    )
    print("=" * 105)
    print(header)
    print("-" * 105)

    n_train_batches = len(train_loader)
    n_val_batches   = len(val_loader)

    for epoch in range(start_epoch, EPOCHS + 1):
        # ── TRAIN ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        t_origin_correct = t_content_correct = t_coarse_correct = 0
        t_animal_correct = t_animal_total = t_total = 0
        epoch_start = time.time()
        step_count  = 0
        total_steps = (n_train_batches + ACCUM_STEPS - 1) // ACCUM_STEPS

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            x_rgb           = batch["rgb"].to(DEVICE, non_blocking=True)
            x_forensic      = batch["forensic"].to(DEVICE, non_blocking=True)
            origin_targets  = batch["origin_target"].to(DEVICE, non_blocking=True)
            content_targets = batch["content_target"].to(DEVICE, non_blocking=True)
            animal_targets  = batch["animal_target"].to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                (z_ai, z_real), content_logits, animal_logits = model(x_rgb, x_forensic)

                loss_origin  = hierarchical_origin_loss(z_ai, z_real, origin_targets,
                                                        pos_weight_ai, pos_weight_real)
                loss_content = content_crit(content_logits, content_targets)

                animal_mask = (content_targets == 1)
                if animal_mask.sum() > 0:
                    loss_animal = animal_crit(animal_logits[animal_mask], animal_targets[animal_mask])
                else:
                    loss_animal = torch.tensor(0.0, device=DEVICE)

                loss = (LAMBDA_ORIGIN  * loss_origin +
                        LAMBDA_CONTENT * loss_content +
                        LAMBDA_ANIMAL  * loss_animal)

            # Gradient accumulation: scale loss by 1/ACCUM_STEPS
            scaler.scale(loss / ACCUM_STEPS).backward()

            if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == n_train_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step_count += 1

            train_loss += loss.item() * len(origin_targets)
            m = compute_metrics(z_ai, z_real, content_logits, animal_logits,
                                origin_targets, content_targets, animal_targets)

            t_origin_correct  += m["origin_correct"]
            t_content_correct += m["content_correct"]
            t_coarse_correct  += m["coarse_correct"]
            t_animal_correct  += m["animal_correct"]
            t_animal_total    += m["animal_total"]
            t_total           += m["total"]

            cur_acc  = t_origin_correct / t_total * 100
            cur_loss = train_loss / t_total
            elapsed  = time.time() - epoch_start
            print(f"  Ep {epoch:2d} [Train] {step_count:3d}/{total_steps} "
                  f"| Loss={cur_loss:.4f} OAcc={cur_acc:.1f}% "
                  f"| {elapsed:.0f}s",
                  end="\r", flush=True)

        scheduler.step()

        train_loss_avg = train_loss / t_total
        tr_o_acc       = t_origin_correct / t_total
        tr_c_acc       = t_coarse_correct / t_total   # coarse for display

        # ── VALIDATION ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        v_origin_correct = v_content_correct = v_coarse_correct = 0
        v_animal_correct = v_animal_total = v_total = 0
        v_per_class_correct = [0, 0, 0]
        v_per_class_total   = [0, 0, 0]
        val_start = time.time()

        with torch.no_grad():
            for val_idx, batch in enumerate(val_loader):
                x_rgb           = batch["rgb"].to(DEVICE, non_blocking=True)
                x_forensic      = batch["forensic"].to(DEVICE, non_blocking=True)
                origin_targets  = batch["origin_target"].to(DEVICE, non_blocking=True)
                content_targets = batch["content_target"].to(DEVICE, non_blocking=True)
                animal_targets  = batch["animal_target"].to(DEVICE, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    (z_ai, z_real), content_logits, animal_logits = model(x_rgb, x_forensic)

                    loss_origin  = hierarchical_origin_loss(z_ai, z_real, origin_targets,
                                                            pos_weight_ai, pos_weight_real)
                    loss_content = content_crit(content_logits, content_targets)

                    animal_mask = (content_targets == 1)
                    if animal_mask.sum() > 0:
                        loss_animal = animal_crit(animal_logits[animal_mask], animal_targets[animal_mask])
                    else:
                        loss_animal = torch.tensor(0.0, device=DEVICE)

                    loss = (LAMBDA_ORIGIN  * loss_origin +
                            LAMBDA_CONTENT * loss_content +
                            LAMBDA_ANIMAL  * loss_animal)
                                   
                val_loss += loss.item() * len(origin_targets)
                m = compute_metrics(z_ai, z_real, content_logits, animal_logits,
                                    origin_targets, content_targets, animal_targets)

                v_origin_correct  += m["origin_correct"]
                v_content_correct += m["content_correct"]
                v_coarse_correct  += m["coarse_correct"]
                v_animal_correct  += m["animal_correct"]
                v_animal_total    += m["animal_total"]
                v_total           += m["total"]

                for c in range(3):
                    v_per_class_correct[c] += m["per_class_correct"][c]
                    v_per_class_total[c]   += m["per_class_total"][c]

                cur_vacc = v_origin_correct / v_total * 100
                print(f"  Ep {epoch:2d} [Val]   {val_idx+1:3d}/{n_val_batches} "
                      f"| OAcc={cur_vacc:.1f}% "
                      f"| {time.time()-val_start:.0f}s",
                      end="\r", flush=True)

        print(" " * 100, end="\r", flush=True)

        val_loss_avg = val_loss / v_total
        vl_o_acc     = v_origin_correct / v_total
        vl_c_acc     = v_coarse_correct / v_total

        per_class_acc = [
            (v_per_class_correct[c] / v_per_class_total[c] * 100.0) if v_per_class_total[c] > 0 else 0.0
            for c in range(3)
        ]

        # ── STATUS & CHECKPOINT ────────────────────────────────────────────────
        if vl_o_acc > best_val_acc:
            best_val_acc   = vl_o_acc
            patience_count = 0
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_origin_acc":   vl_o_acc,
                "per_class_acc":    per_class_acc,
                "patience_count":   patience_count,
            }, CHECKPOINT_PATH)
            status = f"--> Best Saved ({vl_o_acc * 100:.2f}%)"
        else:
            patience_count += 1
            status = f"Patience {patience_count}/{PATIENCE}"

        # Save latest checkpoint after every epoch for seamless resume
        torch.save({
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_origin_acc":   vl_o_acc,
            "best_val_acc":     best_val_acc,
            "per_class_acc":    per_class_acc,
            "patience_count":   patience_count,
        }, LATEST_CHECKPOINT_PATH)

        print(
            f"{epoch:4d} | {train_loss_avg:7.4f} {val_loss_avg:7.4f} | "
            f"{tr_o_acc*100:6.2f}% {vl_o_acc*100:6.2f}% | "
            f"{tr_c_acc*100:6.2f}% {vl_c_acc*100:6.2f}% | "
            f"{per_class_acc[0]:5.1f} {per_class_acc[1]:5.1f} {per_class_acc[2]:5.1f} | {status}"
        )

        history.append({
            "epoch":            epoch,
            "train_loss":       train_loss_avg,
            "val_loss":         val_loss_avg,
            "train_origin_acc": tr_o_acc,
            "val_origin_acc":   vl_o_acc,
            "train_coarse_acc": tr_c_acc,
            "val_coarse_acc":   vl_c_acc,
            "per_class_acc":    per_class_acc,
        })

        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)

        if patience_count >= PATIENCE:
            print(f"\n[EARLY STOPPING] Epoch {epoch}. Best Val Origin Acc: {best_val_acc*100:.2f}%")
            break

    print("=" * 105)
    print(f"Training Complete! Best Validation Origin Accuracy: {best_val_acc*100:.2f}%\n")


if __name__ == "__main__":
    main()
