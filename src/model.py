"""
DS-MTFNet Architecture v5.0 — Hierarchical Origin Head Edition
===============================================================
Key Changes vs v4.x:
  - HIERARCHICAL ORIGIN HEAD: replaces 3-way softmax with 2 independent binary logits
      z_ai   : P(AI)              → easy split, strong forensic signal (score 0.485)
      z_real : P(Real | Not-AI)  → dedicated binary BCE on the hard Real vs Edit pair
    This eliminates the 3-way denominator tug-of-war that caused edit/real oscillation.
  - FORENSIC BRANCH expanded to 10 features (added ELA P95 + Sat Std from audit table)
    - ELA P95 has score 0.226 (AI vs Edit) — strongest ELA discriminator
    - Sat Std  has score 0.242 (Real vs Edit) — useful for the hard pair
  - ForensicMLP: 10→128→64 with BN + input jitter (training stability)
  - Fusion input: 384 + 128 + 64 = 576 → 512 (unchanged)
  - DINOv2 last 4 blocks UNFROZEN at LR=5e-6 (forensic adaptation)

Architecture:
  1. RGB Branch       : DINOv2 ViT-B/14 (560x560 input) → rgb_proj(768→384)
  2. Frequency Branch : FFT-CNN (128-d)
  3. Forensic Branch  : Handcrafted 10-Feature MLP (64-d)
  4. Fusion           : MLP(576→512) + Dropout(0.45)
  5. Heads            : HierarchicalOriginHead, Content(3), Animal(5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Number of forensic features (expanded to 10 in v5.0)
FORENSIC_DIM = 10


class FrequencyBranch(nn.Module):
    """
    Lightweight CNN to extract frequency-domain embeddings from 2D FFT magnitude spectra.
    Uses AdaptiveAvgPool so it handles any input resolution (224, 336, 476, 560…).
    Input: (B, 3, H, W) RGB image tensor
    Output: (B, 128) Frequency Embedding
    """
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1))   # handles any resolution
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        gray = 0.299 * x_rgb[:, 0:1] + 0.587 * x_rgb[:, 1:2] + 0.114 * x_rgb[:, 2:3]
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        mag = torch.log(torch.abs(fft_shift) + 1e-8)
        mean = mag.mean(dim=(1, 2, 3), keepdim=True)
        std  = mag.std(dim=(1, 2, 3), keepdim=True) + 1e-8
        mag_norm = (mag - mean) / std
        feat = self.conv(mag_norm).view(x_rgb.size(0), -1)
        return self.fc(feat)


class ForensicMLPBranch(nn.Module):
    """
    MLP for 10 audit-selected forensic features (v5.0 expanded from 8 to 10).
    Added features from dataset audit (Section 5 - Class Separation Scores):
      - ELA P95   : score 0.226 AI vs Edited (strongest ELA discriminator)
      - Sat Std   : score 0.242 Real vs Edited (useful for the hard pair)
    Includes input feature jitter during training to prevent memorization of exact values.
    Input:  (B, 10)
    Output: (B, 64)
    """
    def __init__(self, in_features: int = FORENSIC_DIM, embed_dim: int = 64,
                 jitter_std: float = 0.03):
        super().__init__()
        self.jitter_std = jitter_std
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(in_features),        # normalizes wildly different feature scales
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, embed_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature-space jitter: forensic statistics are noisy estimates, not ground truth.
        # Small multiplicative+additive noise prevents the MLP memorizing exact values.
        if self.training and self.jitter_std > 0:
            x = x * (1 + torch.randn_like(x) * self.jitter_std) + torch.randn_like(x) * 1e-3
        return self.mlp(x)


class HierarchicalOriginHead(nn.Module):
    """
    Two independent binary decisions replacing the flat 3-way softmax.

    Decision 1 — z_ai   : P(AI)              (easy, best separation score 0.485)
    Decision 2 — z_real : P(Real | Not-AI)  (hard pair, best score 0.243)

    The flat softmax forces these two problems to share the same denominator, creating
    a zero-sum competition where edit↑ forces real↓ and vice versa. Splitting into two
    binary logits gives each decision its own gradient and its own loss term.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        # Each is a small MLP rather than a single Linear for added capacity
        self.ai_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1)        # logit for P(is_AI)
        )
        self.real_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1)        # logit for P(is_Real | not_AI)
        )

    def forward(self, x: torch.Tensor):
        z_ai   = self.ai_head(x).squeeze(-1)    # (B,)
        z_real = self.real_head(x).squeeze(-1)  # (B,)
        return z_ai, z_real

    @staticmethod
    def to_probs(z_ai: torch.Tensor, z_real: torch.Tensor) -> torch.Tensor:
        """
        Converts two binary logits into a proper 3-class probability distribution.
        Class order matches ORIGIN_CLASSES = {ai: 0, ai_edited: 1, real: 2}
        """
        p_ai   = torch.sigmoid(z_ai)                         # P(AI)
        p_real = (1 - p_ai) * torch.sigmoid(z_real)          # P(Real)  = P(Not-AI) * P(Real|Not-AI)
        p_edit = (1 - p_ai) * (1 - torch.sigmoid(z_real))    # P(Edited)= P(Not-AI) * P(Edit|Not-AI)
        return torch.stack([p_ai, p_edit, p_real], dim=1)    # (B, 3)


class DSMTFNetV4(nn.Module):
    """
    DS-MTFNet v5.0 — Hierarchical Origin Head + 10-Feature Forensic Branch
    =========================================================================
    Trainable groups:
      Group A (LR=LR_HEAD)             : FFT-CNN, Forensic-MLP, rgb_proj, Fusion, all Heads
      Group B (LR=LR_BACKBONE_PARTIAL) : DINOv2 last 4 blocks + norm  (forensic adaptation)
      Group C (frozen, LR=0)           : DINOv2 first 8 blocks + patch_embed + cls_token + pos_embed
    """
    def __init__(
        self,
        dinov2_model_name: str = "dinov2_vitb14",
        num_unfreeze_blocks: int = 4,
        dropout: float = 0.45,
    ):
        super().__init__()
        print(f"[MODEL] Loading DINOv2 ViT-B/14 ({dinov2_model_name})...")
        try:
            self.rgb_branch = torch.hub.load("facebookresearch/dinov2", dinov2_model_name)
        except Exception as e:
            print(f"[WARNING] Hub load failed ({e}). Using local cache.")
            self.rgb_branch = torch.hub.load(
                "facebookresearch/dinov2", dinov2_model_name, skip_validation=True
            )

        # ── Step 1: Freeze ALL DINOv2 parameters ─────────────────────────────
        for p in self.rgb_branch.parameters():
            p.requires_grad = False

        # ── Step 2: Selectively unfreeze the last N transformer blocks + norm ─
        total_blocks = len(self.rgb_branch.blocks)  # 12 for ViT-B/14
        unfreeze_from = max(0, total_blocks - num_unfreeze_blocks)
        for block in self.rgb_branch.blocks[unfreeze_from:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.rgb_branch.norm.parameters():
            p.requires_grad = True

        unfrozen_blocks = total_blocks - unfreeze_from
        frozen_blocks   = unfreeze_from
        print(f"[MODEL] DINOv2: {frozen_blocks} blocks FROZEN / {unfrozen_blocks} blocks UNFROZEN (forensic fine-tuning)")

        # ── Projection: DINOv2 768-d → 384-d ──────────────────────────────────
        self.rgb_proj = nn.Sequential(
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.25)
        )

        self.freq_branch     = FrequencyBranch(embed_dim=128)
        self.forensic_branch = ForensicMLPBranch(in_features=FORENSIC_DIM, embed_dim=64)

        # ── Fusion: 384 + 128 + 64 = 576 → 512 ───────────────────────────────
        fusion_in = 384 + 128 + 64  # = 576
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # ── Hierarchical Origin Head (v5.0 upgrade from flat 3-way softmax) ──
        self.origin_head  = HierarchicalOriginHead(in_dim=512)
        self.content_head = nn.Linear(512, 3)  # Human / Face / Animal
        self.animal_head  = nn.Linear(512, 5)  # Cat / Dog / Elephant / Horse / Lion

    def get_param_groups(self):
        """Returns two parameter groups: backbone (low LR) and head (higher LR)."""
        backbone_ids = set()
        total_blocks = len(self.rgb_branch.blocks)
        for block in self.rgb_branch.blocks[total_blocks - 4:]:
            for p in block.parameters():
                backbone_ids.add(id(p))
        for p in self.rgb_branch.norm.parameters():
            backbone_ids.add(id(p))

        backbone_params = [p for p in self.parameters() if p.requires_grad and id(p) in backbone_ids]
        head_params     = [p for p in self.parameters() if p.requires_grad and id(p) not in backbone_ids]
        return head_params, backbone_params

    def forward(self, x_rgb: torch.Tensor, x_forensic: torch.Tensor):
        # 1. RGB / Semantic (partially fine-tuned DINOv2 CLS token)
        rgb_feat = self.rgb_branch(x_rgb)          # (B, 768)
        rgb_feat = self.rgb_proj(rgb_feat)          # (B, 384)

        # 2. Frequency (FFT magnitude spectrum → CNN)
        freq_feat = self.freq_branch(x_rgb)         # (B, 128)

        # 3. Forensic (10 handcrafted features → MLP)
        forensic_feat = self.forensic_branch(x_forensic)  # (B, 64)

        # 4. Fusion
        combined    = torch.cat([rgb_feat, freq_feat, forensic_feat], dim=-1)  # (B, 576)
        shared_feat = self.fusion(combined)                                     # (B, 512)

        # 5. Heads
        z_ai, z_real   = self.origin_head(shared_feat)     # (B,), (B,)
        content_logits = self.content_head(shared_feat)    # (B, 3)
        animal_logits  = self.animal_head(shared_feat)     # (B, 5)

        return (z_ai, z_real), content_logits, animal_logits
