"""
DS-MTFNet v5.0 — Research-Paper-Grade Evaluation Script
=========================================================
Evaluates the trained DS-MTFNet v5.0 checkpoint on the unseen test split.

Generates ALL figures and tables required for an academic research paper:

FIGURES (saved to outputs/figures/ at 300 DPI):
  Fig 1: fig1_confusion_matrix_normalized.png
         Normalized (recall-percentage) confusion matrices for both tasks.
  Fig 2: fig2_confusion_matrix_counts.png
         Raw count confusion matrices with annotation overlay.
  Fig 3: fig3_roc_curves_origin.png
         One-vs-Rest ROC curves per origin class + macro-average AUC.
  Fig 4: fig4_roc_curves_content.png
         One-vs-Rest ROC curves per content class + macro-average AUC.
  Fig 5: fig5_precision_recall_curves.png
         Precision-Recall curves (AP per class) for authenticity task.
  Fig 6: fig6_per_class_metrics_bar.png
         Grouped bar chart: Precision / Recall / F1 per class (both tasks).
  Fig 7: fig7_confidence_distribution.png
         Histogram of model confidence per predicted origin class.
  Fig 8: fig8_multitask_summary.png
         Multi-task overview bar chart: accuracy per task.

TABLES:
  outputs/paper_tables.tex          LaTeX tables for manuscript
  outputs/evaluation_metrics.json   Machine-readable full metrics
  outputs/evaluation_report.md      Complete Markdown research report
  outputs/{split}_predictions.json  Per-sample predictions
"""

import os
import sys
import json
import argparse
import warnings

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

warnings.filterwarnings("ignore")

from src.dataset import DSMTFNetDataset, get_transforms, _collate_fn
from src.model import DSMTFNetV4, HierarchicalOriginHead

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="DS-MTFNet v5.0 Evaluation")
parser.add_argument("--split",      type=str,  default="testing",
                    choices=["testing", "validation", "training"])
parser.add_argument("--checkpoint", type=str,  default=None)
parser.add_argument("--batch_size", type=int,  default=16)
args, _ = parser.parse_known_args()

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
BATCH_SIZE  = args.batch_size
OUTPUT_DIR  = "outputs"
FIG_DIR     = os.path.join(OUTPUT_DIR, "figures")

# Class labels
ORIGIN_NAMES  = ["AI Generated", "AI Edited", "Real Image"]
CONTENT_NAMES = ["Human", "Face", "Animal"]

EXPECTED_TARGETS = {
    "origin_accuracy":  "80.0%+ (v5.0 Hierarchical BCE)",
    "content_accuracy": "95.0%+",
    "joint_accuracy":   "78.0%+",
}

# ── Publication style ─────────────────────────────────────────────────────────
COLORS_ORIGIN  = ["#E63946", "#457B9D", "#2A9D8F"]   # red / blue / teal
COLORS_CONTENT = ["#F4A261", "#264653", "#E9C46A"]
COLORS_3       = ["#2196F3", "#FF5722", "#4CAF50"]    # general 3-class palette

def set_publication_style():
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    12,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
    })

# ── Checkpoint finder ─────────────────────────────────────────────────────────
def find_checkpoint() -> str:
    if args.checkpoint and os.path.exists(args.checkpoint):
        return args.checkpoint
    for p in ["checkpoints/best_v5_model.pth", "checkpoints/latest_v5_model.pth",
              "checkpoints/best_v4_model.pth",  "checkpoints/best_model.pth"]:
        if os.path.exists(p):
            return p
    if os.path.isdir("checkpoints"):
        pths = sorted([os.path.join("checkpoints", f)
                       for f in os.listdir("checkpoints") if f.endswith(".pth")],
                      key=os.path.getmtime, reverse=True)
        if pths:
            return pths[0]
    return None

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 & 2: Confusion Matrices (normalized + raw counts)
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrices(cm_o, cm_c, tag=""):
    """Generates TWO publication figures: normalized (%) and raw-count confusion matrices."""

    for normalized in [True, False]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.subplots_adjust(wspace=0.35)

        for ax, cm, names, colors, task_title in zip(
            axes,
            [cm_o, cm_c],
            [ORIGIN_NAMES, CONTENT_NAMES],
            [plt.cm.Blues, plt.cm.Greens],
            ["Task 1: Authenticity Classification",
             "Task 2: Semantic Content Classification"],
        ):
            if normalized:
                row_sums = cm.sum(axis=1, keepdims=True)
                cm_plot = np.where(row_sums > 0, cm.astype(float) / row_sums, 0)
                fmt_fn  = lambda v, raw: f"{v*100:.1f}%\n(n={raw})"
                vmax    = 1.0
                cbar_fmt = "%.0f%%"
            else:
                cm_plot = cm.astype(float)
                fmt_fn  = lambda v, raw: f"{raw}"
                vmax    = cm.max()
                cbar_fmt = "%d"

            im = ax.imshow(cm_plot, cmap=colors, aspect="auto", vmin=0, vmax=vmax)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if normalized:
                cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))

            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
            ax.set_yticklabels(names, fontsize=10)
            ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
            ax.set_ylabel("True Label",      fontsize=11, labelpad=8)
            ax.set_title(task_title, fontsize=12, fontweight="bold", pad=12)

            thresh = vmax / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    val   = cm_plot[i, j]
                    label = fmt_fn(val, cm[i, j])
                    color = "white" if val > thresh else "black"
                    ax.text(j, i, label, ha="center", va="center",
                            fontsize=9 if normalized else 11,
                            fontweight="bold", color=color)

        mode = "normalized" if normalized else "counts"
        fig_num = 1 if normalized else 2
        suptitle = (f"Fig {fig_num}: Confusion Matrices — {'Recall-Normalized (%)' if normalized else 'Raw Counts'}\n"
                    f"DS-MTFNet v5.0 | Evaluated on: {tag}")
        fig.suptitle(suptitle, fontsize=12, y=1.02, fontweight="bold")

        fname = f"fig{fig_num}_confusion_matrix_{'normalized' if normalized else 'counts'}.png"
        fig.savefig(os.path.join(FIG_DIR, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  [Fig {fig_num}] Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 & 4: ROC Curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curves(y_true, y_probs, class_names, colors, fig_num, task_title, tag=""):
    """One-vs-Rest ROC curves per class + macro-average AUC."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))

    auc_scores = []
    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f"{name}  (AUC = {roc_auc:.4f})")

    # Macro average
    all_fpr = np.unique(np.concatenate([
        roc_curve(y_bin[:, i], y_probs[:, i])[0] for i in range(n_classes)
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, "k--", linewidth=2.5,
            label=f"Macro Average  (AUC = {macro_auc:.4f})")

    ax.plot([0, 1], [0, 1], "gray", linestyle=":", linewidth=1.5, label="Random Classifier")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    ax.set_title(f"Fig {fig_num}: ROC Curves — {task_title}\n"
                 f"DS-MTFNet v5.0 | One-vs-Rest | {tag}", fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    fname = f"fig{fig_num}_roc_curves_{'origin' if fig_num == 3 else 'content'}.png"
    fig.savefig(os.path.join(FIG_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig {fig_num}] Saved: {fname}")
    return {name: float(a) for name, a in zip(class_names, auc_scores)}, float(macro_auc)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: Precision-Recall Curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_pr_curves(y_true, y_probs, class_names, colors, tag=""):
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_bin[:, i], y_probs[:, i])
        ax.step(rec, prec, where="post", color=color, linewidth=2.5,
                label=f"{name}  (AP = {ap:.4f})")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Fig 5: Precision-Recall Curves — Authenticity Classification\n"
                 f"DS-MTFNet v5.0 | {tag}", fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.savefig(os.path.join(FIG_DIR, "fig5_precision_recall_curves.png"), bbox_inches="tight")
    plt.close(fig)
    print("  [Fig 5] Saved: fig5_precision_recall_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6: Grouped Bar Chart — Precision / Recall / F1 per class
# ─────────────────────────────────────────────────────────────────────────────
def plot_per_class_metrics(metrics, tag=""):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.subplots_adjust(wspace=0.35)

    for ax, task, names, colors in zip(
        axes,
        ["origin", "content"],
        [ORIGIN_NAMES, CONTENT_NAMES],
        [COLORS_ORIGIN, COLORS_CONTENT],
    ):
        per_class = metrics[f"{task}_per_class"]
        prec  = [per_class[n]["precision"] for n in names]
        rec   = [per_class[n]["recall"]    for n in names]
        f1    = [per_class[n]["f1"]        for n in names]
        supp  = [per_class[n]["support"]   for n in names]

        x      = np.arange(len(names))
        width  = 0.24

        bars_p = ax.bar(x - width,   prec, width, label="Precision", color="#2196F3", alpha=0.85, edgecolor="white")
        bars_r = ax.bar(x,           rec,  width, label="Recall",    color="#FF5722", alpha=0.85, edgecolor="white")
        bars_f = ax.bar(x + width,   f1,   width, label="F1-Score",  color="#4CAF50", alpha=0.85, edgecolor="white")

        # Annotate value on top of each bar
        for bars in [bars_p, bars_r, bars_f]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                        f"{h*100:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

        # Support annotations below x-axis labels
        for xi, (n, s) in enumerate(zip(names, supp)):
            ax.text(xi, -0.09, f"n={s}", ha="center", fontsize=8, color="gray",
                    transform=ax.get_xaxis_transform())

        ax.set_ylim(0, 1.12)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_xlabel("Class", fontsize=12)
        task_label = "Task 1: Authenticity" if task == "origin" else "Task 2: Semantic Content"
        ax.set_title(task_label, fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", framealpha=0.85)
        ax.axhline(0.8, color="gray", linestyle=":", linewidth=1, label="_nolegend_")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Fig 6: Per-Class Precision / Recall / F1-Score — DS-MTFNet v5.0 | {tag}",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(os.path.join(FIG_DIR, "fig6_per_class_metrics_bar.png"), bbox_inches="tight")
    plt.close(fig)
    print("  [Fig 6] Saved: fig6_per_class_metrics_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7: Confidence Score Distribution
# ─────────────────────────────────────────────────────────────────────────────
def plot_confidence_distribution(all_o_true, all_o_pred, all_o_probs, tag=""):
    """Histogram of max prediction confidence per origin class, split by correct/wrong."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    for i, (ax, name) in enumerate(zip(axes, ORIGIN_NAMES)):
        mask = (all_o_true == i)
        if mask.sum() == 0:
            continue
        confs  = all_o_probs[mask, i]
        correct = (all_o_pred[mask] == i)

        bins = np.linspace(0, 1, 25)
        ax.hist(confs[correct],  bins=bins, color="#4CAF50", alpha=0.75, label="Correct", density=True)
        ax.hist(confs[~correct], bins=bins, color="#F44336", alpha=0.75, label="Incorrect", density=True)

        ax.axvline(confs.mean(), color="navy", linestyle="--", linewidth=1.5,
                   label=f"Mean = {confs.mean()*100:.1f}%")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Confidence", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.85)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Fig 7: Prediction Confidence Distribution per Origin Class — DS-MTFNet v5.0 | {tag}",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.savefig(os.path.join(FIG_DIR, "fig7_confidence_distribution.png"), bbox_inches="tight")
    plt.close(fig)
    print("  [Fig 7] Saved: fig7_confidence_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8: Multi-Task Summary Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_multitask_summary(metrics, auc_o_macro, auc_c_macro, tag=""):
    tasks = [
        "Authenticity\n(Origin) Acc",
        "Semantic Content\nAcc",
        "Joint Accuracy\n(Both Tasks)",
        "Origin\nMacro-AUC",
        "Content\nMacro-AUC",
    ]
    values = [
        metrics["origin_accuracy"]  * 100,
        metrics["content_accuracy"] * 100,
        metrics["joint_accuracy"]   * 100,
        auc_o_macro * 100,
        auc_c_macro * 100,
    ]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(tasks, values, color=colors, alpha=0.88, edgecolor="white", width=0.55)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(80,  color="#E63946", linestyle="--", linewidth=1.5, alpha=0.7, label="Origin target 80%")
    ax.axhline(95,  color="#2A9D8F", linestyle="--", linewidth=1.5, alpha=0.7, label="Content target 95%")
    ax.axhline(90,  color="#6A0572", linestyle=":",  linewidth=1.5, alpha=0.6, label="AUC target 90%")

    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title(f"Fig 8: DS-MTFNet v5.0 Multi-Task Performance Summary\n{tag}",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.savefig(os.path.join(FIG_DIR, "fig8_multitask_summary.png"), bbox_inches="tight")
    plt.close(fig)
    print("  [Fig 8] Saved: fig8_multitask_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table generators
# ─────────────────────────────────────────────────────────────────────────────
def generate_latex_table_origin(metrics, auc_scores) -> str:
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Per-Class Performance on Task 1: Authenticity Classification (DS-MTFNet v5.0, Unseen Test Set)}",
        r"\label{tab:task1_origin}",
        r"\begin{tabular}{lcccccc}", r"\hline",
        r"\textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{AUC-ROC} & \textbf{Support} \\",
        r"\hline",
    ]
    for name in ORIGIN_NAMES:
        m   = metrics["origin_per_class"][name]
        roc = auc_scores.get(name, 0.0)
        lines.append(f"{name} & {m['precision']*100:.2f}\\% & {m['recall']*100:.2f}\\% "
                     f"& {m['f1']*100:.2f}\\% & {roc:.4f} & {m['support']} \\\\")
    lines += [
        r"\hline",
        f"\\textbf{{Macro Avg}} & {metrics['origin_macro']['precision']*100:.2f}\\% "
        f"& {metrics['origin_macro']['recall']*100:.2f}\\% "
        f"& {metrics['origin_macro']['f1']*100:.2f}\\% & {auc_scores.get('macro', 0):.4f} & {metrics['total_samples']} \\\\",
        f"\\textbf{{Overall Accuracy}} & \\multicolumn{{6}}{{c}}{{\\textbf{{{metrics['origin_accuracy']*100:.2f}\\%}}}} \\\\",
        r"\hline", r"\end{tabular}", r"\end{table}",
    ]
    return "\n".join(lines)


def generate_latex_table_content(metrics) -> str:
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Per-Class Performance on Task 2: Semantic Content Classification (DS-MTFNet v5.0)}",
        r"\label{tab:task2_content}",
        r"\begin{tabular}{lcccc}", r"\hline",
        r"\textbf{Category} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Support} \\",
        r"\hline",
    ]
    for name in CONTENT_NAMES:
        m = metrics["content_per_class"][name]
        lines.append(f"{name} & {m['precision']*100:.2f}\\% & {m['recall']*100:.2f}\\% "
                     f"& {m['f1']*100:.2f}\\% & {m['support']} \\\\")
    lines += [
        r"\hline",
        f"\\textbf{{Macro Avg}} & {metrics['content_macro']['precision']*100:.2f}\\% "
        f"& {metrics['content_macro']['recall']*100:.2f}\\% "
        f"& {metrics['content_macro']['f1']*100:.2f}\\% & {metrics['total_samples']} \\\\",
        f"\\textbf{{Overall Accuracy}} & \\multicolumn{{4}}{{c}}{{\\textbf{{{metrics['content_accuracy']*100:.2f}\\%}}}} \\\\",
        r"\hline", r"\end{tabular}", r"\end{table}",
    ]
    return "\n".join(lines)


def pretty_confusion(cm, labels):
    col_w = max(len(l) for l in labels) + 4
    lines = [f"{'Predicted ->':>16} " + "".join(f"{l:>{col_w}}" for l in labels)]
    lines.append("-" * (18 + col_w * len(labels)))
    for i, row in enumerate(cm):
        lines.append(f"{labels[i]:>16} |" + "".join(f"{v:>{col_w}}" for v in row))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR,    exist_ok=True)
    set_publication_style()

    checkpoint = find_checkpoint()
    if not checkpoint:
        print("[ERROR] No checkpoint found in checkpoints/. Please train the model first using train.py.")
        return

    print("\n" + "=" * 72)
    print("  DS-MTFNet v5.0 — Research-Paper Evaluation Framework")
    print(f"  Device       : {DEVICE}")
    print(f"  Checkpoint   : {checkpoint}")
    print("=" * 72 + "\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    eval_dir = os.path.join("dataset", args.split)
    if not os.path.exists(eval_dir):
        eval_dir = "dataset/testing" if os.path.exists("dataset/testing") else "dataset/validation"
    is_unseen = "testing" in eval_dir
    print(f"  Split        : {eval_dir}  ({'UNSEEN TEST DATA' if is_unseen else 'VALIDATION DATA'})")

    test_ds = DSMTFNetDataset(eval_dir, transform=get_transforms(train=False), is_train=False)
    if len(test_ds) == 0:
        print(f"[ERROR] No images found at {eval_dir}")
        return
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, collate_fn=_collate_fn)
    print(f"  Samples      : {len(test_ds):,}  |  Batches: {len(loader)}\n")

    # ── Model ────────────────────────────────────────────────────────────────
    model = DSMTFNetV4(dinov2_model_name="dinov2_vitb14",
                       num_unfreeze_blocks=4, dropout=0.45).to(DEVICE)
    ckpt  = torch.load(checkpoint, map_location=DEVICE)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"  Checkpoint loaded: {checkpoint}\n")

    use_amp = (DEVICE == "cuda")

    # ── Inference ────────────────────────────────────────────────────────────
    all_o_true, all_o_pred, all_o_probs = [], [], []
    all_c_true, all_c_pred, all_c_probs = [], [], []
    sample_details = []

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x_rgb      = batch["rgb"].to(DEVICE)
            x_forensic = batch["forensic"].to(DEVICE)
            o_lbl      = batch["origin_target"]
            c_lbl      = batch["content_target"]

            with torch.cuda.amp.autocast(enabled=use_amp):
                (z_ai, z_real), c_logits, _ = model(x_rgb, x_forensic)

            o_probs = HierarchicalOriginHead.to_probs(z_ai, z_real).cpu()   # (B, 3)
            c_probs = torch.softmax(c_logits, dim=1).cpu()                   # (B, 3)
            o_preds = o_probs.argmax(dim=1).tolist()
            c_preds = c_probs.argmax(dim=1).tolist()

            all_o_true.extend(o_lbl.tolist())
            all_o_pred.extend(o_preds)
            all_o_probs.append(o_probs.numpy())
            all_c_true.extend(c_lbl.tolist())
            all_c_pred.extend(c_preds)
            all_c_probs.append(c_probs.numpy())

            for ot, op, ct, cp in zip(o_lbl.tolist(), o_preds, c_lbl.tolist(), c_preds):
                sample_details.append({
                    "origin_true": ot, "origin_pred": op,
                    "content_true": ct, "content_pred": cp,
                    "joint_correct": (ot == op and ct == cp),
                })

            print(f"  Batch {bi+1:3d}/{len(loader)} processed", end="\r", flush=True)

    print(" " * 60, end="\r")

    all_o_true  = np.array(all_o_true)
    all_o_pred  = np.array(all_o_pred)
    all_o_probs = np.vstack(all_o_probs)
    all_c_true  = np.array(all_c_true)
    all_c_pred  = np.array(all_c_pred)
    all_c_probs = np.vstack(all_c_probs)

    # ── Core Metrics ─────────────────────────────────────────────────────────
    acc_o     = accuracy_score(all_o_true, all_o_pred)
    acc_c     = accuracy_score(all_c_true, all_c_pred)
    acc_joint = sum(s["joint_correct"] for s in sample_details) / len(sample_details)

    p_o, r_o, f_o, s_o = precision_recall_fscore_support(
        all_o_true, all_o_pred, labels=[0, 1, 2], zero_division=0)
    p_c, r_c, f_c, s_c = precision_recall_fscore_support(
        all_c_true, all_c_pred, labels=[0, 1, 2], zero_division=0)

    cm_o = confusion_matrix(all_o_true, all_o_pred, labels=[0, 1, 2])
    cm_c = confusion_matrix(all_c_true, all_c_pred, labels=[0, 1, 2])

    metrics = {
        "model":          "DS-MTFNet v5.0",
        "checkpoint":     checkpoint,
        "eval_split":     eval_dir,
        "total_samples":  len(sample_details),
        "origin_accuracy":  float(acc_o),
        "content_accuracy": float(acc_c),
        "joint_accuracy":   float(acc_joint),
        "origin_per_class":  {ORIGIN_NAMES[i]:  {"precision": float(p_o[i]), "recall": float(r_o[i]),
                                                  "f1": float(f_o[i]), "support": int(s_o[i])}
                               for i in range(3)},
        "origin_macro":  {"precision": float(p_o.mean()), "recall": float(r_o.mean()), "f1": float(f_o.mean())},
        "content_per_class": {CONTENT_NAMES[i]: {"precision": float(p_c[i]), "recall": float(r_c[i]),
                                                  "f1": float(f_c[i]), "support": int(s_c[i])}
                               for i in range(3)},
        "content_macro": {"precision": float(p_c.mean()), "recall": float(r_c.mean()), "f1": float(f_c.mean())},
        "confusion_matrix_origin":  cm_o.tolist(),
        "confusion_matrix_content": cm_c.tolist(),
    }

    # ── Terminal Summary ──────────────────────────────────────────────────────
    tag = f"{'Unseen Test Set' if is_unseen else 'Validation Set'} | n={len(test_ds)}"
    print("=" * 72)
    print("  PERFORMANCE SUMMARY")
    print("=" * 72)
    print(f"  Task 1 — Authenticity Accuracy : {acc_o*100:6.2f}%  | Target: {EXPECTED_TARGETS['origin_accuracy']}")
    print(f"  Task 2 — Content Accuracy      : {acc_c*100:6.2f}%  | Target: {EXPECTED_TARGETS['content_accuracy']}")
    print(f"  Joint  — Both Tasks Correct    : {acc_joint*100:6.2f}%  | Target: {EXPECTED_TARGETS['joint_accuracy']}")
    print("-" * 72)
    print("\nTask 1 — Authenticity Classification:")
    print(classification_report(all_o_true, all_o_pred, target_names=ORIGIN_NAMES, digits=4, zero_division=0))
    print("Confusion Matrix:")
    print(pretty_confusion(cm_o, ORIGIN_NAMES))
    print("\nTask 2 — Semantic Content Classification:")
    print(classification_report(all_c_true, all_c_pred, target_names=CONTENT_NAMES, digits=4, zero_division=0))

    # ── Generate ALL Figures ──────────────────────────────────────────────────
    print("\n[FIGURES] Generating research-paper figures...")

    # Fig 1 & 2: Confusion matrices
    plot_confusion_matrices(cm_o, cm_c, tag=tag)

    # Fig 3: ROC — Authenticity
    auc_o_per_class, auc_o_macro = plot_roc_curves(
        all_o_true, all_o_probs, ORIGIN_NAMES, COLORS_ORIGIN, 3, "Authenticity Classification", tag)
    metrics["roc_auc_origin"] = {**auc_o_per_class, "macro": auc_o_macro}

    # Fig 4: ROC — Content
    auc_c_per_class, auc_c_macro = plot_roc_curves(
        all_c_true, all_c_probs, CONTENT_NAMES, COLORS_CONTENT, 4, "Semantic Content Classification", tag)
    metrics["roc_auc_content"] = {**auc_c_per_class, "macro": auc_c_macro}

    # Fig 5: Precision-Recall
    plot_pr_curves(all_o_true, all_o_probs, ORIGIN_NAMES, COLORS_ORIGIN, tag)

    # Fig 6: Per-class grouped bars
    plot_per_class_metrics(metrics, tag)

    # Fig 7: Confidence distributions
    plot_confidence_distribution(all_o_true, all_o_pred, all_o_probs, tag)

    # Fig 8: Multi-task summary
    plot_multitask_summary(metrics, auc_o_macro, auc_c_macro, tag)

    # ── Save Research Artifacts ───────────────────────────────────────────────
    print("\n[SAVING] Research artifacts...")

    # JSON metrics
    with open(os.path.join(OUTPUT_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Per-sample predictions
    preds_file = os.path.join(OUTPUT_DIR, f"{args.split}_predictions.json")
    with open(preds_file, "w") as f:
        json.dump(sample_details, f, indent=2)

    # LaTeX tables (with AUC added to origin table)
    auc_for_table = {**auc_o_per_class, "macro": auc_o_macro}
    latex_o  = generate_latex_table_origin(metrics, auc_for_table)
    latex_c  = generate_latex_table_content(metrics)
    latex_full = (
        "% ================================================================\n"
        "% DS-MTFNet v5.0 — Auto-Generated LaTeX Tables for Research Paper\n"
        "% ================================================================\n\n"
        + latex_o + "\n\n" + latex_c + "\n"
    )
    with open(os.path.join(OUTPUT_DIR, "paper_tables.tex"), "w") as f:
        f.write(latex_full)

    # Markdown research report
    split_title = f"{'Unseen Test Set' if is_unseen else 'Validation Set'} (`{eval_dir}`)"
    md = []
    md.append(f"# DS-MTFNet v5.0 — Research Evaluation Report\n")
    md.append(f"**Split Evaluated**: {split_title} — {'**UNSEEN DATA**' if is_unseen else 'Validation'}\n")
    md.append(f"**Samples**: {len(test_ds):,} | **Checkpoint**: `{checkpoint}` | **Device**: `{DEVICE}`\n\n")
    md.append("---\n\n## 1. Multi-Task Performance Summary\n\n")
    md.append("| Task | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Macro-AUC | Target |\n")
    md.append("|------|----------|-----------------|--------------|----------|-----------|--------|\n")
    md.append(f"| **Authenticity (Origin)** | **{acc_o*100:.2f}%** | {p_o.mean()*100:.2f}% | {r_o.mean()*100:.2f}% | {f_o.mean()*100:.2f}% | {auc_o_macro:.4f} | {EXPECTED_TARGETS['origin_accuracy']} |\n")
    md.append(f"| **Semantic Content** | **{acc_c*100:.2f}%** | {p_c.mean()*100:.2f}% | {r_c.mean()*100:.2f}% | {f_c.mean()*100:.2f}% | {auc_c_macro:.4f} | {EXPECTED_TARGETS['content_accuracy']} |\n")
    md.append(f"| **Joint (Both Correct)** | **{acc_joint*100:.2f}%** | — | — | — | — | {EXPECTED_TARGETS['joint_accuracy']} |\n\n")

    md.append("---\n\n## 2. Task 1: Authenticity Classification (Origin)\n\n")
    md.append("```\n" + classification_report(all_o_true, all_o_pred, target_names=ORIGIN_NAMES, digits=4, zero_division=0) + "```\n\n")
    md.append("### Confusion Matrix (Raw Counts)\n```\n" + pretty_confusion(cm_o, ORIGIN_NAMES) + "\n```\n\n")

    md.append("### AUC-ROC per Class\n\n| Class | AUC-ROC |\n|-------|--------|\n")
    for name, a in auc_o_per_class.items():
        md.append(f"| {name} | {a:.4f} |\n")
    md.append(f"| **Macro Average** | **{auc_o_macro:.4f}** |\n\n")

    md.append("---\n\n## 3. Task 2: Semantic Content Classification\n\n")
    md.append("```\n" + classification_report(all_c_true, all_c_pred, target_names=CONTENT_NAMES, digits=4, zero_division=0) + "```\n\n")
    md.append("### Confusion Matrix (Raw Counts)\n```\n" + pretty_confusion(cm_c, CONTENT_NAMES) + "\n```\n\n")

    md.append("---\n\n## 4. Generated Research Figures\n\n")
    md.append("| Figure | File | Description |\n|--------|------|-------------|\n")
    figs = [
        ("Fig 1", "fig1_confusion_matrix_normalized.png", "Recall-normalized confusion matrices (both tasks)"),
        ("Fig 2", "fig2_confusion_matrix_counts.png", "Raw-count confusion matrices with annotations"),
        ("Fig 3", "fig3_roc_curves_origin.png", "One-vs-Rest ROC curves — Authenticity (with AUC)"),
        ("Fig 4", "fig4_roc_curves_content.png", "One-vs-Rest ROC curves — Content Classification"),
        ("Fig 5", "fig5_precision_recall_curves.png", "Precision-Recall curves with AP per class"),
        ("Fig 6", "fig6_per_class_metrics_bar.png", "Grouped Precision/Recall/F1 bar chart per class"),
        ("Fig 7", "fig7_confidence_distribution.png", "Confidence score distribution (correct vs incorrect)"),
        ("Fig 8", "fig8_multitask_summary.png", "Multi-task performance overview bar chart"),
    ]
    for label, fname, desc in figs:
        md.append(f"| {label} | `outputs/figures/{fname}` | {desc} |\n")

    md.append("\n---\n\n## 5. LaTeX Tables\n\n")
    md.append("See `outputs/paper_tables.tex` for copy-paste ready LaTeX tables.\n\n")
    md.append("```latex\n" + latex_full + "```\n")

    with open(os.path.join(OUTPUT_DIR, "evaluation_report.md"), "w", encoding="utf-8") as f:
        f.writelines(md)

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  EVALUATION COMPLETE — Research Artifacts Saved")
    print("=" * 72)
    print(f"  Figures (8 total) : outputs/figures/")
    for _, fname, _ in figs:
        print(f"    {fname}")
    print(f"  Metrics JSON      : outputs/evaluation_metrics.json")
    print(f"  Predictions JSON  : {preds_file}")
    print(f"  LaTeX Tables      : outputs/paper_tables.tex")
    print(f"  Markdown Report   : outputs/evaluation_report.md")
    print("=" * 72 + "\n")

    print(f"  Origin AUC:  {' | '.join(f'{n}: {a:.4f}' for n, a in auc_o_per_class.items())} | Macro: {auc_o_macro:.4f}")
    print(f"  Content AUC: {' | '.join(f'{n}: {a:.4f}' for n, a in auc_c_per_class.items())} | Macro: {auc_c_macro:.4f}")


if __name__ == "__main__":
    evaluate()
