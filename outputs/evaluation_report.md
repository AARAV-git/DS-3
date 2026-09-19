# DS-MTFNet v5.0 — Research Evaluation Report
**Split Evaluated**: Unseen Test Set (`dataset\testing`) — **UNSEEN DATA**
**Samples**: 1,080 | **Checkpoint**: `checkpoints/best_v5_model.pth` | **Device**: `cuda`

---

## 1. Multi-Task Performance Summary

| Task | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Macro-AUC | Target |
|------|----------|-----------------|--------------|----------|-----------|--------|
| **Authenticity (Origin)** | **89.91%** | 89.81% | 89.91% | 89.83% | 0.9709 | 80.0%+ (v5.0 Hierarchical BCE) |
| **Semantic Content** | **98.33%** | 98.35% | 98.33% | 98.33% | 0.9994 | 95.0%+ |
| **Joint (Both Correct)** | **88.43%** | — | — | — | — | 78.0%+ |

---

## 2. Task 1: Authenticity Classification (Origin)

```
              precision    recall  f1-score   support

AI Generated     0.9416    0.9861    0.9634       360
   AI Edited     0.8612    0.8444    0.8527       360
  Real Image     0.8914    0.8667    0.8789       360

    accuracy                         0.8991      1080
   macro avg     0.8981    0.8991    0.8983      1080
weighted avg     0.8981    0.8991    0.8983      1080
```

### Confusion Matrix (Raw Counts)
```
    Predicted ->     AI Generated       AI Edited      Real Image
------------------------------------------------------------------
    AI Generated |             355               4               1
       AI Edited |              19             304              37
      Real Image |               3              45             312
```

### AUC-ROC per Class

| Class | AUC-ROC |
|-------|--------|
| AI Generated | 0.9904 |
| AI Edited | 0.9456 |
| Real Image | 0.9760 |
| **Macro Average** | **0.9709** |

---

## 3. Task 2: Semantic Content Classification

```
              precision    recall  f1-score   support

       Human     0.9914    0.9583    0.9746       360
        Face     0.9756    1.0000    0.9877       360
      Animal     0.9835    0.9917    0.9876       360

    accuracy                         0.9833      1080
   macro avg     0.9835    0.9833    0.9833      1080
weighted avg     0.9835    0.9833    0.9833      1080
```

### Confusion Matrix (Raw Counts)
```
    Predicted ->      Human      Face    Animal
------------------------------------------------
           Human |       345         9         6
            Face |         0       360         0
          Animal |         3         0       357
```

---

## 4. Generated Research Figures

| Figure | File | Description |
|--------|------|-------------|
| Fig 1 | `outputs/figures/fig1_confusion_matrix_normalized.png` | Recall-normalized confusion matrices (both tasks) |
| Fig 2 | `outputs/figures/fig2_confusion_matrix_counts.png` | Raw-count confusion matrices with annotations |
| Fig 3 | `outputs/figures/fig3_roc_curves_origin.png` | One-vs-Rest ROC curves — Authenticity (with AUC) |
| Fig 4 | `outputs/figures/fig4_roc_curves_content.png` | One-vs-Rest ROC curves — Content Classification |
| Fig 5 | `outputs/figures/fig5_precision_recall_curves.png` | Precision-Recall curves with AP per class |
| Fig 6 | `outputs/figures/fig6_per_class_metrics_bar.png` | Grouped Precision/Recall/F1 bar chart per class |
| Fig 7 | `outputs/figures/fig7_confidence_distribution.png` | Confidence score distribution (correct vs incorrect) |
| Fig 8 | `outputs/figures/fig8_multitask_summary.png` | Multi-task performance overview bar chart |

---

## 5. LaTeX Tables

See `outputs/paper_tables.tex` for copy-paste ready LaTeX tables.

```latex
% ================================================================
% DS-MTFNet v5.0 — Auto-Generated LaTeX Tables for Research Paper
% ================================================================

\begin{table}[htbp]
\centering
\caption{Per-Class Performance on Task 1: Authenticity Classification (DS-MTFNet v5.0, Unseen Test Set)}
\label{tab:task1_origin}
\begin{tabular}{lcccccc}
\hline
\textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{AUC-ROC} & \textbf{Support} \\
\hline
AI Generated & 94.16\% & 98.61\% & 96.34\% & 0.9904 & 360 \\
AI Edited & 86.12\% & 84.44\% & 85.27\% & 0.9456 & 360 \\
Real Image & 89.14\% & 86.67\% & 87.89\% & 0.9760 & 360 \\
\hline
\textbf{Macro Avg} & 89.81\% & 89.91\% & 89.83\% & 0.9709 & 1080 \\
\textbf{Overall Accuracy} & \multicolumn{6}{c}{\textbf{89.91\%}} \\
\hline
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{Per-Class Performance on Task 2: Semantic Content Classification (DS-MTFNet v5.0)}
\label{tab:task2_content}
\begin{tabular}{lcccc}
\hline
\textbf{Category} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Support} \\
\hline
Human & 99.14\% & 95.83\% & 97.46\% & 360 \\
Face & 97.56\% & 100.00\% & 98.77\% & 360 \\
Animal & 98.35\% & 99.17\% & 98.76\% & 360 \\
\hline
\textbf{Macro Avg} & 98.35\% & 98.33\% & 98.33\% & 1080 \\
\textbf{Overall Accuracy} & \multicolumn{4}{c}{\textbf{98.33\%}} \\
\hline
\end{tabular}
\end{table}
```
