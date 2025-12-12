"""
make_report_figures.py

Generate report figures:
1) CNN training curves 
2) CNN confusion matrix 
3) Wav2Vec2 confusion matrix 
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


# Config 
DEFAULT_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
SHORT_UPPER = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]

# Utils
def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _normalize_label(x):
    if isinstance(x, str):
        return x.strip()
    return x


def detect_mapping_style(values) -> str:
    """Heuristically detect if labels are ANG/DIS... or long names."""
    uniq = set([str(v).strip() for v in values])
    if uniq.issubset(set(SHORT_UPPER)):
        return "short_upper"
    if uniq.issubset(set([s.lower() for s in SHORT_UPPER])):
        return "short_lower"
    if uniq.issubset(set(DEFAULT_LABELS)) or uniq.issubset(set([s.capitalize() for s in DEFAULT_LABELS])):
        return "long"
    return "mixed"


def to_index_series(s: pd.Series, labels: list[str], map_style: str = "auto") -> pd.Series:
    """
    Map a label Series (str or int) to 0..K-1 indices following 'labels' order.
    """
    s = s.apply(_normalize_label)
    if s.dtype.kind in ("i", "u"):
        return s.astype(int)

    # Detect source style
    style = map_style if map_style != "auto" else detect_mapping_style(s.values)
    canonical = {}
    for su, long in zip(SHORT_UPPER, DEFAULT_LABELS):
        canonical[su] = long
        canonical[su.lower()] = long
    for long in DEFAULT_LABELS:
        canonical[long] = long
        canonical[long.capitalize()] = long

    mapped = s.map(lambda x: canonical.get(str(x), str(x))).str.lower()

    label_to_idx = {lab.lower(): i for i, lab in enumerate(labels)}
    idx = mapped.map(lambda x: label_to_idx.get(x, np.nan))
    if idx.isna().any():
        missing = sorted(set(mapped[idx.isna()]))
        raise ValueError(
            f"Unmapped labels found: {missing}. Provided labels={labels}. "
            f"If your CSV uses ANG/DIS..., pass --labels angry,disgust,fear,happy,neutral,sad "
            f"(order matters)."
        )
    return idx.astype(int)


# Plotting
def plot_training_curves(hist_csv: str, out_png: str) -> None:
    """
    Draw loss and accuracy curves vs epoch into a single figure with two y-axes.
    Accepts forgiving column names; will fabricate 'epoch' if missing.
    """
    df = pd.read_csv(hist_csv)

    # Normalize column names
    rename_map = {
        "Epoch": "epoch",
        "epoch": "epoch",
        "Train_Loss": "train_loss",
        "Val_Loss": "val_loss",
        "Train_Acc": "train_acc",
        "Val_Acc": "val_acc",
        "train_loss": "train_loss",
        "val_loss": "val_loss",
        "train_acc": "train_acc",
        "val_acc": "val_acc",
    }
    df = df.rename(columns=rename_map)
    needed = {"train_loss", "val_loss", "train_acc", "val_acc"}
    if not needed.issubset(df.columns):
        raise ValueError(
            f"cnn-history CSV must have columns {needed}. Found: {list(df.columns)}"
        )
    if "epoch" not in df.columns:
        df["epoch"] = np.arange(1, len(df) + 1)

    plt.figure(figsize=(9, 6))
    ax1 = plt.gca()
    l1, = ax1.plot(df["epoch"], df["train_loss"], label="Train Loss")
    l2, = ax1.plot(df["epoch"], df["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()
    l3, = ax2.plot(df["epoch"], df["train_acc"], linestyle="--", label="Train Acc")
    l4, = ax2.plot(df["epoch"], df["val_acc"], linestyle="--", label="Val Acc")
    ax2.set_ylabel("Accuracy")

    lines = [l1, l2, l3, l4]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best")
    plt.title("CNN training dynamics: loss and accuracy")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def make_confusion(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels_order: list[str],
    out_png: str,
    report_txt: str,
    map_style: str = "auto",
) -> None:
    """
    Create row-normalized confusion matrix and write an sklearn classification report.
    """
    y_true_idx = to_index_series(y_true, labels_order, map_style=map_style)
    y_pred_idx = to_index_series(y_pred, labels_order, map_style=map_style)

    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(labels_order))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels_order)),
        yticks=np.arange(len(labels_order)),
        xticklabels=[l.capitalize() for l in labels_order],
        yticklabels=[l.capitalize() for l in labels_order],
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix (row-normalized)",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if val > 0.5 else "black",
                fontsize=10,
            )

    fig.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

    rep = classification_report(
        y_true_idx,
        y_pred_idx,
        target_names=[l.capitalize() for l in labels_order],
        digits=2,
    )
    with open(report_txt, "w") as f:
        f.write(rep)


def main():
    parser = argparse.ArgumentParser(description="Create SER report figures.")
    parser.add_argument(
        "--cnn-history",
        type=str,
        required=True,
        help="CSV with columns: epoch,train_loss,val_loss,train_acc,val_acc",
    )
    parser.add_argument(
        "--cnn-preds",
        type=str,
        required=True,
        help="CSV with columns: y_true,y_pred for CNN",
    )
    parser.add_argument(
        "--w2v2-preds",
        type=str,
        required=True,
        help="CSV with columns: y_true,y_pred for Wav2Vec2",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="angry,disgust,fear,happy,neutral,sad",
        help="Comma-separated label order for matrices (matches report).",
    )
    parser.add_argument(
        "--map-style",
        type=str,
        default="auto",
        choices=["auto", "long", "short_upper"],
        help="How to interpret string labels in CSVs.",
    )
    parser.add_argument("--outdir", type=str, default="figures", help="Output folder")
    args = parser.parse_args()

    labels_order = [t.strip().lower() for t in args.labels.split(",")]
    ensure_dir(args.outdir)

    # 1) Training curves
    plot_training_curves(args.cnn_history, os.path.join(args.outdir, "cnn_train_curve.png"))

    # 2) CNN confusion + report
    df_cnn = pd.read_csv(args.cnn_preds)
    if not {"y_true", "y_pred"}.issubset(df_cnn.columns):
        raise ValueError("cnn-preds CSV must have columns: y_true,y_pred")
    make_confusion(
        df_cnn["y_true"],
        df_cnn["y_pred"],
        labels_order,
        os.path.join(args.outdir, "cnn_confusion.png"),
        os.path.join(args.outdir, "classification_report_cnn.txt"),
        map_style=args.map_style,
    )

    # 3) Wav2Vec2 confusion + report
    df_w2v = pd.read_csv(args.w2v2_preds)
    if not {"y_true", "y_pred"}.issubset(df_w2v.columns):
        raise ValueError("w2v2-preds CSV must have columns: y_true,y_pred")
    make_confusion(
        df_w2v["y_true"],
        df_w2v["y_pred"],
        labels_order,
        os.path.join(args.outdir, "w2v2_confusion.png"),
        os.path.join(args.outdir, "classification_report_w2v2.txt"),
        map_style=args.map_style,
    )

    print(
        "Saved:\n"
        f"  - {os.path.join(args.outdir, 'cnn_train_curve.png')}\n"
        f"  - {os.path.join(args.outdir, 'cnn_confusion.png')}\n"
        f"  - {os.path.join(args.outdir, 'w2v2_confusion.png')}\n"
        f"  - classification_report_cnn.txt\n"
        f"  - classification_report_w2v2.txt\n"
        f"in: {args.outdir}/"
    )

main()