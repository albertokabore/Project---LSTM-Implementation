"""Generate report figures from saved results.

Usage::

    python -m cnn_lstm.plots

Reads ``results/`` (written by ``cnn_lstm.experiments``) and writes PNGs to
``results/figures``. Never touches the dataset folder.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator, PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .data import Augmenter, index_dataset, load_image, stratified_split  # noqa: E402
from .experiments import DATA_ROOT, IMAGE_SIZE, RESULTS_DIR  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"
BLUE, ORANGE, TEAL, PURPLE = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
GREY = "#6B7280"
HEATMAP = LinearSegmentedColormap.from_list("report_blue", ["#F1F7FD", "#6BAED6", "#08519C"])
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10,
                     "text.color": "black", "axes.labelcolor": "black", "axes.titlecolor": "black",
                     "xtick.color": "black", "ytick.color": "black"})


def _load():
    info = json.loads((RESULTS_DIR / "data_info.json").read_text())
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    runs = {p.parent.name: json.loads(p.read_text()) for p in (RESULTS_DIR / "runs").glob("*/result.json")}
    return info, summary, runs


def class_distribution(info):
    classes = info["classes"]
    counts = info["class_counts"]
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bottom = np.zeros(len(classes))
    for split, color, label in (("train", BLUE, "Training"), ("val", ORANGE, "Validation"),
                                ("test", TEAL, "Test")):
        vals = np.array(counts[split])
        ax.bar(x, vals, bottom=bottom, color=color, edgecolor="white", linewidth=0.5, label=label)
        bottom += vals
    ax.set_xticks(x, classes)
    ax.set_ylabel("Images")
    ax.set_title("Images per class and split (stratified 70/15/15)")
    ax.set_ylim(0, bottom.max() * 1.18)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png")
    plt.close(fig)


def augmentation_examples(info, n=6):
    index = index_dataset(DATA_ROOT)
    training, _, _ = stratified_split(index.labels, .15, .15, info["split_seed"])
    # Illustrate training transforms using training images, preserving the holdout.
    paths = [index.paths[i] for i in training[::max(1, len(training) // n)][:n]]
    imgs = np.stack([load_image(DATA_ROOT / p, IMAGE_SIZE) for p in paths])
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255
    aug = Augmenter(seed=3)
    rows = [x] + [aug(x) for _ in range(3)]
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.3, len(rows) * 1.4))
    for r, batch in enumerate(rows):
        for c in range(n):
            axes[r, c].imshow(batch[c].permute(1, 2, 0).numpy())
            axes[r, c].axis("off")
        axes[r, 0].set_title("original" if r == 0 else f"augmented #{r}", fontsize=8, loc="left")
    fig.suptitle("Training-time augmentation: flip, shift, brightness/contrast", fontsize=10)
    fig.savefig(FIG_DIR / "augmentation_examples.png")
    plt.close(fig)


def training_curves(run, fname):
    h = run["history"]
    ep = np.arange(1, len(h["train_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, label in ((a1, "loss", "Cross-entropy loss"), (a2, "acc", "Accuracy")):
        ax.plot(ep, h[f"train_{key}"], color=BLUE, ls="--", lw=2, label="Training")
        ax.plot(ep, h[f"val_{key}"], color=ORANGE, lw=2, label="Validation")
        ax.axvline(h["best_epoch"], color=GREY, ls=":", lw=1.2)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#CBD5E1", alpha=0.45, linewidth=0.6)
        if key == "acc":
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend(frameon=False)
    a1.set_title(f"{run['name']}: best epoch {h['best_epoch']} (dotted)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)


def confusion(run, classes, fname):
    cm = np.array(run["test"]["confusion_matrix"])
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    im = ax.imshow(norm, cmap=HEATMAP, vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if norm[i, j] > 0.55 else "black", fontsize=9)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Test confusion matrix: {run['name']}\nCounts shown; shading = fraction of true class",
                 fontsize=10)
    ax.set_xticks(np.arange(len(classes) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(classes) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Fraction of true class")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)


def experiment_comparison(summary):
    rows = summary["runs"]
    names = [r["name"] for r in rows][::-1]
    val = [r["val_acc"] for r in rows][::-1]
    test = [r["test_acc"] for r in rows][::-1]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(names) + 1))
    ax.barh(y + 0.2, val, 0.4, color=ORANGE, label="Validation")
    ax.barh(y - 0.2, test, 0.4, color=TEAL, label="Test")
    for offset, values in ((0.2, val), (-0.2, test)):
        for yi, value in zip(y, values):
            ax.text(value + 0.003, yi + offset, f"{value:.1%}", va="center", fontsize=8)
    ax.set_yticks(y, names)
    ax.set_xlim(min(test + val) - 0.05, 1.0)
    ax.set_xlabel("Accuracy")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("All runs, ranked by validation macro-F1 (selection metric)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "experiment_comparison.png")
    plt.close(fig)


def architecture_curves(runs):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    styles = {"pixel_lstm": (PURPLE, ":"), "cnn_only": (TEAL, "--"), "cnn_lstm": (ORANGE, "-"),
              "cnn_bilstm": (BLUE, "-.")}
    for name, (color, ls) in styles.items():
        if name in runs:
            v = runs[name]["history"]["val_acc"]
            ax.plot(range(1, len(v) + 1), v, color=color, ls=ls, lw=2, label=name)
    ax.set_xlabel("Epoch")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.set_ylabel("Validation accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color="#CBD5E1", alpha=0.45, linewidth=0.6)
    ax.set_title("Architecture comparison: validation accuracy per epoch")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "architecture_curves.png")
    plt.close(fig)


def sample_predictions(run, info, n=16):
    probs = np.load(RESULTS_DIR / "runs" / run["name"] / "test_probs.npy")
    classes = info["classes"]
    paths = info["test_paths"]
    label_of = {c: i for i, c in enumerate(classes)}
    y_true = np.array([label_of[p.split("/")[0]] for p in paths])
    pred = probs.argmax(1)
    wrong = np.flatnonzero(pred != y_true)
    right = np.flatnonzero(pred == y_true)
    rng = np.random.default_rng(0)
    pick = np.concatenate([rng.choice(right, n // 2, replace=False),
                           rng.choice(wrong, min(n // 2, len(wrong)), replace=False)])
    cols = 8
    fig, axes = plt.subplots(int(np.ceil(len(pick) / cols)), cols, figsize=(cols * 1.5, 4))
    for ax, i in zip(axes.flat, pick):
        ax.imshow(load_image(DATA_ROOT / paths[i], 96))
        ok = pred[i] == y_true[i]
        ax.set_title(f"{classes[pred[i]]} {probs[i].max():.0%}\n(true: {classes[y_true[i]]})",
                     fontsize=7, color="black")
        ax.axis("off")
    for ax in list(axes.flat)[len(pick):]:
        ax.axis("off")
    fig.suptitle("Test predictions: correct classifications (top), errors (bottom)", fontsize=10)
    fig.savefig(FIG_DIR / "sample_predictions.png")
    plt.close(fig)


def architecture_diagram():
    """Block diagram of the CNN-BiLSTM (snake layout: CNN on top, sequence model below)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    boxes = [
        ("Input image", "64 x 64 x 3 RGB", "standardised"),
        ("Conv block 1", "2 x (conv3x3, BN, ReLU)", "pool -> 32 x 32 x 32"),
        ("Conv block 2", "64 filters", "pool -> 16 x 16 x 64"),
        ("Conv block 3", "128 filters", "pool -> 8 x 8 x 128"),
        ("Rows to sequence", "8 rows = 8 time steps", "1,024 values each"),
        ("Bidirectional LSTM", "128 units each way", "top-down + bottom-up"),
        ("Classifier head", "256 -> 128, BN, ReLU", "dropout: base 0.5; selected 0"),
        ("Prediction", "8 class scores", "softmax"),
    ]
    w, h, gap = 2.2, 1.1, 0.5
    xs = [i * (w + gap) for i in range(4)]
    pos = [(x, 2.0) for x in xs] + [(x, 0.0) for x in reversed(xs)]
    fills = ["#EAF5FB", "#D9EEF9", "#D9EEF9", "#D9EEF9", "#FFF0DB", "#F4E5F0", "#DFF3EB", "#DFF3EB"]
    edges = [BLUE, BLUE, BLUE, BLUE, ORANGE, PURPLE, TEAL, TEAL]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for i, ((name, l1, l2), (x, y)) in enumerate(zip(boxes, pos)):
        lstm = i == 5
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fills[i], ec=edges[i],
                                    lw=2 if lstm else 1.2))
        ax.text(x + w / 2, y + h * 0.72, name, ha="center", va="center", fontsize=10, weight="bold")
        ax.text(x + w / 2, y + h * 0.42, l1, ha="center", va="center", fontsize=8.5, color="black")
        ax.text(x + w / 2, y + h * 0.17, l2, ha="center", va="center", fontsize=8.5, color="black")
    arrows = [((xs[i] + w, 2.0 + h / 2), (xs[i + 1], 2.0 + h / 2)) for i in range(3)]
    arrows.append(((xs[3] + w / 2, 2.0), (xs[3] + w / 2, h)))
    arrows += [((xs[i + 1], h / 2), (xs[i] + w, h / 2)) for i in reversed(range(3))]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=GREY, lw=1.2))
    ax.text(xs[3] + w / 2 - 0.1, 1.55, "split into rows", ha="right", fontsize=8.5, color="black")
    ax.set_xlim(-0.1, xs[3] + w + 0.1)
    ax.set_ylim(-0.1, 3.2)
    ax.axis("off")
    ax.set_title("CNN-BiLSTM: convolutional features followed by recurrence over feature-map rows",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "architecture_diagram.png")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    architecture_diagram()
    info, summary, runs = _load()
    best = runs[summary["selected"]]
    class_distribution(info)
    augmentation_examples(info)
    training_curves(best, "training_curves_selected.png")
    if "tune_no_reg" in runs:
        training_curves(runs["cnn_bilstm"], "training_curves_baseline.png")
    confusion(best, info["classes"], "confusion_matrix_selected.png")
    experiment_comparison(summary)
    architecture_curves(runs)
    sample_predictions(best, info)
    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
