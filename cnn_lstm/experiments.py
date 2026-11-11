"""Run every experiment, save histories, metrics and checkpoints.

Usage::

    python -m cnn_lstm.experiments                 # all experiments (skips finished ones)
    python -m cnn_lstm.experiments --only cnn_bilstm --epochs 2   # smoke run
    python -m cnn_lstm.experiments --rerun         # ignore cached results

Model selection uses the validation split only. Test metrics are recorded for every run
so the architectures can be compared, but no choice is made from them.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .data import Normalizer, Split, channel_stats, index_dataset, load_or_build_cache, stratified_split
from .metrics import compute_metrics
from .models import build_model, count_parameters, describe
from .train import TrainConfig, fit, predict, set_seed

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "natural-images" / "natural_images"
CACHE_DIR = ROOT / "artifacts" / "cache"
CKPT_DIR = ROOT / "artifacts" / "checkpoints"
RESULTS_DIR = ROOT / "results"

IMAGE_SIZE = 64
SPLIT_SEED = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15


@dataclass
class Experiment:
    name: str
    group: str  # "architecture" or "tuning"
    description: str
    arch: str
    model: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)


BASE = dict(hidden=128, layers=1, bidirectional=True, dropout=0.5)

EXPERIMENTS: list[Experiment] = [
    # Architecture comparison
    Experiment("pixel_lstm", "architecture", "LSTM only, 64 rows of raw pixels", "pixel_lstm",
               dict(hidden=128, layers=2, dropout=0.3)),
    Experiment("cnn_only", "architecture", "CNN + global average pooling, no LSTM", "cnn",
               dict(dropout=0.5)),
    Experiment("cnn_lstm", "architecture", "CNN + unidirectional LSTM", "cnn_lstm",
               {**BASE, "bidirectional": False}),
    Experiment("cnn_bilstm", "architecture", "CNN + bidirectional LSTM (baseline)", "cnn_lstm", BASE),
    # Five single-factor changes, followed by one combined ablation of three settings.
    Experiment("tune_hidden256", "tuning", "LSTM hidden size 256", "cnn_lstm", {**BASE, "hidden": 256}),
    Experiment("tune_layers2", "tuning", "Two stacked LSTM layers", "cnn_lstm", {**BASE, "layers": 2}),
    Experiment("tune_dropout03", "tuning", "Dropout 0.3", "cnn_lstm", {**BASE, "dropout": 0.3}),
    Experiment("tune_lr3e4", "tuning", "Learning rate 3e-4", "cnn_lstm", BASE, dict(lr=3e-4)),
    Experiment("tune_no_aug", "tuning", "No data augmentation", "cnn_lstm", BASE, dict(augment=False)),
    Experiment("tune_no_reg", "tuning", "No augmentation, dropout or weight decay", "cnn_lstm",
               {**BASE, "dropout": 0.0}, dict(augment=False, weight_decay=0.0)),
]


def prepare_data(log=print):
    t0 = time.perf_counter()
    index = index_dataset(DATA_ROOT)
    images = load_or_build_cache(DATA_ROOT, index, IMAGE_SIZE, CACHE_DIR)
    tr, va, te = stratified_split(index.labels, VAL_SIZE, TEST_SIZE, SPLIT_SEED)
    splits = {n: Split(images[i], index.labels[i]) for n, i in (("train", tr), ("val", va), ("test", te))}
    mean, std = channel_stats(splits["train"].x)
    info = {
        "classes": index.classes,
        "num_images": len(index.paths),
        "exact_duplicates_removed": index.duplicates_removed,
        "image_size": IMAGE_SIZE,
        "split_seed": SPLIT_SEED,
        "split_sizes": {n: int(len(s.y)) for n, s in splits.items()},
        "class_counts": {
            n: np.bincount(s.y, minlength=len(index.classes)).tolist() for n, s in splits.items()
        },
        "channel_mean": mean.tolist(),
        "channel_std": std.tolist(),
        "test_paths": [index.paths[i] for i in te],
    }
    log(f"data ready in {time.perf_counter() - t0:.1f}s: {info['split_sizes']}, "
        f"{index.duplicates_removed} exact duplicates removed")
    return splits, Normalizer(mean, std), info


def save_if_best(model: torch.nn.Module, result: dict) -> bool:
    """Keep one checkpoint: the run with the best validation macro-F1 so far.

    Disk on the training machine is scarce, so per-run checkpoints are not kept;
    every run can be reproduced from its config and seed.
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    meta_file = CKPT_DIR / "best_model.json"
    score = result["val"]["macro_f1"]
    if meta_file.exists() and json.loads(meta_file.read_text())["val_macro_f1"] >= score:
        return False
    torch.save(model.state_dict(), CKPT_DIR / "best_model.pt")
    meta = {k: result[k] for k in ("name", "arch", "model", "params")}
    meta_file.write_text(json.dumps({**meta, "val_macro_f1": score}, indent=2))
    return True


def run_experiment(exp: Experiment, splits, norm, classes, epochs: int | None, log=print) -> dict:
    out_dir = RESULTS_DIR / "runs" / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(**exp.train)
    if epochs is not None:
        cfg.epochs = epochs
    set_seed(cfg.seed)
    model = build_model(exp.arch, len(classes), image_size=IMAGE_SIZE, **exp.model)
    log(f"\n== {exp.name}: {exp.description} | {count_parameters(model):,} params | {cfg}")

    t0 = time.perf_counter()
    hist = fit(model, splits["train"], splits["val"], norm, cfg, log=log)
    train_seconds = time.perf_counter() - t0

    val_probs = predict(model, splits["val"].x, norm)
    test_probs = predict(model, splits["test"].x, norm)
    result = {
        "name": exp.name,
        "group": exp.group,
        "description": exp.description,
        "arch": exp.arch,
        "model": {k: list(v) if isinstance(v, tuple) else v for k, v in exp.model.items()},
        "train": asdict(cfg),
        "params": count_parameters(model),
        "train_seconds": train_seconds,
        "epochs_run": len(hist.train_loss),
        "history": asdict(hist),
        "val": compute_metrics(splits["val"].y, val_probs, classes),
        "test": compute_metrics(splits["test"].y, test_probs, classes),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    np.save(out_dir / "test_probs.npy", test_probs.astype(np.float32))
    save_if_best(model, result)
    (out_dir / "model_summary.txt").write_text(describe(model, IMAGE_SIZE))
    log(f"  -> val acc {result['val']['accuracy']:.4f} f1 {result['val']['macro_f1']:.4f} | "
        f"test acc {result['test']['accuracy']:.4f} | {train_seconds / 60:.1f} min")
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="experiment names to run")
    ap.add_argument("--epochs", type=int, help="override epochs (smoke tests)")
    ap.add_argument("--rerun", action="store_true", help="retrain even if results exist")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    RESULTS_DIR.mkdir(exist_ok=True)
    log_file = (RESULTS_DIR / "training_log.txt").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    splits, norm, info = prepare_data(log)
    info["environment"] = {
        "package_version": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": platform.processor(),
        "threads": torch.get_num_threads(),
    }
    (RESULTS_DIR / "data_info.json").write_text(json.dumps(info, indent=2))

    chosen = [e for e in EXPERIMENTS if not args.only or e.name in args.only]
    unknown = set(args.only or []) - {e.name for e in EXPERIMENTS}
    if unknown:
        raise SystemExit(f"unknown experiments: {sorted(unknown)}")
    for exp in chosen:
        if (RESULTS_DIR / "runs" / exp.name / "result.json").exists() and not args.rerun:
            log(f"skip {exp.name} (already done)")
            continue
        run_experiment(exp, splits, norm, info["classes"], args.epochs, log)

    write_summary(log)
    log_file.close()


def write_summary(log=print) -> dict | None:
    """Rank finished runs by validation macro-F1 and record the selected model."""
    results = [json.loads(p.read_text()) for p in sorted((RESULTS_DIR / "runs").glob("*/result.json"))]
    if not results:
        return None
    rows = sorted(
        (
            {
                "name": r["name"],
                "group": r["group"],
                "description": r["description"],
                "params": r["params"],
                "epochs_run": r["epochs_run"],
                "best_epoch": r["history"]["best_epoch"],
                "train_minutes": round(r["train_seconds"] / 60, 2),
                "final_train_acc": r["history"]["train_acc"][r["history"]["best_epoch"] - 1],
                "val_acc": r["val"]["accuracy"],
                "val_macro_f1": r["val"]["macro_f1"],
                "test_acc": r["test"]["accuracy"],
                "test_macro_precision": r["test"]["macro_precision"],
                "test_macro_recall": r["test"]["macro_recall"],
                "test_macro_f1": r["test"]["macro_f1"],
            }
            for r in results
        ),
        key=lambda row: (-row["val_macro_f1"], -row["val_acc"], row["params"]),
    )
    summary = {"selected_by": "validation macro-F1", "selected": rows[0]["name"], "runs": rows}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log("\nrank  name              val_acc  val_f1   test_acc  test_f1  params")
    for i, r in enumerate(rows, 1):
        log(f"{i:>4}  {r['name']:<16}  {r['val_acc']:.4f}   {r['val_macro_f1']:.4f}   "
            f"{r['test_acc']:.4f}    {r['test_macro_f1']:.4f}   {r['params']:,}")
    return summary


if __name__ == "__main__":
    main()
