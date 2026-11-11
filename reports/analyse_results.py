"""Recompute report statistics from saved predictions without fitting any model."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cnn_lstm.data import index_dataset  # noqa: E402
OUT = ROOT / "reports" / "analysis_results.json"


def wilson_interval(correct: int, total: int, confidence: float = 0.95) -> list[float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return [center - radius, center + radius]


def macro_f1(y: np.ndarray, predicted: np.ndarray, classes: int) -> float:
    cm = np.bincount(y * classes + predicted, minlength=classes**2).reshape(classes, classes)
    denominator = cm.sum(0) + cm.sum(1)
    return float(np.divide(2 * np.diag(cm), denominator,
                           out=np.zeros(classes), where=denominator > 0).mean())


def analyse(resamples: int = 2000, seed: int = 42) -> dict:
    info = json.loads((ROOT / "results/data_info.json").read_text())
    summary = json.loads((ROOT / "results/summary.json").read_text())
    classes = info["classes"]
    labels = {name: i for i, name in enumerate(classes)}
    y = np.array([labels[p.split("/")[0]] for p in info["test_paths"]])
    selected = summary["selected"]
    selected_probs = np.load(ROOT / f"results/runs/{selected}/test_probs.npy")
    predictions = selected_probs.argmax(1)
    baseline = np.load(ROOT / "results/runs/cnn_bilstm/test_probs.npy").argmax(1)
    correct = predictions == y

    # Class-stratified resampling fixes each class's observed support. The same
    # sampled images are used for the selected/baseline accuracy comparison.
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == k) for k in range(len(classes))]
    f1_samples, improvement_samples = [], []
    for _ in range(resamples):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        f1_samples.append(macro_f1(y[indices], predictions[indices], len(classes)))
        improvement_samples.append(float((predictions[indices] == y[indices]).mean()
                                         - (baseline[indices] == y[indices]).mean()))

    # Candidate flags are a descriptive sensitivity check, not a new test set
    # and not a finding that every matching hash represents a duplicate.
    inventory = json.loads((ROOT / "reports/audit_evidence/image_inventory.json").read_text())
    hash_of = {row["path"]: row["ahash"] for row in inventory}
    test_paths = set(info["test_paths"])
    index = index_dataset(ROOT / "data/natural-images/natural_images")
    other_paths = set(index.paths) - test_paths
    other_hashes = {row["ahash"] for row in inventory if row["path"] in other_paths}
    flagged = np.array([hash_of[p] in other_hashes for p in info["test_paths"]])
    sensitivity = {}
    for name, mask in (("hash_match_candidates", flagged), ("remaining_test_images", ~flagged)):
        sensitivity[name] = {
            "images": int(mask.sum()), "correct": int(correct[mask].sum()),
            "accuracy": float(correct[mask].mean()),
            "macro_f1": macro_f1(y[mask], predictions[mask], len(classes)),
        }
    # Macro F1 is not comparable across subsets missing classes.
    sensitivity["hash_match_candidates"].pop("macro_f1")

    all_runs = [json.loads(p.read_text()) for p in (ROOT / "results/runs").glob("*/result.json")]
    inputs = [ROOT / "results/data_info.json", ROOT / "results/summary.json",
              ROOT / f"results/runs/{selected}/test_probs.npy",
              ROOT / "results/runs/cnn_bilstm/test_probs.npy",
              ROOT / "reports/audit_evidence/image_inventory.json"]
    return {
        "selected": selected, "test_images": len(y), "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "accuracy_wilson_95": wilson_interval(int(correct.sum()), len(y)),
        "macro_f1": macro_f1(y, predictions, len(classes)),
        "macro_f1_bootstrap_95": np.quantile(f1_samples, [0.025, 0.975]).tolist(),
        "bootstrap": {"resamples": resamples, "seed": seed,
                      "method": "Class-stratified percentile bootstrap; class supports fixed."},
        "selected_minus_baseline_accuracy": float(correct.mean() - (baseline == y).mean()),
        "paired_accuracy_improvement_bootstrap_95": np.quantile(improvement_samples, [0.025, 0.975]).tolist(),
        "selected_correct_baseline_wrong": int(np.sum(correct & (baseline != y))),
        "selected_wrong_baseline_correct": int(np.sum(~correct & (baseline == y))),
        "sensitivity": sensitivity,
        "recorded_total_elapsed_hours": sum(r["train_seconds"] for r in all_runs) / 3600,
        "median_epoch_seconds": {r["name"]: float(np.median(r["history"]["epoch_seconds"])) for r in all_runs},
        "scope": "Post hoc descriptive analysis of the original image split. Intervals condition on the fitted "
                 "models and assume independent sampled images; they exclude retraining, seed, selection, "
                 "distribution-shift and related-image uncertainty. Hash-filtered results do not correct leakage.",
        "input_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    }


if __name__ == "__main__":
    result = analyse()
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "input_sha256"}, indent=2))
