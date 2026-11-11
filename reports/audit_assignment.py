"""Review existing assignment evidence without training or changing source results.

Run: python reports/audit_assignment.py
Optional publisher check: python reports/audit_assignment.py --verify-manifest
Only files under reports/audit_evidence/ are written.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cnn_lstm.data import channel_stats, index_dataset, load_image, Normalizer, stratified_split
from cnn_lstm.metrics import compute_metrics
from cnn_lstm.models import build_model, count_parameters
from cnn_lstm.train import predict

OUT = ROOT / "reports" / "audit_evidence"
DATA = ROOT / "data" / "natural-images"
SOURCE = DATA / "natural_images"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def differences(actual, expected, prefix=""):
    """Compare every saved metric, including confusion matrices and supports."""
    errors = []
    for key in actual.keys() | expected.keys():
        name = f"{prefix}.{key}" if prefix else key
        if key not in actual or key not in expected:
            errors.append(name)
        elif isinstance(actual[key], dict) and isinstance(expected[key], dict):
            errors.extend(differences(actual[key], expected[key], name))
        elif not np.allclose(actual[key], expected[key], rtol=0, atol=1e-10):
            errors.append(name)
    return errors


def verify_manifest():
    """Compare all publisher filenames and sizes; the API provides no byte hashes."""
    import requests

    url = "https://www.kaggle.com/api/v1/datasets/list/prasunroy/natural-images"
    published, token, pages = {}, None, 0
    with requests.Session() as session:
        while True:
            params = {"pageSize": 200}
            if token:
                params["pageToken"] = token
            response = session.get(url, params=params, timeout=45)
            response.raise_for_status()
            page = response.json()
            if page.get("hasErrorMessage"):
                raise RuntimeError(page["errorMessage"])
            files = page["datasetFiles"]
            before = len(published)
            published.update({f["name"]: int(f["totalBytes"]) for f in files})
            pages += 1
            next_token = page.get("nextPageToken")
            if not next_token:
                break
            if len(published) == before or next_token == token:
                raise RuntimeError("Manifest pagination made no progress")
            token = next_token
            if pages % 10 == 0:
                print(f"Manifest: {len(published)} files checked", flush=True)
    local = {p.relative_to(DATA).as_posix(): p.stat().st_size
             for p in DATA.rglob("*") if p.is_file()}
    result = {
        "url": url, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_pages": pages, "publisher_files": len(published),
        "publisher_total_bytes": sum(published.values()), "local_files": len(local),
        "missing_local": sorted(published.keys() - local.keys()),
        "extra_local": sorted(local.keys() - published.keys()),
        "size_mismatches": [p for p in published.keys() & local.keys() if published[p] != local[p]],
        "scope": "Filenames and byte sizes only; publisher cryptographic hashes are not supplied.",
    }
    (OUT / "publisher_manifest.json").write_text(json.dumps(published, indent=2), encoding="utf-8")
    (OUT / "publisher_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def audit_local():
    torch.set_num_threads(8)
    info = json.loads((ROOT / "results/data_info.json").read_text())
    summary = json.loads((ROOT / "results/summary.json").read_text())
    index = index_dataset(SOURCE)
    train, val, test = stratified_split(index.labels, .15, .15, 42)
    assignments = {index.paths[i]: name for name, ids in (("train", train), ("val", val), ("test", test))
                   for i in ids}
    with np.load(ROOT / "artifacts/cache/images_64px.npz", allow_pickle=False) as cache:
        cache_paths = cache["paths"].tolist()
        images = cache["images"].copy()
    assert cache_paths == index.paths, "Cache path order does not match current image index"
    records, byte_groups, pixel_groups, hash_groups = [], defaultdict(list), defaultdict(list), defaultdict(list)
    cache_differences, copy_differences = [], []
    path_to_index = {p: i for i, p in enumerate(index.paths)}
    for folder in sorted(p for p in SOURCE.iterdir() if p.is_dir()):
        for path in sorted(folder.iterdir()):
            rel = path.relative_to(SOURCE).as_posix()
            raw = path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            byte_groups[sha].append(rel)
            copy = DATA / "data" / "natural_images" / rel
            if not copy.is_file() or copy.read_bytes() != raw:
                copy_differences.append(rel)
            record = {"path": rel, "bytes": len(raw), "sha256": sha}
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.load()
                    record.update(format=im.format, mode=im.mode, width=im.width, height=im.height)
                    rgb = im.convert("RGB")
                    pixkey = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                    pixel_groups[pixkey].append(rel)
                    small = np.asarray(im.convert("L").resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float32)
                    ahash = np.packbits((small > small.mean()).ravel()).tobytes().hex()
                    hash_groups[ahash].append(rel)
                    record["ahash"] = ahash
                if rel in path_to_index and not np.array_equal(load_image(path, 64), images[path_to_index[rel]]):
                    cache_differences.append(rel)
            except Exception as exc:
                record["error"] = repr(exc)
            records.append(record)
    print(f"Decoded {len(records)} raw images; checked every cached image against fresh resize", flush=True)
    mean, std = channel_stats(images[train])
    def crossing(groups, left, right):
        pairs = []
        for members in groups.values():
            a = [p for p in members if assignments.get(p) == left]
            b = [p for p in members if assignments.get(p) == right]
            if a and b:
                pairs.append({left: a, right: b})
        return pairs
    crossings = {}
    for method, groups in (("byte_sha256", byte_groups), ("decoded_rgb_sha256", pixel_groups), ("average_hash_8x8", hash_groups)):
        crossings[method] = {f"{a}_vs_{b}": crossing(groups, a, b)
                             for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    checked = []
    classes = info["classes"]
    y = index.labels[test]
    for path in sorted((ROOT / "results/runs").glob("*/result.json")):
        run = json.loads(path.read_text())
        probs = np.load(path.parent / "test_probs.npy", allow_pickle=False)
        assert probs.shape == (len(test), len(classes))
        actual = compute_metrics(y, probs, classes)
        history = run["history"]
        best_loss, expected_epoch = np.inf, None
        for epoch, loss in enumerate(history["val_loss"], 1):
            if loss < best_loss - 1e-4:
                best_loss, expected_epoch = loss, epoch
        model = build_model(run["arch"], len(classes), image_size=64, **run["model"])
        # Validation metrics can be checked internally against the saved confusion matrix.
        cm = np.asarray(run["val"]["confusion_matrix"])
        truth = np.repeat(np.arange(len(classes)), cm.sum(1))
        pred = np.concatenate([np.repeat(np.arange(len(classes)), row) for row in cm])
        from_cm = compute_metrics(truth, np.eye(len(classes))[pred], classes)
        from_cm.pop("top2_accuracy")  # A confusion matrix cannot recover top-2 accuracy.
        saved_val = {k: v for k, v in run["val"].items() if k != "top2_accuracy"}
        checked.append({
            "name": run["name"], "test_metric_mismatches": differences(actual, run["test"]),
            "test_probabilities_finite": bool(np.isfinite(probs).all()),
            "test_probabilities_in_range": bool(((probs >= 0) & (probs <= 1)).all()),
            "max_probability_sum_error": float(np.max(np.abs(probs.sum(1) - 1))),
            "validation_confusion_metric_mismatches": differences(from_cm, saved_val),
            "validation_support_matches_split": cm.sum(1).tolist() == info["class_counts"]["val"],
            "params": count_parameters(model), "saved_params_match": count_parameters(model) == run["params"],
            "history_lengths_match": all(len(history[k]) == run["epochs_run"] for k in
                                          ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_seconds")),
            "best_epoch_matches_loss_rule": expected_epoch == history["best_epoch"],
            "val_accuracy_matches_best_epoch": bool(np.isclose(run["val"]["accuracy"], history["val_acc"][history["best_epoch"] - 1])),
            "test_accuracy": actual["accuracy"], "test_correct": int((probs.argmax(1) == y).sum()),
            "test_macro_f1": actual["macro_f1"], "train_seconds": run["train_seconds"],
            "summed_epoch_seconds": sum(history["epoch_seconds"]),
            "median_epoch_seconds": float(np.median(history["epoch_seconds"])),
        })
    print("Recomputed test metrics for all saved experiments", flush=True)
    meta = json.loads((ROOT / "artifacts/checkpoints/best_model.json").read_text())
    model = build_model(meta["arch"], len(classes), image_size=64, **meta["model"])
    ckpt = ROOT / "artifacts/checkpoints/best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    norm = Normalizer(mean, std)
    selected = json.loads((ROOT / "results/runs" / meta["name"] / "result.json").read_text())
    checkpoints = {}
    for name, ids in (("val", val), ("test", test)):
        probs = predict(model, images[ids], norm)
        metrics = compute_metrics(index.labels[ids], probs, classes)
        checkpoints[name] = {"metrics": metrics, "saved_metric_mismatches": differences(metrics, selected[name])}
        if name == "test":
            saved_probs = np.load(ROOT / "results/runs" / meta["name"] / "test_probs.npy")
            checkpoints[name]["max_probability_difference"] = float(np.max(np.abs(probs - saved_probs)))
            checkpoints[name]["different_predicted_labels"] = int((probs.argmax(1) != saved_probs.argmax(1)).sum())
    print("Reloaded selected checkpoint and independently evaluated validation and test", flush=True)
    ranked = sorted(summary["runs"], key=lambda r: (-r["val_macro_f1"], -r["val_acc"], r["params"]))
    notebook = json.loads((ROOT / "notebooks/LSTM_Image_Classification.ipynb").read_text(encoding="utf-8"))
    all_files = [p for p in DATA.rglob("*") if p.is_file()]
    valid = [r for r in records if "error" not in r]
    evidence_paths = [p for folder in ("cnn_lstm", "results/runs", "artifacts/checkpoints")
                      for p in (ROOT / folder).rglob("*")
                      if p.is_file() and p.suffix in {".py", ".json", ".pt", ".npy"}]
    report = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": {n: importlib.metadata.version(n) for n in
                     ("torch", "numpy", "scikit-learn", "pillow", "matplotlib", "pandas", "kagglehub", "pytest", "python-docx")},
        "raw": {"all_files": len(all_files), "total_bytes": sum(p.stat().st_size for p in all_files),
                "images_in_primary_copy": len(records), "class_counts": dict(Counter(p["path"].split("/")[0] for p in records)),
                "decode_errors": [r for r in records if "error" in r],
                "formats": dict(Counter(r["format"] for r in valid)), "modes": dict(Counter(r["mode"] for r in valid)),
                "smaller_than_64": sum(r["width"] < 64 or r["height"] < 64 for r in valid),
                "extreme_aspect_ratios": sum(not .5 <= r["width"] / r["height"] <= 2 for r in valid),
                "duplicate_copy_mismatches": copy_differences,
                "byte_duplicate_groups": [v for v in byte_groups.values() if len(v) > 1],
                "decoded_rgb_duplicate_groups": [v for v in pixel_groups.values() if len(v) > 1]},
        "preprocessing": {"indexed_images": len(index.paths), "exact_duplicates_removed": index.duplicates_removed,
                          "cache_mismatches_against_fresh_images": cache_differences,
                          "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
                          "test_paths_match_saved_order": [index.paths[i] for i in test] == info["test_paths"],
                          "channel_mean": mean.tolist(), "channel_std": std.tolist(),
                          "normalization_matches_saved": bool(np.array_equal(mean, info["channel_mean"]) and np.array_equal(std, info["channel_std"]))},
        "cross_split_duplicate_candidates": crossings,
        "experiments": checked, "selected_checkpoint": {"name": meta["name"], "sha256": digest(ckpt), **checkpoints},
        "summary_ranking_matches_selection_rule": ranked == summary["runs"] and summary["selected"] == ranked[0]["name"],
        "notebook": {"code_cells": sum(c["cell_type"] == "code" for c in notebook["cells"]),
                     "executed_code_cells": sum(c.get("execution_count") is not None for c in notebook["cells"]),
                     "placeholder_cells": [i for i, c in enumerate(notebook["cells"]) if "filled in after execution" in "".join(c["source"])],
                     "error_outputs": [i for i, c in enumerate(notebook["cells"]) if any(o.get("output_type") == "error" for o in c.get("outputs", []))]},
        "source_evidence_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in evidence_paths},
        "scope": "Existing evidence only; no retraining. Other runs' original weights and validation probabilities are unavailable. Average-hash collisions require visual review. Local hashes identify this audit's inputs and do not establish publisher identity.",
    }
    (OUT / "local_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "image_inventory.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    # Contact sheet for candidate train/test overlaps, not an automated duplicate verdict.
    pairs = []
    for group in crossings["average_hash_8x8"]["train_vs_test"]:
        pairs.append((group["train"][0], group["test"][0]))
    if pairs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pairs = pairs[:10] + [p for p in pairs[10:] if p[0].startswith("motorbike/")][:2]
        fig, axes = plt.subplots(len(pairs), 2, figsize=(7, 2 * len(pairs)), squeeze=False)
        for row, (left, right) in enumerate(pairs):
            for ax, rel, split in zip(axes[row], (left, right), ("train", "test")):
                with Image.open(SOURCE / rel) as im:
                    ax.imshow(im.convert("RGB"))
                ax.set_title(f"{split}: {rel}", fontsize=9)
                ax.axis("off")
        fig.suptitle("Identical 8x8 average hashes: candidates for visual review")
        fig.tight_layout(rect=(0, 0, 1, .985))
        fig.savefig(OUT / "candidate_train_test_pairs.png", dpi=100)
        plt.close(fig)
    print(f"Wrote {OUT / 'local_audit.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-manifest", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.verify_manifest:
        verify_manifest()
    else:
        audit_local()


if __name__ == "__main__":
    main()
