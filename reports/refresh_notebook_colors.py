"""Refresh notebook plot sources and images from saved results, without training.

    python reports/refresh_notebook_colors.py

Markdown, non-plot code, numerical outputs and photographic examples are retained.
"""

from __future__ import annotations

import base64
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from reports.complete_notebook import color_figure_style

NOTEBOOK = ROOT / "notebooks/LSTM_Image_Classification.ipynb"
EVIDENCE = ROOT / "reports/audit_evidence"
PLOT_CELLS = ("ee650eab", "21e32130", "e12d967b", "dcd7e3fc", "c0e8f60a")
SOURCE_CELLS = {"2727d6cb", "ee650eab", "21e32130", "e12d967b", "dcd7e3fc"}


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def image_output(cell):
    outputs = [out for out in cell.outputs if "image/png" in out.get("data", {})]
    if len(outputs) != 1:
        raise ValueError(f"Expected one figure in cell {cell.id}, found {len(outputs)}")
    return outputs[0]


def refresh() -> None:
    original = nbformat.read(NOTEBOOK, as_version=4)
    notebook = deepcopy(original)
    cells = {cell.id: cell for cell in notebook.cells}
    protected = set((ROOT / "results").rglob("*.json"))
    protected.update((ROOT / "results").rglob("*.npy"))
    protected.update((ROOT / "results/figures").glob("*.png"))
    protected.update((ROOT / "artifacts/checkpoints").glob("*"))
    protected.update((ROOT / "cnn_lstm" / name) for name in
                     ("data.py", "models.py", "train.py", "experiments.py", "metrics.py"))
    protected.update(ROOT / name for name in
                     ("reports/report_text.json", "reports/references.json", "reports/analysis_results.json",
                      "reports/LSTM_Project_Report_Color.docx"))
    protected = {path for path in protected if path.is_file()}
    hashes = {path: digest(path) for path in protected}
    notebook_before = digest(NOTEBOOK)

    color_figure_style(notebook)
    repeated = deepcopy(notebook)
    color_figure_style(repeated)
    assert repeated == notebook, "Style application must be idempotent"

    namespace = {"__name__": "__notebook_color_preview__"}
    with redirect_stdout(StringIO()):
        exec(compile(cells["2727d6cb"].source, "notebook-setup", "exec"), namespace)
    assert namespace["RUN_TRAINING"] is False
    info = json.loads((ROOT / "results/data_info.json").read_text())
    summary = json.loads((ROOT / "results/summary.json").read_text())
    runs = {path.parent.name: json.loads(path.read_text())
            for path in (ROOT / "results/runs").glob("*/result.json")}
    best = runs[summary["selected"]]
    probs = np.load(ROOT / f"results/runs/{best['name']}/test_probs.npy")
    labels = {name: i for i, name in enumerate(info["classes"])}
    y = np.array([labels[path.split("/")[0]] for path in info["test_paths"]])
    metrics = namespace["compute_metrics"](y, probs, info["classes"])
    assert metrics == best["test"]

    # Read the original displayed EDA counts, preserving its pre-deduplication scope.
    eda_html = next(out["data"]["text/html"] for out in cells["b75ec5d7"].outputs
                    if "text/html" in out.get("data", {}))
    eda = pd.read_html(StringIO(eda_html), index_col=0)[0]
    if isinstance(eda.columns, pd.MultiIndex):
        eda.columns = eda.columns.get_level_values(0)
    assert int(eda["images"].sum()) == 6899
    namespace.update(info=info, summary=summary, runs=runs, best=best, m=metrics, eda=eda)

    images = {}
    for cell_id in PLOT_CELLS:
        plt.close("all")
        exec(compile(cells[cell_id].source, f"notebook-{cell_id}", "exec"), namespace)
        assert len(plt.get_fignums()) == 1, cell_id
        fig = plt.gcf()
        if cell_id == "e12d967b":
            cm = np.asarray(metrics["confusion_matrix"])
            np.testing.assert_array_equal(fig.axes[0].images[0].get_array(),
                                          cm / cm.sum(axis=1, keepdims=True))
        elif cell_id in {"21e32130", "c0e8f60a"}:
            expected_runs = [best] if cell_id == "21e32130" else [runs["cnn_bilstm"], best]
            for run_index, run in enumerate(expected_runs):
                for axis_index, metric in enumerate(("loss", "acc")):
                    ax = fig.axes[run_index * 2 + axis_index]
                    for line, split in zip(ax.lines[:2], ("train", "val")):
                        np.testing.assert_array_equal(line.get_ydata(), run["history"][f"{split}_{metric}"])
        elif cell_id == "dcd7e3fc":
            for line in fig.axes[0].lines:
                np.testing.assert_array_equal(line.get_ydata(), runs[line.get_label()]["history"]["val_acc"])
        elif cell_id == "ee650eab":
            np.testing.assert_array_equal(fig.axes[0].containers[0].datavalues,
                                          eda["images"].sort_values().to_numpy())
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight", facecolor="white")
        output = image_output(cells[cell_id])
        output.data["image/png"] = base64.b64encode(buffer.getvalue()).decode("ascii")
        output.data["text/plain"] = repr(fig)
        images[cell_id] = buffer.getvalue()
    plt.close("all")

    diagram = (ROOT / "results/figures/architecture_diagram.png").read_bytes()
    image_output(cells["model-diagram"]).data["image/png"] = base64.b64encode(diagram).decode("ascii")
    images["model-diagram"] = diagram
    for before, after in zip(original.cells, notebook.cells, strict=True):
        assert before.id == after.id and before.cell_type == after.cell_type
        if before.id not in SOURCE_CELLS:
            assert before.source == after.source
        if before.cell_type == "code":
            assert before.execution_count == after.execution_count
            non_images = lambda cell: [out for out in cell.outputs if "image/png" not in out.get("data", {})]
            assert non_images(before) == non_images(after), before.id
            if before.id not in images:
                assert before.outputs == after.outputs
        else:
            assert before == after
    assert all(digest(path) == expected for path, expected in hashes.items())
    nbformat.validate(notebook)
    temporary = NOTEBOOK.with_suffix(".color.tmp")
    nbformat.write(notebook, temporary)
    os.replace(temporary, NOTEBOOK)
    image_dir = EVIDENCE / "notebook_color_figures"
    image_dir.mkdir(parents=True, exist_ok=True)
    for cell_id, data in images.items():
        (image_dir / f"{cell_id}.png").write_bytes(data)
    validation = {
        "scope": "Update notebook plotting styles and six embedded figures to match the color report.",
        "notebook_sha256_before": notebook_before,
        "notebook_sha256_after": digest(NOTEBOOK),
        "plot_source_cells": sorted(SOURCE_CELLS),
        "refreshed_figure_cells": list(images),
        "plot_data_matches_saved_results": True,
        "numerical_outputs_preserved": True,
        "markdown_and_training_code_preserved": True,
        "photographic_illustrations_preserved": True,
        "style_application_idempotent": True,
        "protected_files_unchanged": len(hashes),
        "report_path": "reports/LSTM_Project_Report_Color.docx",
        "report_sha256": digest(ROOT / "reports/LSTM_Project_Report_Color.docx"),
        "training_run": False,
        "whole_notebook_rerun": False,
    }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "notebook_color_validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {len(images)} notebook figures; numeric outputs and analysis are preserved.")
    print(f"All {len(hashes)} protected results, report figures, checkpoint and model files are unchanged.")


if __name__ == "__main__":
    refresh()
