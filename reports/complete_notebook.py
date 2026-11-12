"""Complete and execute the submission notebook using saved models and results.

No training is run. Existing experiment evidence is read without modification.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "notebooks/LSTM_Image_Classification.ipynb"
EXTRA_IDS = {"project-abstract", "model-diagram", "evaluation-method", "evaluation-uncertainty",
             "project-references", "project-conclusion"}


def color_figure_style(notebook) -> None:
    """Match the report palette while preserving analysis and training cells."""
    cells = {cell.id: cell for cell in notebook.cells}
    setup = cells["2727d6cb"].source
    if "from matplotlib.colors import LinearSegmentedColormap" not in setup:
        setup = setup.replace("from PIL import Image", "from PIL import Image\n"
                              "from matplotlib.colors import LinearSegmentedColormap")
    if "from matplotlib.ticker import MaxNLocator, PercentFormatter" not in setup:
        setup = setup.replace("from matplotlib.colors import LinearSegmentedColormap",
                              "from matplotlib.colors import LinearSegmentedColormap\n"
                              "from matplotlib.ticker import MaxNLocator, PercentFormatter")
    palette = '''TRAIN_COLOR = "#0072B2"
VALIDATION_COLOR = "#D55E00"
TEST_COLOR = "#009E73"
ARCHITECTURE_COLORS = {"pixel_lstm": "#CC79A7", "cnn_only": TEST_COLOR,
                       "cnn_lstm": VALIDATION_COLOR, "cnn_bilstm": TRAIN_COLOR}
REPORT_HEATMAP = LinearSegmentedColormap.from_list("report_blue", ["#F1F7FD", "#6BAED6", "#08519C"])'''
    if "TRAIN_COLOR =" not in setup:
        if "REPORT_HEATMAP =" in setup:
            setup = re.sub(r"(?m)^REPORT_HEATMAP =[^\n]*", lambda _: palette, setup)
        else:
            setup = setup.replace("from matplotlib.ticker import MaxNLocator, PercentFormatter",
                                  "from matplotlib.ticker import MaxNLocator, PercentFormatter\n" + palette)
    cells["2727d6cb"].source = setup
    cells["ee650eab"].source = '''fig, ax = plt.subplots(figsize=(8, 3.2))
eda["images"].sort_values().plot.barh(ax=ax, color=TRAIN_COLOR)
ax.set_xlabel("images")
ax.set_title("Images per class")
plt.tight_layout()'''
    cells["21e32130"].source = '''def plot_history(run, ax_loss, ax_acc):
    hh = run["history"]
    ep = range(1, len(hh["train_loss"]) + 1)
    for ax, key in ((ax_loss, "loss"), (ax_acc, "acc")):
        ax.plot(ep, hh[f"train_{key}"], color=TRAIN_COLOR, ls="--", lw=2, label="Training")
        ax.plot(ep, hh[f"val_{key}"], color=VALIDATION_COLOR, lw=2, label="Validation")
        ax.axvline(hh["best_epoch"], color="#6B7280", ls=":", lw=1.2)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.set_ylabel("Cross-entropy loss" if key == "loss" else "Accuracy")
        ax.grid(axis="y", color="#CBD5E1", alpha=0.45, linewidth=0.6)
        if key == "acc":
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend(frameon=False)
    ax_loss.set_title(f"{run['name']} (dotted line = best epoch {hh['best_epoch']})", loc="left")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
plot_history(best, *axes)
plt.tight_layout()'''
    cells["e12d967b"].source = '''cm = np.array(m["confusion_matrix"])
share = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.5, 5.8))
image = ax.imshow(share, cmap=REPORT_HEATMAP, vmin=0, vmax=1)
for i in range(len(cm)):
    for j in range(len(cm)):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if share[i, j] > 0.55 else "black")
ax.set_xticks(range(len(cm)), info["classes"], rotation=45, ha="right")
ax.set_yticks(range(len(cm)), info["classes"])
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title("Test confusion matrix (counts)")
ax.set_xticks(np.arange(len(cm) + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(len(cm) + 1) - 0.5, minor=True)
ax.grid(which="minor", color="white", linewidth=0.7)
ax.tick_params(which="minor", bottom=False, left=False)
colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
colorbar.set_label("Fraction of true class")
colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
plt.tight_layout()

off = [(info["classes"][i], info["classes"][j], int(cm[i, j])) for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j]]
pd.DataFrame(sorted(off, key=lambda t: -t[2])[:8], columns=["true", "predicted", "count"])'''
    cells["dcd7e3fc"].source = '''fig, ax = plt.subplots(figsize=(9, 3.8))
line_styles = {"pixel_lstm": ":", "cnn_only": "--", "cnn_lstm": "-", "cnn_bilstm": "-."}
for name in ("pixel_lstm", "cnn_only", "cnn_lstm", "cnn_bilstm"):
    v = runs[name]["history"]["val_acc"]
    ax.plot(range(1, len(v) + 1), v, label=name, color=ARCHITECTURE_COLORS[name], ls=line_styles[name], lw=2.2 if name == "cnn_bilstm" else 1.5)
ax.set_xlabel("epoch")
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
ax.set_ylabel("validation accuracy")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.grid(axis="y", color="#CBD5E1", alpha=0.45, linewidth=0.6)
ax.set_title("Architecture comparison")
ax.legend(frameon=False)
plt.tight_layout()'''


def write_notebook_text(notebook, text) -> None:
    """Apply the study narrative without workflow instructions or build notes."""
    notebook.cells = [cell for cell in notebook.cells
                      if cell.id not in {"project-references", "project-conclusion"}
                      and not (cell.cell_type == "markdown"
                               and re.search(r"^#+\s+(?:\d+[.]?\s+)?References\s*$", cell.source, re.M | re.I))]
    narrative = text["notebook"]
    identity = text["author"]
    contents = "\n".join(f"| {i} | {title} |" for i, title in enumerate(narrative["titles"], 1))
    notebook.cells[0].source = (
        f"# {narrative['title']}\n\n"
        f"**Author:** {identity['name']}\n\n"
        f"**Affiliation:** {identity['role']}, {identity['institution']}\n\n"
        "**Dataset:** Natural Images, eight classes\n\n"
        "| Section | Topic |\n| --- | --- |\n" + contents
    )
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        if cell.id in narrative["sections"]:
            cell.source = narrative["sections"][cell.id]
        elif cell.id in narrative["observations"]:
            name, title = narrative["observations"][cell.id]
            bullets = text["observations"][name]
            assert sum(len(item.split()) for item in bullets) == 100
            cell.source = f"### {title}\n\n" + "\n".join("- " + item for item in bullets)
        elif cell.id == "af57302b":
            bullets = text["final_insights"]
            assert sum(len(item.split()) for item in bullets) == 300
            cell.source = "## 11. Final Insights and Recommendations\n\n" + "\n".join("- " + item for item in bullets)
    notebook.cells.append(nbformat.v4.new_markdown_cell(
        "## 12. Conclusion\n\n" + narrative["conclusion"], id="project-conclusion"))


def concise_notebook_output(notebook) -> None:
    """Keep analytical output and omit routine execution commentary."""
    omitted = [
        'print("Reloaded checkpoint probabilities match the saved predictions exactly.")\n',
        'print("Text-marked examples omitted from display only; all 38 errors remain in the metrics.")',
        'print(additional["scope"])',
        '# Text-marked source photographs are omitted from illustrations only.\n',
    ]
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = re.sub(r"(?m)^RUN_TRAINING = False[^\n]*", "RUN_TRAINING = False", cell.source)
        for line in omitted:
            source = source.replace(line, "")
        source = source.replace("splits, norm, info = E.prepare_data()", "splits, norm, info = E.prepare_data(log=lambda _: None)")
        source = source.replace('print(f"{len(runs)} runs loaded | selected: {best_name}")',
                                'print(f"Selected model: {best_name}")')
        source = source.replace('Illustrative examples by class (original resolution)', 'Sample Images by Class')
        source = source.replace('High-probability classification errors (illustrative subset)', 'Misclassified Test Images')
        source = source.replace('Baseline (top) vs no augmentation, dropout or weight decay (bottom)',
                                'Baseline and Selected Model')
        cell.source = source.rstrip()


def complete() -> None:
    text = json.loads((ROOT / "reports/report_text.json").read_text(encoding="utf-8"))
    notebook = nbformat.read(PATH, as_version=4)
    base = [cell for cell in notebook.cells if cell.id not in EXTRA_IDS]
    assert len(base) == 48, "The notebook structure changed; review the section mapping."

    base[4].source = base[4].source.replace("torch.set_num_threads(8)\n", "").replace(
        "set_seed(42)", "torch.set_num_threads(8)\nset_seed(42)")
    base[7].source = '''DATASET_DIR = ROOT / "data" / "natural-images"
if not E.DATA_ROOT.exists():
    import kagglehub
    kagglehub.dataset_download("prasunroy/natural-images", output_dir=str(DATASET_DIR))

all_files = [p for p in DATASET_DIR.rglob("*") if p.is_file()]
local_manifest = {p.relative_to(DATASET_DIR).as_posix(): p.stat().st_size for p in all_files}
published_manifest = json.loads((ROOT / "reports/audit_evidence/publisher_manifest.json").read_text())
verification = json.loads((ROOT / "reports/audit_evidence/publisher_verification.json").read_text())
assert local_manifest == published_manifest, "Current files differ from the saved publisher manifest"
print(f"files: {len(all_files):,} | bytes: {sum(local_manifest.values()):,}")
print(f"All names and sizes match the publisher manifest fetched {verification['checked_at_utc']}.")
print("The publisher manifest supplies no cryptographic content hashes.")
top = pd.Series([p.relative_to(DATASET_DIR).parts[0] for p in all_files]).value_counts()
top.rename("files").to_frame()'''
    base[20].source = '''# Text-marked source photographs are omitted from illustrations only.
display_rules = json.loads((ROOT / "reports/display_exclusions.json").read_text())
display_replacements = display_rules["example_replacements"]
fig, axes = plt.subplots(len(index.classes), 6, figsize=(9, 12))
rng = np.random.default_rng(0)
for r, cls in enumerate(index.classes):
    for c, i in enumerate(rng.choice(np.flatnonzero(index.labels == r), 6, replace=False)):
        path = display_replacements.get(index.paths[i], index.paths[i])
        with Image.open(E.DATA_ROOT / path) as im:
            axes[r, c].imshow(im.convert("RGB"))
        axes[r, c].axis("off")
    axes[r, 0].set_title(cls, loc="left", fontsize=9)
fig.suptitle("Illustrative examples by class (original resolution)")
plt.tight_layout()'''
    base[25].source = '''# Screen for average-hash matches across the actual split assignments.
hash_of = dict(zip(audit["class"] + "/" + audit["file"], audit["ahash"]))
training_idx, validation_idx, test_idx = D.stratified_split(index.labels, .15, .15, 42)
training_hashes = {hash_of[index.paths[i]] for i in training_idx}
training_validation_hashes = training_hashes | {hash_of[index.paths[i]] for i in validation_idx}
test_with_twin = [p for p in info["test_paths"] if hash_of[p] in training_validation_hashes]
print("Test candidates matching training:", sum(hash_of[p] in training_hashes for p in info["test_paths"]))
print("Test candidates matching training or validation:", len(test_with_twin), "of", len(info["test_paths"]))
print("By class:", pd.Series([p.split('/')[0] for p in test_with_twin]).value_counts().to_dict())
print("A matching hash flags possible similarity; it does not prove duplication.")'''
    if "saved_probabilities" not in base[37].source:
        base[37].source = base[37].source.replace('m = compute_metrics(',
            'saved_probabilities = np.load(E.RESULTS_DIR / "runs" / meta["name"] / "test_probs.npy")\n'
            'np.testing.assert_array_equal(probs, saved_probabilities)\n'
            'print("Reloaded checkpoint probabilities match the saved predictions exactly.")\n'
            'm = compute_metrics(')
    base[33].source = base[33].source.replace("training time:", "recorded elapsed:")
    base[40].source = '''pred = probs.argmax(1)
wrong = np.flatnonzero(pred != splits["test"].y)
print(f"{len(wrong)} of {len(pred)} test images misclassified")
excluded_displays = set(display_rules["excluded_paths"])
eligible = [i for i in wrong if info["test_paths"][i] not in excluded_displays]
show = sorted(eligible, key=lambda i: -float(probs[i].max()))[:12]
fig, axes = plt.subplots(2, 6, figsize=(12, 6))
for ax, i in zip(axes.flat, show):
    ax.imshow(D.load_image(E.DATA_ROOT / info["test_paths"][i], 96))
    ax.set_title(f"pred {info['classes'][pred[i]]} {probs[i].max():.0%}\\ntrue {info['classes'][splits['test'].y[i]]}",
                 fontsize=9, color="firebrick")
    ax.axis("off")
fig.suptitle("High-probability classification errors (illustrative subset)")
plt.tight_layout(rect=(0, 0, 1, .94), h_pad=2.0)
print("Text-marked examples omitted from display only; all 38 errors remain in the metrics.")'''

    diagram = nbformat.v4.new_code_cell(
        'from IPython.display import Image as DisplayImage, display\n'
        'display(DisplayImage(filename=str(ROOT / "results/figures/architecture_diagram.png")))', id="model-diagram")
    method = nbformat.v4.new_markdown_cell(text["notebook"]["sections"]["evaluation-method"],
                                           id="evaluation-method")
    uncertainty = nbformat.v4.new_code_cell('''from reports.analyse_results import analyse
additional = analyse()
intervals = pd.DataFrame([
    ["Test accuracy", additional["accuracy"], *additional["accuracy_wilson_95"]],
    ["Test macro F1", additional["macro_f1"], *additional["macro_f1_bootstrap_95"]],
    ["Selected minus baseline accuracy", additional["selected_minus_baseline_accuracy"],
     *additional["paired_accuracy_improvement_bootstrap_95"]],
], columns=["estimate", "value", "lower_95", "upper_95"])
display(intervals.set_index("estimate"))
display(pd.DataFrame(additional["sensitivity"]).T)
print("Paired contrast: corrected baseline errors:", additional["selected_correct_baseline_wrong"],
      "| introduced errors:", additional["selected_wrong_baseline_correct"])
print(additional["scope"])''', id="evaluation-uncertainty")
    notebook.cells = []
    for i, cell in enumerate(base):
        notebook.cells.append(cell)
        if i == 30:
            notebook.cells.append(diagram)
        elif i == 40:
            notebook.cells.extend([method, uncertainty])
    write_notebook_text(notebook, text)
    concise_notebook_output(notebook)
    color_figure_style(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code" and "RUN_TRAINING =" in cell.source:
            assert cell.source.startswith("RUN_TRAINING = False")
    # Put runtime files in the project; startup does not need a global profile.
    runtime = ROOT / "artifacts/jupyter"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    os.environ["IPYTHONDIR"] = str(runtime / "ipython")
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    client = NotebookClient(notebook, timeout=300, kernel_name="python3", allow_errors=False,
                            resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, PATH)
    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    html, _ = exporter.from_notebook_node(notebook)
    html = html.replace("</head>", '<style>body{background:white}body *{color:black !important}</style></head>', 1)
    PATH.with_suffix(".html").write_text(html, encoding="utf-8")
    print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells without training.")
    print(f"Notebook and HTML written: {PATH}")


if __name__ == "__main__":
    complete()
