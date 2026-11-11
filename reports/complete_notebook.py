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


def black_figure_text(notebook) -> None:
    """Use black annotations and distinguish chart series by line style."""
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source
        if "RUN_TRAINING = False" in source and "REPORT_HEATMAP =" not in source:
            source = source.replace("from PIL import Image", "from PIL import Image\n"
                "from matplotlib.colors import LinearSegmentedColormap\n"
                'REPORT_HEATMAP = LinearSegmentedColormap.from_list("report_gray", ["white", "#c4c4c4"])')
            source += ('\nplt.rcParams.update({"text.color": "black", "axes.labelcolor": "black", '
                       '"axes.titlecolor": "black", "xtick.color": "black", "ytick.color": "black"})')
        source = source.replace('color="grey", label="train"', 'color="black", ls="--", label="train"')
        source = source.replace('color="tab:blue"', 'color="black"')
        source = source.replace('color="firebrick"', 'color="black"')
        source = source.replace('cmap="Blues"', 'cmap=REPORT_HEATMAP')
        source = re.sub(r'color="white" if share\[i, j\] > 0\.5 else "black"', 'color="black"', source)
        if 'for name in ("pixel_lstm"' in source:
            if "line_styles =" not in source:
                source = source.replace('for name in ("pixel_lstm"',
                    'line_styles = {"pixel_lstm": ":", "cnn_only": "--", "cnn_lstm": "-.", "cnn_bilstm": "-"}\n'
                    'for name in ("pixel_lstm"')
            source = source.replace('v, label=name, lw=', 'v, label=name, color="black", ls=line_styles[name], lw=')
        cell.source = source


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
    black_figure_text(notebook)
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
