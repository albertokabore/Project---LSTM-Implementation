"""Build the Word project report from saved results.

    python reports/build_report.py

Measured results come from saved JSON and prediction arrays. Discussion and
references are read from report sources. Building this document does not execute
or modify the notebook, regenerate figures, or overwrite analysis results.
"""

from __future__ import annotations

import json
import argparse
import hashlib
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from lxml import etree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.section import WD_SECTION  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
OUT = ROOT / "reports" / "LSTM_Project_Report.docx"
TEXT_FILE = ROOT / "reports" / "report_text.json"

REQUIRED_TEXT = [
    "abstract", "results_discussion", "results_improve", "optimisation_experiments",
    "optimisation_insights", "conclusion_findings", "conclusion_future",
]

FONT = "Calibri"
BLACK = RGBColor(0, 0, 0)

# ----------------------------------------------------------------------------- helpers


def set_cell_shading(cell, hex_fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def add_field(paragraph, instr: str, placeholder: str = "") -> None:
    """Insert a Word field (PAGE, TOC ...). Word fills it in when fields are updated."""
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    text = OxmlElement("w:instrText")
    text.set(qn("xml:space"), "preserve")
    text.text = instr
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    run._r.extend([begin, text, sep])
    paragraph.add_run(placeholder)
    end_run = paragraph.add_run()
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    end_run._r.append(end)


class Report:
    def __init__(self):
        self.doc = Document()
        self.table_no = 0
        self.figure_no = 0
        self._styles()
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        for side in ("left_margin", "right_margin"):
            setattr(sec, side, Cm(2.3))
        sec.top_margin = sec.bottom_margin = Cm(2.2)

    def _styles(self):
        st = self.doc.styles
        normal = st["Normal"]
        normal.font.name = FONT
        normal.font.size = Pt(11)
        normal.font.color.rgb = BLACK
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        normal.paragraph_format.space_after = Pt(6)
        normal.paragraph_format.line_spacing = 1.15
        for name, size, before, after in (("Title", 26, 0, 6), ("Heading 1", 16, 18, 6),
                                          ("Heading 2", 13, 12, 4), ("Heading 3", 11.5, 10, 3)):
            s = st[name]
            s.font.name = FONT
            s.font.size = Pt(size)
            s.font.bold = True
            s.font.color.rgb = BLACK
            rfonts = s.element.rPr.rFonts
            for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                rfonts.set(qn(attr), FONT)
            for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                if rfonts.get(qn(attr)) is not None:
                    del rfonts.attrib[qn(attr)]
            s.paragraph_format.space_before = Pt(before)
            s.paragraph_format.space_after = Pt(after)
            s.paragraph_format.keep_with_next = True
        cap = st["Caption"]
        cap.font.name = FONT
        cap.font.size = Pt(9.5)
        cap.font.italic = True
        cap.font.color.rgb = BLACK
        cap.paragraph_format.space_after = Pt(10)

    # ---- text
    def h(self, text, level=1):
        self.doc.add_heading(text, level=level)

    def p(self, text, bold_lead: str | None = None, italic=False, align=None):
        para = self.doc.add_paragraph()
        if bold_lead:
            para.add_run(bold_lead).bold = True
        run = para.add_run(text)
        run.italic = italic
        if align:
            para.alignment = align
        return para

    def bullets(self, items, numbered=False):
        style = "List Number" if numbered else "List Bullet"
        for item in items:
            para = self.doc.add_paragraph(style=style)
            if isinstance(item, tuple):
                para.add_run(item[0]).bold = True
                para.add_run(item[1])
            else:
                para.add_run(item)

    def caption(self, kind: str, text: str):
        if kind == "Table":
            self.table_no += 1
            n = self.table_no
        else:
            self.figure_no += 1
            n = self.figure_no
        para = self.doc.add_paragraph(style="Caption")
        para.add_run(f"{kind} {n}. ").bold = True
        para.add_run(text)
        if kind == "Table":
            para.paragraph_format.keep_with_next = True
        return n

    # ---- tables and figures
    def table(self, header, rows, widths_cm, caption, align_right_from=1, bold_last=False):
        self.caption("Table", caption)
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for column, width in zip(t.columns, widths_cm):
            column.width = Cm(width)
        for i, text in enumerate(header):
            cell = t.rows[0].cells[i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(text))
            run.bold = True
            run.font.size = Pt(9.5)
            run.font.color.rgb = BLACK
            set_cell_shading(cell, "E6E6E6")
        for r_i, row in enumerate(rows):
            cells = t.add_row().cells
            for i, val in enumerate(row):
                cells[i].text = ""
                run = cells[i].paragraphs[0].add_run(str(val))
                run.font.size = Pt(9.5)
                if bold_last and r_i == len(rows) - 1:
                    run.bold = True
                if i >= align_right_from:
                    cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
                if r_i % 2 == 1:
                    set_cell_shading(cells[i], "F2F2F2")
        for row in t.rows:
            for i, w in enumerate(widths_cm):
                row.cells[i].width = Cm(w)
            tr_pr = row._tr.get_or_add_trPr()
            cant = OxmlElement("w:cantSplit")
            tr_pr.append(cant)
        # repeat header row on page breaks
        hdr = t.rows[0]._tr.get_or_add_trPr()
        rep = OxmlElement("w:tblHeader")
        hdr.append(rep)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return t

    def figure(self, path: Path, caption: str, width_cm=16.0):
        if not path.exists():
            raise FileNotFoundError(f"missing figure {path}; run `python -m cnn_lstm.plots` first")
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = True
        para.add_run().add_picture(str(path), width=Cm(width_cm))
        cap = self.caption("Figure", caption)
        return cap

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def pct(x: float, digits=1) -> str:
    return f"{100 * x:.{digits}f}%"


def black_text(document) -> None:
    """Apply explicit black text to styles, runs, numbering and footers."""
    for style in document.styles:
        if hasattr(style, "font"):
            style.font.color.rgb = BLACK
    roots = [document.element, document.styles.element, document.part.numbering_part.element]
    roots.extend(section.footer._element for section in document.sections)
    for root in roots:
        for run in root.xpath(".//w:r"):
            properties = run.get_or_add_rPr()
            color = properties.find(qn("w:color"))
            if color is None:
                color = OxmlElement("w:color")
                properties.append(color)
            color.attrib.clear()
            color.set(qn("w:val"), "000000")
        for color in root.xpath(".//w:color"):
            color.attrib.clear()
            color.set(qn("w:val"), "000000")


def clean_template_properties(path: Path) -> None:
    """Remove default application claims and unmeasured template statistics."""
    source = BytesIO(path.read_bytes())
    target = BytesIO()
    with ZipFile(source) as original, ZipFile(target, "w") as revised:
        for item in original.infolist():
            data = original.read(item.filename)
            if item.filename == "docProps/app.xml":
                root = etree.fromstring(data)
                omitted = {"Application", "AppVersion", "Template", "TotalTime", "Pages", "Words",
                           "Characters", "Lines", "Paragraphs", "CharactersWithSpaces"}
                for node in list(root):
                    if etree.QName(node).localname in omitted:
                        root.remove(node)
                data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
            elif item.filename == "docProps/core.xml":
                root = etree.fromstring(data)
                for node in list(root):
                    if etree.QName(node).localname in {"created", "modified", "lastPrinted", "revision"}:
                        root.remove(node)
                data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
            revised.writestr(item, data)
    path.write_bytes(target.getvalue())


def lstm_equations_png(path: Path) -> None:
    lines = [
        r"$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$  (forget gate)",
        r"$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$  (input gate)",
        r"$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$  (output gate)",
        r"$\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$  (candidate memory)",
        r"$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$  (cell state)",
        r"$h_t = o_t \odot \tanh(c_t)$  (hidden state)",
    ]
    fig = plt.figure(figsize=(6.4, 2.3))
    for i, line in enumerate(lines):
        fig.text(0.02, 0.9 - i * 0.16, line, fontsize=12, va="center")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)



# ----------------------------------------------------------------------------- content

def build(output: Path = OUT) -> Path:
    text = json.loads(TEXT_FILE.read_text(encoding="utf-8")) if TEXT_FILE.exists() else {}
    required = REQUIRED_TEXT + ["cleaning_findings", "challenges", "observations", "final_insights", "author"]
    missing = [key for key in required if not text.get(key)]
    if missing:
        sys.exit(f"Missing report content: {missing}")
    for name, bullets in text["observations"].items():
        if sum(len(item.split()) for item in bullets) != 100:
            raise ValueError(f"{name}: observations must contain 100 words")
    if sum(len(item.split()) for item in text["final_insights"]) != 300:
        raise ValueError("Final insights and recommendations must contain 300 words")

    info = json.loads((RESULTS / "data_info.json").read_text())
    summary = json.loads((RESULTS / "summary.json").read_text())
    runs = {p.parent.name: json.loads(p.read_text()) for p in (RESULTS / "runs").glob("*/result.json")}
    best = runs[summary["selected"]]
    baseline = runs["cnn_bilstm"]
    classes = info["classes"]
    tm = best["test"]
    analysis = json.loads((ROOT / "reports/analysis_results.json").read_text(encoding="utf-8"))
    # The report consumes reviewed evidence; recalculation is a separate command.
    for relative_path, expected in analysis["input_sha256"].items():
        if hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Analysis evidence changed: {relative_path}; recompute analysis before building")

    r = Report()
    d = r.doc
    d.core_properties.title = "LSTM Implementation Project: Image Classification"
    d.core_properties.subject = "Natural Images classification using CNN and LSTM models"
    d.core_properties.author = text["author"]["name"]
    d.core_properties.last_modified_by = ""
    d.core_properties.comments = ""
    for _ in range(4):
        d.add_paragraph()
    title = d.add_paragraph("LSTM Implementation Project", style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r.p("Image Classification with Convolutional Neural Networks and LSTM",
        align=WD_ALIGN_PARAGRAPH.CENTER)
    d.add_paragraph()
    for line in (text["author"]["name"], text["author"]["role"], text["author"]["institution"]):
        r.p(line, align=WD_ALIGN_PARAGRAPH.CENTER)
    r.page_break()

    r.h("Abstract")
    for para in text["abstract"]:
        r.p(para)

    r.h("1. Introduction")
    r.h("1.1 CNN and LSTM concepts", 2)
    r.p("A convolutional neural network applies learned filters to local image regions. Successive layers combine "
        "edges, colours and textures into classification features, with weights shared across locations "
        "(LeCun et al., 1998). Pooling reduces spatial resolution and the amount of computation.")
    r.p("Long Short-Term Memory networks process ordered sequences using a memory cell and hidden state "
        "(Hochreiter & Schmidhuber, 1997). The modern formulation uses input, forget and output gates to control "
        "updates and information flow; the adaptive forget gate was introduced by Gers et al. (2000). A "
        "bidirectional network reads the sequence in both directions and combines the states "
        "(Schuster & Paliwal, 1997).")
    r.p("The input gate controls new content, the forget gate scales retained memory, and the output gate "
        "controls the exposed hidden representation. The additive cell-state update provides a path for "
        "information and gradients across sequence steps, helping an LSTM learn dependencies that a simple "
        "recurrent network may struggle to retain (Hochreiter & Schmidhuber, 1997; Gers et al., 2000).")
    r.h("1.2 Spatial sequence hypothesis", 2)
    r.p("Here, each row of a CNN feature map becomes one sequence step. Row order may help represent layouts "
        "such as wheels below a vehicle body or a head above a torso. Global average pooling removes the final "
        "spatial indices, although earlier convolutions can encode spatial context. Recurrence over images was "
        "investigated in ReNet, which uses horizontal and vertical recurrent sweeps (Visin et al., 2015). "
        "CNN and LSTM combinations also appear in scene text recognition (Shi et al., 2017) and video recognition "
        "(Donahue et al., 2015). These studies motivate the design; the present comparison evaluates its performance "
        "on Natural Images.")
    r.p("This project classifies individual still images. Its sequence steps represent spatial rows within one "
        "image; they do not represent video frames or separate photographs. The hypothesis is that gated "
        "integration of ordered local features can help distinguish object layouts. The eight-step sequence "
        "tests that spatial design without establishing an advantage on long temporal sequences.")
    r.h("1.3 Objectives and dataset choice", 2)
    r.bullets([
        "Inspect and preprocess labelled photographs using resizing, normalization and training augmentation.",
        "Implement a CNN with LSTM in PyTorch, including batch normalization and dropout in baseline experiments.",
        "Compare architectures and training settings using validation metrics, then report test accuracy, precision, recall and F1.",
        "Document errors, uncertainty, limitations and specific follow-up experiments.",
    ])
    r.p("The analysis used Natural Images (Roy, 2018), the dataset linked in the assignment. "
        "CIFAR-10 (Krizhevsky, 2009) was provided as an alternative example.")

    r.h("2. Methodology")
    r.h("2.1 Dataset and split", 2)
    r.p("Natural Images contains 6,899 photographs in eight classes: " + ", ".join(classes) + ". "
        "The downloaded archive stores two copies. The pipeline reads only natural_images and indexes two exact "
        "duplicate pairs once, leaving 6,897 images. A saved publisher-manifest "
        "comparison matches all 13,798 downloaded filenames and sizes, totalling 359,403,192 bytes. "
        "Publisher cryptographic hashes were unavailable.")
    r.p("Folder names provide one label per image, encoded as integers 0 through 7 in alphabetical class order. "
        "After exact-duplicate indexing, the split first reserves 15% for testing, then takes "
        "15/85 of the remainder for validation. Both stratified operations use seed 42. This preserves "
        "class representation while reserving separate data for fitting, configuration selection and reporting.")
    counts = info["class_counts"]
    rows = [[c] + [str(counts[s][i]) for s in ("train", "val", "test")]
            + [str(sum(counts[s][i] for s in ("train", "val", "test")))] for i, c in enumerate(classes)]
    totals = [sum(counts[s]) for s in ("train", "val", "test")]
    rows.append(["Total", *[f"{v:,}" for v in totals], f"{sum(totals):,}"])
    r.table(["Class", "Training", "Validation", "Test", "Total"], rows,
            [3.4, 3.1, 3.1, 3.1, 3.7],
            "Class counts after exact-duplicate indexing; stratified 70/15/15 allocation with seed 42.", bold_last=True)
    r.figure(FIGS / "class_distribution.png", "Class counts within the fixed split.", 15)
    r.h("2.2 Image quality and cleaning", 2)
    r.bullets(text["cleaning_findings"])
    r.p("In the original dataset, fruit has 1,000 images and dog has 702, a class-count ratio of 1.42. "
        "Car and fruit photographs are consistently 100x100 pixels. Fruit's average brightness is 0.617, compared "
        "with 0.408 to 0.487 for other classes. Dimensions, illumination and source annotations may provide "
        "acquisition cues; model dependence on these features was not tested.")
    r.h("2.3 Preprocessing and augmentation", 2)
    r.bullets([
        "Resize RGB images to 64x64 using bicubic interpolation. This limits computation at the cost of detail and "
        "aspect-ratio distortion; aspect-preserving padding is a proposed comparison.",
        "Use training-only channel means " + ", ".join(f"{v:.3f}" for v in info["channel_mean"]) +
        " and standard deviations " + ", ".join(f"{v:.3f}" for v in info["channel_std"]) +
        " for RGB normalization after scaling pixels to [0, 1].",
        "When enabled, augment training batches using horizontal flips with probability 0.5, reflection-padded "
        "shifts of up to six pixels, and brightness/contrast multipliers from 0.8 to 1.2. Exclude vertical flips "
        "to preserve typical object orientation.",
        "Apply deterministic resizing and normalization to validation and test inputs. Augmentation settings for "
        "the baseline and selected models are reported in Section 3.2.",
    ])
    r.figure(FIGS / "augmentation_examples.png",
             "Training photographs before augmentation (top) and three sampled transformations.", 14)
    r.h("2.4 Evaluation independence", 2)
    r.p("No byte-identical or decoded-pixel-identical images cross the split boundaries in the saved audit. "
        "However, 39 test images share an 8x8 grayscale average hash with training images; including validation "
        "increases this to 41, comprising 38 fruit and three motorbike images. Visual inspection of candidate "
        "pairs identified closely related views and false positives. Hash matches therefore require visual "
        "confirmation. Potential related-image overlap limits inference about independent photographs.")
    r.h("2.5 Study design and rationale", 2)
    r.p("The experiment uses a compact CNN followed by an LSTM, trained end to end from random initialization. "
        "No pretrained feature extractor or external training images are used. Three pooling stages balance "
        "local feature extraction against the CPU budget, and a single recurrent layer with 128 units per "
        "direction provides the reference hybrid configuration. Section 3 specifies its tensor dimensions "
        "and the alternatives used to evaluate these choices.")
    r.p("Stratification addresses representation in each split, while training-only normalization prevents "
        "validation or test intensities from influencing preprocessing parameters. Unweighted cross-entropy "
        "retains the observed training class frequencies; no oversampling or class weighting is used. "
        "Validation macro F1 gives all eight classes equal influence in ranking, which is useful given "
        "the unequal class counts. Accuracy and support-weighted F1 complement that view.")

    r.h("3. Model Implementation and Training")
    r.h("3.1 Architecture and design choices", 2)
    r.p("The CNN uses three blocks, each with two 3x3 convolutions, batch normalization, ReLU and 2x2 max pooling. "
        "Channel counts are 32, 64 and 128. The resulting 8x8x128 map is rearranged into eight rows of 1,024 "
        "values. A single bidirectional LSTM layer has 128 units per direction. Its final states form a 256-value "
        "vector, passed through a 128-unit classification head to eight logits. Softmax is applied for reporting "
        "probabilities; training uses cross-entropy on logits.")
    r.p("Convolutions use stride 1 and padding 1, preserving resolution until each pooling operation. In "
        "PyTorch's channel-first format, a batch follows [B, 3, 64, 64] to [B, 128, 8, 8], then "
        "[B, 8, 1,024]. Concatenating the final forward and backward hidden states yields [B, 256]. "
        "The head applies a bias-free linear layer, batch normalization, ReLU and a final linear layer "
        "with eight outputs. B denotes batch size; every prediction concerns one image.")
    r.figure(FIGS / "architecture_diagram.png",
             "CNN with bidirectional LSTM; dropout differs between the baseline and selected configurations.", 16.4)
    r.table(["Stage", "Output per image", "Parameters"], [
        ["CNN block 1: 3 to 32 channels", "32 x 32 x 32", "10,208"],
        ["CNN block 2: 32 to 64 channels", "64 x 16 x 16", "55,552"],
        ["CNN block 3: 64 to 128 channels", "128 x 8 x 8", "221,696"],
        ["Rows to sequence", "8 x 1,024", "0"],
        ["Bidirectional LSTM", "256 final-state values", "1,181,696"],
        ["Classification head", "8 logits", "34,056"],
        ["Total", "", f"{best['params']:,}"],
    ], [6.2, 6.0, 4.2],
        "Selected model summary; feature-map dimensions are channels x height x width. "
        "The CNN blocks total 287,456 parameters. Dropout has no trainable parameters.",
        bold_last=True, align_right_from=2)
    r.bullets([
        "Convolutions provide local feature extraction before recurrence; eight sequence steps reduce the recurrent "
        "sequence length compared with 64 raw-pixel rows.",
        "Bidirectional reading makes both row orders available. The learned weights determine their influence.",
        "The pixel LSTM has 315,016 parameters and the CNN control has 305,128, compared with 1,503,208 in the "
        "bidirectional hybrid. Unequal capacity prevents attributing performance differences solely to recurrence.",
    ])
    r.p("The pixel LSTM control reads 64 rows of 192 RGB values, applies layer normalization, and uses two "
        "unidirectional recurrent layers with 128 hidden units. The CNN control uses the same convolutional "
        "backbone followed by global average pooling and a classification head. The unidirectional hybrid "
        "reads the same eight feature-map rows as the bidirectional model but supplies a 128-value final "
        "state to the head. These controls compare raw-pixel recurrence, convolutional features and spatial "
        "recurrence under the recorded training budget.")
    r.h("3.2 Training and model selection", 2)
    cfg = best["train"]
    r.p("Implementation uses PyTorch (Paszke et al., 2019), with scikit-learn metrics (Pedregosa et al., 2011). "
        f"Original metadata records Python {info['environment']['python']}, PyTorch {info['environment']['torch']}, "
        f"and {info['environment']['threads']} CPU threads. No GPU was used.")
    r.table(["Setting", "Bidirectional baseline", "Selected configuration"], [
        ["Optimizer", "AdamW", "AdamW"],
        ["Peak learning rate", f"{baseline['train']['lr']:g}", f"{cfg['lr']:g}"],
        ["Weight decay", f"{baseline['train']['weight_decay']:g}", f"{cfg['weight_decay']:g}"],
        ["Training augmentation", str(baseline["train"]["augment"]), str(cfg["augment"])],
        ["Sequence/head dropout", str(baseline["model"]["dropout"]), str(best["model"]["dropout"])],
        ["Spatial dropout", "0.1", "0.0"],
        ["Batch size / epoch limit", "64 / 20", f"{cfg['batch_size']} / {cfg['epochs']}"],
        ["Loss / label smoothing", "Cross-entropy / 0", "Cross-entropy / 0"],
        ["Split / training seed", "42 / 42", "42 / 42"],
        ["Retained epoch", str(baseline["history"]["best_epoch"]), str(best["history"]["best_epoch"])],
    ], [5.4, 5.5, 5.5], "Training settings read from the saved configurations, with architecture-derived spatial dropout.")
    r.p("Batch normalization follows each convolution and the first head linear layer (Ioffe & Szegedy, 2015). "
        "Baseline spatial and sequence/head dropout follow Tompson et al. (2015) and Srivastava et al. (2014), "
        "respectively. AdamW applies decoupled weight decay when enabled (Loshchilov & Hutter, 2019). The one-cycle schedule "
        "(Smith & Topin, 2018) warms up for 15% of steps, then uses cosine annealing toward a small final rate. "
        "The loop clips gradient norm at 1.0. Validation loss must decrease by more than 1e-4 to reset the six-epoch "
        "patience counter and retain new weights. Runs are ranked by validation macro F1 after their checkpoints "
        "are restored. Test predictions were recorded after each experiment, before the full search concluded.")
    r.p("Each epoch shuffles the training indices using the seeded generator and performs 76 mini-batch "
        "updates, including the final 27 images. A batch passes through augmentation when enabled, "
        "normalization, the model and mean cross-entropy loss. Backpropagation updates the CNN, LSTM "
        "and head jointly; clipping precedes the optimizer step, and the learning-rate scheduler advances "
        "once per update. Validation runs with gradients disabled and batch normalization in evaluation mode.")
    r.p("Dropout placement is explicit: baseline spatial dropout removes feature channels after pooling "
        "with probability 0.1, and probability 0.5 is used on the row sequence and twice in the classifier "
        "head. A one-layer LSTM has no internal inter-layer dropout; the two-layer variant also applies "
        "dropout between its recurrent layers. All dropout probabilities are zero in the selected model. "
        "The name tune_no_reg refers to removal of augmentation, dropout and weight decay; batch "
        "normalization and other training controls remain active.")
    r.h("3.3 Training observations", 2)
    r.bullets(text["observations"]["training"])
    r.p("All ten runs reached the 20-epoch limit; the six-epoch early-stopping threshold was never reached. "
        "The selected weights were restored from epoch 17, rather than taken from epoch 20. Figure 4 shows "
        "training loss continuing toward zero while validation loss levels off around 0.16, with a "
        "4.83-percentage-point training-to-validation accuracy gap at the retained epoch. This pattern "
        "shows a generalization gap and supports checkpoint restoration; it does not demonstrate that "
        "additional training or removing regularization would improve independent-image performance.")
    r.h("3.4 Computational constraints", 2)
    r.bullets(text["challenges"])

    r.h("4. Results and Evaluation")
    r.p(f"Validation macro F1 selected {summary['selected']}. Reloading its retained checkpoint reproduces the saved "
        "test probabilities exactly in the local audit. Metrics use the highest-probability class. For each class, "
        "precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is their harmonic mean. Macro F1 averages class F1 "
        "without weighting; weighted F1 weights by class support.")
    r.p("Accuracy is the fraction of correctly classified images. Top-2 accuracy checks whether the true "
        "class appears among the two highest-scoring classes and is reported only as a supplementary "
        "measure. The main decision still uses one class per image, with no threshold tuning or test-time augmentation.")
    r.table(["Metric", "Validation", "Test"], [
        ["Accuracy", pct(best["val"]["accuracy"], 2), pct(tm["accuracy"], 2)],
        ["Macro precision", pct(best["val"]["macro_precision"], 2), pct(tm["macro_precision"], 2)],
        ["Macro recall", pct(best["val"]["macro_recall"], 2), pct(tm["macro_recall"], 2)],
        ["Macro F1", f"{best['val']['macro_f1']:.4f}", f"{tm['macro_f1']:.4f}"],
        ["Weighted F1", f"{best['val']['weighted_f1']:.4f}", f"{tm['weighted_f1']:.4f}"],
        ["Top-2 accuracy", pct(best["val"]["top2_accuracy"], 2), pct(tm["top2_accuracy"], 2)],
    ], [6.4, 5.0, 5.0], "Performance of the selected epoch-17 checkpoint.")
    pc = tm["per_class"]
    r.table(["Class", "Precision", "Recall", "F1", "Support"], [
        [c, pct(pc[c]["precision"], 2), pct(pc[c]["recall"], 2), f"{pc[c]['f1']:.4f}", pc[c]["support"]]
        for c in sorted(classes, key=lambda c: pc[c]["f1"])
    ], [3.6, 3.2, 3.2, 3.2, 3.2], "Test metrics by class, ordered by F1.")
    r.figure(FIGS / "training_curves_selected.png",
             "Selected-run histories. Training metrics are recorded during optimization; validation uses evaluation mode.", 16)
    r.figure(FIGS / "confusion_matrix_selected.png",
             "Test confusion matrix. Labels show counts and colour represents the fraction of each true class.", 13)
    r.figure(FIGS / "sample_predictions.png",
             "Examples of correct predictions and errors. Displayed softmax values have not been calibrated.", 16)
    r.h("4.1 Observations", 2)
    r.bullets(text["results_discussion"])
    cm = tm["confusion_matrix"]
    cat, dog = classes.index("cat"), classes.index("dog")
    r.p(f"The confusion matrix identifies {cm[cat][dog]} cats predicted as dogs and "
        f"{cm[dog][cat]} dogs predicted as cats: {100 * (cm[cat][dog] + cm[dog][cat]) / (analysis['test_images'] - analysis['correct']):.1f}% "
        "of all test errors. Shared appearance or difficult backgrounds are plausible explanations to "
        "investigate; the confusion matrix alone cannot identify their cause. Prediction examples illustrate "
        "individual outcomes, while the complete per-class table supports the aggregate conclusions.")
    r.p("Test accuracy exceeds validation accuracy by 1.16 percentage points. These are different image "
        "samples, so the difference does not establish improved generalization. High softmax confidence "
        "also does not establish calibration. The perfect motorbike recall concerns 118 images on this "
        "split, with potential related-image overlap considered in the sensitivity analysis.")
    r.h("4.2 Uncertainty and descriptive sensitivity", 2)
    acc_lo, acc_hi = analysis["accuracy_wilson_95"]
    f1_lo, f1_hi = analysis["macro_f1_bootstrap_95"]
    r.table(["Estimate", "Value", "95% interval"], [
        ["Accuracy (Wilson)", pct(analysis["accuracy"], 2), f"{pct(acc_lo, 2)} to {pct(acc_hi, 2)}"],
        ["Macro F1 (bootstrap)", f"{analysis['macro_f1']:.4f}", f"{f1_lo:.4f} to {f1_hi:.4f}"],
    ], [6.4, 4.0, 6.0], "Conditional uncertainty from the existing test predictions.")
    r.p("Accuracy uses a Wilson score interval (Wilson, 1927). Macro F1 uses 2,000 class-stratified bootstrap "
        "resamples with replacement, seed 42, retaining each class's observed count and taking the 2.5th and 97.5th "
        "percentiles (Efron, 1979). Both calculations condition on the fitted model and assume independent sampled "
        "images. They exclude seed, retraining, model-selection and distribution-shift uncertainty. Related images "
        "can violate independence and make these intervals too narrow.")
    r.table(["Descriptive subset", "Images", "Correct", "Accuracy"], [
        ["All test images", analysis["test_images"], analysis["correct"], pct(analysis["accuracy"], 2)],
        ["Average-hash candidates", *[analysis["sensitivity"]["hash_match_candidates"][k] for k in ("images", "correct")],
         pct(analysis["sensitivity"]["hash_match_candidates"]["accuracy"], 2)],
        ["Remaining test images", *[analysis["sensitivity"]["remaining_test_images"][k] for k in ("images", "correct")],
         pct(analysis["sensitivity"]["remaining_test_images"]["accuracy"], 2)],
    ], [7.0, 3.0, 3.0, 3.4], "Post hoc subset analysis using candidate flags from training or validation hash matches.")
    r.p("The subset analysis is post hoc. Candidate removal does not establish an independent holdout: the hash "
        "detector can miss related images and flag unrelated ones. The complete test set is the primary evaluation.")
    r.h("4.3 Further improvements", 2)
    r.bullets(text["results_improve"])

    r.h("5. Optimization and Insights")
    r.h("5.1 Architecture and tuning experiments", 2)
    for para in text["optimisation_experiments"]:
        r.p(para)
    configuration_rows = []
    for name in ("pixel_lstm", "cnn_only", "cnn_lstm", "cnn_bilstm", "tune_hidden256",
                 "tune_layers2", "tune_dropout03", "tune_lr3e4", "tune_no_aug", "tune_no_reg"):
        run = runs[name]
        model_cfg, train_cfg = run["model"], run["train"]
        recurrent = run["arch"] != "cnn"
        configuration_rows.append([
            name, model_cfg.get("hidden", "N/A"), model_cfg.get("layers", "N/A"),
            ("Bi" if model_cfg.get("bidirectional", False) else "Uni") if recurrent else "N/A",
            f"{model_cfg['dropout']:g}", "Yes" if train_cfg["augment"] else "No",
            f"{train_cfg['lr']:g}", f"{train_cfg['weight_decay']:g}",
        ])
    r.table(["Run", "Hidden", "Layers", "Dir.", "Dropout", "Aug.", "Peak LR", "Decay"],
            configuration_rows, [3.8, 1.6, 1.4, 1.3, 1.9, 1.5, 2.5, 2.4],
            "Complete experiment settings from saved configurations. Hidden units are per direction; "
            "dropout denotes sequence/head probability (head only for the CNN). Aug. means training "
            "augmentation; Bi/Uni mean bidirectional/unidirectional, and decay means AdamW weight decay. "
            "All runs use seed 42, batch size 64 and a "
            "20-epoch limit. N/A means not applicable.", align_right_from=1)
    r.table(["Run", "Parameters", "Epoch", "Val F1", "Test accuracy", "Test F1"], [
        [row["name"], f"{row['params']:,}", row["best_epoch"], f"{row['val_macro_f1']:.4f}",
         pct(row["test_acc"], 2), f"{row['test_macro_f1']:.4f}"] for row in summary["runs"]
    ], [4.0, 2.7, 1.4, 2.4, 3.0, 2.9],
        "All ten runs ranked by validation macro F1. F1 values are fractions; accuracy is a percentage.")
    r.figure(FIGS / "experiment_comparison.png", "Validation and test accuracy; ordering uses validation macro F1.", 15)
    r.figure(FIGS / "architecture_curves.png", "Validation histories of the four architecture configurations.", 15)
    r.figure(FIGS / "training_curves_baseline.png", "Bidirectional baseline histories for comparison with the selected run.", 16)
    gain = analysis["selected_minus_baseline_accuracy"]
    lo, hi = analysis["paired_accuracy_improvement_bootstrap_95"]
    r.p(f"The selected configuration improves test accuracy by {gain * 100:.2f} percentage points over the "
        f"bidirectional baseline. On the same images, it corrects {analysis['selected_correct_baseline_wrong']} "
        f"baseline errors and introduces {analysis['selected_wrong_baseline_correct']} errors where the baseline was "
        f"correct. A paired class-stratified bootstrap gives a conditional 95% difference interval of "
        f"{lo * 100:.2f} to {hi * 100:.2f} percentage points. This post hoc contrast is unadjusted for the search "
        "and does not isolate a causal effect of any single setting.")
    r.h("5.2 Observations", 2)
    r.bullets(text["optimisation_insights"])
    r.p("The search was a manually specified set of ten runs, rather than a grid search or a repeated-seed "
        "optimization. Increasing hidden size or depth tests extra recurrent capacity; reducing dropout "
        "tests a weaker masking setting; the lower learning rate tests slower updates within the same epoch "
        "budget. Disabling augmentation tests whether those transformations help on this split. The combined "
        "run tests a joint change and cannot identify the separate contribution of dropout or weight decay. "
        "The tied dropout parameter also changes multiple masking locations together.")
    r.p("The configuration without augmentation achieved validation macro F1 of 0.9342 versus 0.9040 for "
        "the bidirectional baseline; the combined run reached 0.9471. These observations suggest that the "
        "original regularization and augmentation settings may be too strong for this training budget, "
        "but they do not establish a general benefit from removing regularization. A follow-up factorial "
        "comparison should vary augmentation, dropout and weight decay independently, use comparable "
        "search budgets for each architecture, and select models before evaluating a new holdout.")

    r.h("6. Conclusion")
    r.h("6.1 Final insights", 2)
    r.bullets(text["conclusion_findings"])
    r.h("6.2 Recommendations", 2)
    r.bullets(text["conclusion_future"])

    r.h("References")
    references = json.loads((ROOT / "reports/references.json").read_text(encoding="utf-8"))
    for ref in references:
        para = d.add_paragraph()
        para.paragraph_format.left_indent = Cm(1.27)
        para.paragraph_format.first_line_indent = Cm(-1.27)
        para.paragraph_format.line_spacing = 2.0
        para.paragraph_format.keep_together = True
        for chunk, italic in ref:
            para.add_run(chunk).italic = italic

    r.h("Appendix A. Reproduction and evidence")
    r.p("Use Python 3.11 from the repository root and install the reviewed dependencies with "
        "python -m pip install -r requirements.txt. Read the primary image directory at "
        "data/natural-images/natural_images; resized arrays are cached at artifacts/cache/images_64px.npz. "
        "results/data_info.json records class order, normalization statistics and test paths. The deterministic "
        "indexing and seed-42 split reproduce the remaining split assignments. The historical dependency "
        "inventory is incomplete, so the reviewed package set is not a guarantee of identical retraining.")
    r.p("The notebook reruns preparation and checkpoint evaluation with RUN_TRAINING=False. Saved experiment "
        "results, predictions and weights are preserved. The current package inventory is recorded separately "
        "from historical run metadata.")
    r.p("Illustration-only exclusions are recorded in reports/display_exclusions.json. "
        "These exclusions do not alter split membership or reported metrics.")
    r.bullets([
        "Check implementation behavior: python -m pytest -q",
        "Recompute supplementary statistics from saved predictions: python reports/analyse_results.py",
        "Rebuild only this report from existing evidence: python reports/build_report.py",
    ])
    r.p("No training or notebook execution is needed to rebuild the report. To conduct new experiments, "
        "use separate output locations and preserve the current evidence; the canonical experiment runner "
        "caches runs by name and its --rerun option overwrites saved results. Only the selected model's "
        "checkpoint is retained. Other runs can be checked against saved predictions and histories, "
        "but their fitted weights cannot be re-evaluated without retraining.")
    r.p("Evidence: results/summary.json; results/runs/*/result.json and test_probs.npy; "
        "artifacts/checkpoints/best_model.pt; reports/audit_evidence/local_audit.json; "
        "reports/audit_evidence/publisher_verification.json; reports/analysis_results.json.")
    for section in d.sections:
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_field(footer, "PAGE", "1")
    black_text(d)
    output.parent.mkdir(parents=True, exist_ok=True)
    d.save(output)
    clean_template_properties(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT, help="Destination Word document")
    print(f"Report written: {build(parser.parse_args().output)}")
