# LSTM Implementation Project: CNN-LSTM Image Classification

Image classification on the [Kaggle Natural Images dataset](https://www.kaggle.com/datasets/prasunroy/natural-images)
(8 classes, 6,899 photographs) with a CNN feature extractor followed by a bidirectional LSTM that reads the
feature map row by row. Built with PyTorch and trained on a CPU.

## Repository layout

| Path | What it holds |
| --- | --- |
| `cnn_lstm/data.py` | Dataset indexing (with duplicate removal), stratified split, resize cache, normalisation, augmentation |
| `cnn_lstm/models.py` | `CNNLSTM` (main model), `PixelLSTM` and `CNNClassifier` baselines, model summary |
| `cnn_lstm/train.py` | Training loop: AdamW, one-cycle LR, gradient clipping, early stopping |
| `cnn_lstm/metrics.py` | Accuracy, precision, recall, F1 (macro, weighted, per class), confusion matrix |
| `cnn_lstm/experiments.py` | All experiments (architecture comparison and tuning), model selection |
| `cnn_lstm/plots.py` | Report figures |
| `notebooks/LSTM_Image_Classification.ipynb` | End-to-end walkthrough with outputs |
| `notebooks/LSTM_Image_Classification.html` | Executed notebook for browser viewing |
| `reports/LSTM_Project_Report.docx` | Complete Word report with figures, tables and APA references |
| `reports/ASSIGNMENT_REVIEW.md` | Assignment coverage, checks and remaining study limitations |
| `tests/` | Unit tests (pytest) |
| `results/` | Per-run histories and metrics, `summary.json`, `figures/`, training log |
| `data/` | Raw dataset, git-ignored and never modified |
| `artifacts/` | Image cache and best checkpoint, git-ignored |

## Quick start

```bash
pip install -r requirements.txt

# 1. Download the dataset unchanged (public, no Kaggle API key needed)
python -c "import kagglehub; kagglehub.dataset_download('prasunroy/natural-images', output_dir='data/natural-images')"

# 2. Optional: train experiments that do not already have saved results
# Recorded elapsed durations of the existing runs total 26.15 hours, including unexplained long intervals.
python -m cnn_lstm.experiments

# 3. Figures for the report
python -m cnn_lstm.plots

# 4. Tests
python -m pytest

# 5. Execute the notebook using saved results and the selected checkpoint; no training
python reports/complete_notebook.py

# 6. Build the Word report
python reports/build_report.py
```

Use `python -m pytest -q` for a quick implementation check. Avoid using `--rerun` with an epoch override on
canonical experiments: that command replaces their saved results. Existing runs are cached by name; changed
training settings require a separate experiment directory and fresh validation selection.

## Data handling

* The Kaggle archive stores the dataset twice (`natural_images/` and `data/natural_images/`). Only
  `natural_images/` is read.
* Two byte-identical image pairs (`car_0236`/`car_0675`, `flower_0400`/`flower_0772`) are each indexed once.
  This removes exact copies, while related views may still cross split boundaries. Source files remain unchanged.
* Split: stratified 70/15/15 (4,827 / 1,035 / 1,035 images), seed 42. Normalisation statistics come from the
  training split only. The model is selected on validation macro-F1, and the test split is used only for reporting.

## Results

Validation macro F1 selects `tune_no_reg`: a CNN with bidirectional LSTM, no augmentation, dropout or weight
decay, and retained batch normalization. It correctly classifies 997/1,035 test images (96.33% accuracy), with
macro precision 96.04%, macro recall 96.01%, and macro F1 0.9602. Cat/dog confusion accounts for 27 of 38 errors.

The original image split contains 41 test images sharing average hashes with training or validation images.
These are similarity candidates requiring visual review. One seed, unequal capacity and tuning budgets, and
possible related-image overlap limit conclusions. The report includes conditional uncertainty and a descriptive
subset analysis; neither establishes performance on independent photographs.

`reports/analyse_results.py` regenerates supplementary statistics from saved predictions.
`reports/audit_assignment.py` checks source images, preprocessing, metrics, and the selected checkpoint.
The historical run metadata records an incomplete environment; `requirements.txt` describes the checked
review environment, and `reports/audit_evidence/environment_review.json` records its package inventory.
