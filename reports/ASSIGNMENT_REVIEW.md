# Assignment review - November 11, 2026

The notebook and report cover the supplied assignment instructions. The abstract appears in the report only. The notebook contains 300 words of final insights and recommendations in bullet points, followed by a separate conclusion as its last cell. Citations and the reference list are retained in the Word report only. The notebook and report identify Albert Kabore, PhD student at the University of the Cumberlands. All 27 notebook code cells execute without errors. The Word report contains every requested section, eight quantitative tables, nine figures, and APA references. The separate Neural Networks Implementation Project Rubric was not supplied, so any additional criteria in that document have not been assessed.

## Assignment coverage

| Requirement | Evidence |
| --- | --- |
| Collection | The linked Natural Images dataset is used consistently; duplicate archive copies and publisher-manifest verification are explained. |
| Preprocessing | RGB conversion, 64x64 resizing, stratification, training-only normalization and training augmentation are implemented and illustrated. |
| Model design | PyTorch CNN-LSTM, row sequence dimensions, architecture diagram and parameter summary are present. |
| Training | AdamW, one-cycle schedule, clipping, early stopping and checkpoint selection are explained. Baselines demonstrate dropout and batch normalization; the selected configuration is identified accurately. |
| Evaluation | Accuracy, precision, recall, F1, class supports, confusion matrix, prediction examples and conditional uncertainty are reported. |
| Optimization | Four architecture runs and six tuning runs are compared by validation macro F1. The final combined ablation is distinguished from five single-factor changes. |
| Reporting | Abstract, Introduction, Methodology, Model Implementation and Training, Results and Evaluation, Optimization and Insights, Conclusion, and References are complete. |

## Verification

- **17 implementation tests passed.**
- Every saved test metric for all ten experiments matches its prediction array.
- The selected checkpoint reproduces validation/test metrics and the exact saved test probabilities.
- Hashes of all **35 protected experiment files** remain unchanged.
- All 6,899 primary-copy source images decode; both archive copies match; every cached resize matches a fresh conversion.
- All ten findings sections contain **100 words** each. Final insights/recommendations contain **300 words** in both the notebook and the report conclusion, excluding headings and bullet markers.
- Wilson intervals match SciPy and macro F1 matches scikit-learn, including an absent-class case.
- Notebook schema, report sections, figures and document watermark objects were checked. Report text and figure annotations use black; chart shading is grayscale. The report PDF and its export workflow have been removed.

## Watermarks

Two notebook illustrations contained source text overlays: `cat/cat_0757.jpg` and `flower/flower_0575.jpg`. Their displayed examples were replaced or omitted using the rules in `display_exclusions.json`. Source photographs, split membership, predictions and metrics are unchanged. Internal editing markers, workflow commentary and repeated citation paragraphs were removed from the notebook. Section headings describe the study, data, model and results. The generated Word report has no watermark objects. It identifies Albert Kabore as the student author and adds no tool attribution.

## Findings and study limits

The selected model correctly classifies **997/1,035** test images: **96.33% accuracy**, **96.04% macro precision**, **96.01% macro recall**, and **0.9602 macro F1**. Cat/dog confusion accounts for **27 of 38** errors. Conditional 95% intervals are **95.00%-97.31%** for accuracy and **0.9481-0.9718** for macro F1.

There are 39 test images sharing average hashes with training images, or 41 when validation is included. Hash matches are similarity candidates, with some related views and some false positives on visual screening. All 41 flagged images are classified correctly; the remaining images score **956/994 = 96.18%**. This post hoc subset does not create an independent holdout or correct possible leakage.

One seed, unequal capacity and tuning coverage, a 20-epoch cap, and the combined augmentation/dropout/weight-decay change restrict general conclusions. Intervals exclude seed, retraining, selection and distribution-shift uncertainty; related images can violate independence. Test metrics were exposed after each experiment, despite validation-based ranking. Other runs lack retained checkpoints and were checked using predictions and logs. Recorded elapsed durations total 26.15 hours, including unexplained intervals; active computation time is unknown.

The report recommends verified image groups, an external holdout, repeated seeds, separate ablations, comparable tuning budgets, and targeted animal-class error analysis. It claims no new training run or revised-split performance.

## Evidence

See `audit_evidence/local_audit.json`, `publisher_verification.json`, `pre_edit_hashes.json`, `environment_review.json`, and `analysis_results.json`. Shared narrative and APA references are in `report_text.json` and `references.json`. The initial review is preserved at `audit_evidence/initial_assignment_review.md`. Reproduction commands are in the repository README; notebook execution uses `RUN_TRAINING=False`.
