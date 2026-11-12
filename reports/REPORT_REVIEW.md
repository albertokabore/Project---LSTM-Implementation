# Report-only review - November 11, 2026

The report was checked against all eight reporting requirements supplied by the student. The initial content review preserved the existing figures, model code, saved predictions, experiment metrics and checkpoints. During that initial review, the notebook was neither executed nor modified.

The current submission is [LSTM_Project_Report_Final.docx](LSTM_Project_Report_Final.docx). A subsequent visual revision added color to seven charts, including the confusion matrix, learning curves, class distribution, experiment comparison and architecture diagram. Training uses blue, validation uses orange and test results use green. The two photographic illustrations, quantitative tables and recorded results are preserved. The report ends with References. Color and data checks for that earlier visual revision are recorded in `audit_evidence/color_revision_validation.json`.

The report now has three separate closing sections: **6. Final Insights**, **7. Recommendations**, and **8. Conclusion**. Final Insights interprets the findings; Recommendations gives future actions; the separate prose Conclusion synthesizes the project's purpose, workflow and results. The Conclusion is the last narrative section before References. Final Insights and Recommendations retain 300 words combined. Their report-specific text and the distinct conclusion are stored in `report_closing.json`; the notebook and shared narrative source were not modified for this correction.

At the student's subsequent request, the notebook's plotting sources and six embedded figures were also updated to match the report palette. All numerical outputs, written analysis, non-plot code and photographic examples are preserved. No training or full notebook execution was performed. Rebuilding the notebook now retains the color settings. The focused refresh command is `python reports/refresh_notebook_colors.py`; preservation checks are recorded in `audit_evidence/notebook_color_validation.json`.

| Required section | Coverage in the revised report |
| --- | --- |
| Abstract | Concise purpose, dataset, methods, experiment count, selected configuration, quantitative results and principal limitations. |
| Introduction | CNN filters and pooling; LSTM memory and gates; bidirectional recurrence; the rationale for spatial row sequences; objectives and supporting literature. The report identifies the task as classification of individual still images. |
| Methodology | Dataset source, labels, image quality, exact duplicates, class counts, two-stage stratified split, resizing, training-only normalization, augmentation, evaluation independence and reasons for the study design. |
| Model Implementation and Training | Architecture diagram, tensor dimensions, CNN block parameter counts, baseline designs, dropout placement, loss, seeds, mini-batch procedure, optimizer, scheduler, gradient clipping, checkpoint restoration, early stopping and computational constraints. |
| Results and Evaluation | Validation/test and per-class tables; training curves; confusion matrix; prediction examples; uncertainty and descriptive sensitivity analysis; error interpretation and specific improvements. |
| Optimization and Insights | A complete settings table and ranked results for all ten experiments; single-factor and combined changes; interpretation of tuning outcomes; fair-comparison limits and proposed controlled follow-up experiments. |
| Final Insights and Recommendations | Separate sections interpret the findings and propose future actions, retaining 300 words combined. |
| Conclusion | A distinct prose synthesis of the purpose, workflow, principal results, contribution and future direction. It is the final narrative section before References. |
| References | Eighteen alphabetized APA-style entries with in-text citations, hanging indents, double spacing, italics and source links. Conference entries include additional publisher, editor and volume details where applicable. |

## Material improvements

- Explained why the CNN-LSTM design, training-only normalization, stratification and macro F1 fit the study.
- Expanded the model summary and described all comparison architectures.
- Clarified that checkpoint selection within a run uses validation loss, while configuration selection uses validation macro F1. All runs reached 20 epochs; the selected weights came from epoch 17.
- Added the complete experiment configuration table so run names can be interpreted without opening the notebook or code.
- Interpreted the validation-loss plateau, training-to-validation gap, directional cat/dog errors, and the difference between validation and test accuracy.
- Distinguished observations from explanations that still require controlled experiments.
- Removed the reproduction and internal-verification appendix following the student's feedback. The academic report ends with References and retains the eight required reporting sections. Report generation consumes existing analysis without overwriting it or executing the notebook.

## Verification

The document contains all eight required sections, nine numbered tables and nine numbered figures. Dataset counts, configuration settings and ranked results match their saved sources. All scalar test metrics, per-class metrics and confusion matrices for all ten runs were independently recomputed from saved predictions; the maximum scalar difference was zero. The selected model's parameter counts agree with its implementation.

Every reference is cited in the report. Citation matching, reference order, hanging indents, italics, double spacing, image identity, table and image dimensions, black document text and absence of watermark objects were checked. The existing architecture and selected-run training-curve images were visually inspected. Pagination was not rendered in Word during this review.

During the initial content review, SHA-256 comparisons confirmed that the notebook, its then-existing HTML export, model code, saved analysis, figures, experiment evidence and checkpoints were unchanged. Of 55 snapshotted files, only the intentionally revised report reference source changed; the other 54 were identical. Those historical checks and that report version's hash are recorded in `audit_evidence/report_review_validation.json`, with the initial snapshot in `audit_evidence/report_review_before.json`. The later color revision separately confirms that all 28 protected notebook, result, prediction, checkpoint and report-content files remain unchanged.

The report retains the study's material limits: one seed, unequal capacity and tuning coverage, possible related-image overlap, test metrics recorded during the search, and incomplete historical environment information. No new training runs or performance claims were introduced. The supplied reporting requirements are covered; a separate grading rubric was not supplied.
