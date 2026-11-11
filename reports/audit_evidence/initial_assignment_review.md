**Assignment review — November 11, 2026**

The implementation and saved results have substantial, reproducible supporting evidence. I found no evidence of fabricated experimental data or invented scores in the checks performed. The submission is nevertheless **not ready to submit**: the report is missing, the notebook is incomplete, several statements need correction, and visually similar images cross the split boundaries.

This is a review against the instructions provided in the conversation. The separate *Neural Networks Implementation Project Rubric* was not available, so compliance with its additional criteria cannot be certified. Original assignment code, notebook, raw data, checkpoint, and experiment results were left unchanged. The new files in `reports/` document this review.

**What was actually verified**

| Check | Result and evidence |
| --- | --- |
| Dataset identity | Kaggle's public manifest was fetched across 69 API pages. All 13,798 local filenames and file sizes matched; total size 359,403,192 bytes. No missing, extra, or differently sized files. See `audit_evidence/publisher_verification.json` and `publisher_manifest.json`. |
| Published dataset description | The [Kaggle data card](https://www.kaggle.com/prasunroy/natural-images/notebooks) describes 6,899 images in the same eight classes found locally. |
| Raw image validity | Every image in the primary 6,899-image copy fully decoded: 6,898 JPEGs and one PNG, all RGB, with no unreadable files. The two archive copies are byte-identical for every corresponding image. |
| Exact duplicates | Two duplicate pairs were found: `car_0236`/`car_0675` and `flower_0400`/`flower_0772`. The pipeline indexes each pair once, leaving 6,897 images. No identical file hashes or identical decoded RGB images cross the current train, validation, and test splits. |
| Cached images | Every one of the 6,897 cached arrays exactly matched a fresh RGB conversion and 64×64 bicubic resize of its source image. |
| Split and normalization | Recreated the seed-42 stratified split: 4,827 training, 1,035 validation, and 1,035 test images. All class counts and test-path ordering matched. Recomputed training-only channel means and standard deviations exactly matched the saved values. |
| Saved predictions | Recomputed every saved test metric, including per-class scores and confusion matrices, for all ten experiments. No discrepancies. All probability arrays are finite, in range, and sum to one within floating-point tolerance. |
| Selected weights | Reloaded `best_model.pt` and evaluated validation and test images. Both metric dictionaries matched. The new test probability array was exactly equal to the saved array, with zero differing predicted labels. |
| Histories and summaries | All parameter counts, history lengths, best-epoch choices, rounded epoch log values, summary fields, experiment architectures, and validation ranking matched their corresponding saved evidence. |
| Existing tests | `python -m pytest -q`: **17 passed**. |

The manifest comparison verifies filenames and sizes, not cryptographic identity with the publisher: the manifest supplies no publisher hashes. Local SHA-256 inventories identify the reviewed files but cannot establish their historical origin. The other nine experiments have no retained weights, so their scores were verified against prediction files and logs, not by replaying their original checkpoints. No experiments were retrained during this review. These are limits on the audit, not evidence of fabrication.

Synthetic images appear in the unit tests as explicitly generated fixtures. The experiment pipeline loads the Kaggle images, not those fixtures. Synthetic test fixtures are not fabricated experimental observations.

**Verified results that can be reported**

The selected configuration is `tune_no_reg`, a CNN plus bidirectional LSTM with 128 hidden units per direction, one recurrent layer, and 1,503,208 trainable parameters. It ran for 20 epochs, and its epoch-17 weights were restored using the validation-loss rule. Runs were then ranked by validation macro-F1; these are two distinct selection decisions.

| Metric | Validation | Test |
| --- | ---: | ---: |
| Accuracy | 95.1691% | 96.3285% |
| Macro precision | 94.7512% | 96.0357% |
| Macro recall | 94.7030% | 96.0058% |
| Macro F1 | 0.947081 | 0.960176 |
| Top-2 accuracy | 98.9372% | 99.4203% |

The selected model correctly classified 997 of 1,035 test images. Dog F1 was 0.8421 and cat F1 was 0.8848. Cat-to-dog and dog-to-cat errors account for 27 of the 38 mistakes. Report these weaknesses alongside the overall score.

| Architecture comparison, before tuning | Parameters | Test accuracy | Test macro F1 |
| --- | ---: | ---: | ---: |
| Pixel LSTM | 315,016 | 82.6087% | 0.810150 |
| CNN only | 305,128 | 87.6329% | 0.844821 |
| CNN + unidirectional LSTM | 895,976 | 89.8551% | 0.875548 |
| CNN + bidirectional LSTM | 1,503,208 | 92.1739% | 0.912627 |

These are observations from one split and one seed. The models differ substantially in parameter count, and the CNN baseline was not given the same tuning search as the hybrid. The comparison supports a gain for these tested configurations; it does not isolate recurrence as its sole cause or establish universal superiority.

**Findings that require attention**

1. **High priority: finish the required submission.** The notebook contains 25 code cells, of which only 16 have execution counts. Training-result display, checkpoint evaluation, and optimization cells have not been executed. Four narrative cells still say “filled in after execution” (zero-based cells 35, 41, 46, and 47). It has no abstract or full references section. `reports/LSTM_Project_Report.docx` and `reports/report_text.json` do not exist. The report builder explicitly refuses to build without its discussion text. A report-generation script is not a completed report. Only the architecture figure currently exists in `results/figures/`; the remaining report figures have not been generated.

2. **High priority: qualify the independence of the evaluation.** There are 39 test images sharing an identical 8×8 grayscale average hash with a training image: 36 fruit and three motorbike images. Including validation raises this to 41 test images: 38 fruit and three motorbikes. Visual inspection of the saved contact sheet shows several very similar pairs, including `fruit/fruit_0972.jpg` versus `fruit/fruit_0021.jpg`, and `motorbike/motorbike_0273.jpg` versus `motorbike/motorbike_0272.jpg`. The detector also produces false positives, such as similarly shaped objects with different colors; a hash match alone does not prove duplication. No exact decoded duplicates cross the splits. The justified concern is potential near-duplicate or object-instance overlap, which may make the score optimistic for independent-image generalization. Its numerical effect has not been measured. Do not claim that byte-level deduplication guarantees that no picture or closely related view can occur in multiple splits.

   Review candidate families visually and, where appropriate, assign related images to groups before making a stratified group split. Freeze the revised protocol before fitting and rerun the architecture comparison and tuning. Disclose that this redesign follows review of the original evaluation; a truly new external holdout would provide stronger evidence. Do not simply remove all hash collisions and reuse the old score, or label the 39/41 candidates as 39/41 proven duplicate images.

3. **High priority: describe the winning configuration truthfully.** The baseline uses augmentation, dropout, and weight decay. The selected model uses **no augmentation, dropout 0.0, and weight decay 0.0**; batch normalization remains. The assignment's requested techniques were implemented and tested, but this distinction must be explicit. In `build_report.py`, the hard-coded training description at lines 407–410 would incorrectly assign weight decay 1e-4 and augmentation to the selected run. The table caption at lines 469–470 incorrectly says all training accuracies were measured with augmentation and dropout active. Correct these descriptions before building the report. “No augmentation, dropout, or weight decay” is more precise than “no regularization,” because other training constraints remain.

4. **Medium priority: correct unsupported experimental interpretations.** The notebook calls the search one factor at a time, but `tune_no_reg` changes augmentation, dropout, and weight decay together relative to the baseline. The preceding `tune_no_aug` controls augmentation only; changing dropout and weight decay together from it still does not isolate either effect. Describe the final run as a combined ablation. Do not claim that any single disabled regularizer caused its improvement, that augmentation generally harms classification, or that 100% training accuracy proves absence of overfitting. The restored model's recorded training accuracy is 100% versus 95.17% validation accuracy. Training histories are measured during optimization, with each run's own augmentation/dropout settings, rather than a uniform evaluation-mode pass over the training set.

5. **Medium priority: make runtime claims match the observations.** README's “about 2.5 h” for all experiments is not the observed elapsed total: the saved ten runs sum to approximately **26.15 hours**, with extremely long individual epochs. Neither that total nor an explanation such as machine sleep should be presented as active compute time without evidence. The selected run records 25.37 minutes. Median epoch durations for the CNN-based runs range from about 61 to 93 seconds; the notebook's “56 to 62 seconds” does not represent them generally. State recorded elapsed durations, and explicitly say the unusually long intervals were not diagnosed. `perf_counter` values alone do not establish their cause.

6. **Medium priority: fix the environment and rerun instructions.** `requirements.txt` labels its pins as the versions used for the results but specifies NumPy 2.3.4 and pandas 2.3.3; the reviewed environment has 2.3.2 for both. Historical versions cannot be inferred conclusively from today's installation, because the run metadata records neither package. Record a complete environment for future runs and distinguish it from the existing historical evidence. PyTorch 2.14.0+cpu is actually installed and recorded locally; it should not be declared invented merely because of the version number. The report generator requires `python-docx`, which is absent from both dependency declarations. Add a report dependency or extra. The documented smoke command uses `--rerun` with a one-epoch override and overwrites the canonical `cnn_bilstm` result; subsequent default runs skip any existing result solely by its name. Use isolated smoke-test output and validate configuration/data fingerprints before accepting cached runs. `save_if_best` can also retain an old checkpoint after a lower-scoring rerun replaces its result files. The current checkpoint and results match, but this behavior can break that agreement on later runs.

7. **Medium priority: revise overly strong theoretical claims.** `build_report.py:372` says bidirectionality gives top and bottom “equal weight.” It makes both directions available; learned weights do not guarantee equal influence. The notebook says a wide airplane image is “squashed vertically” by square resizing; its width-to-height ratio instead decreases toward one. Say that the resize distorts its proportions. Distinguish feature-map stride from receptive field: centers are eight input pixels apart, but each final convolutional cell has a nominal 36×36 receptive field, not an eight-pixel view. Global average pooling removes explicit final spatial indices, but earlier convolutions can still encode spatial context. ReNet is related recurrent image analysis, not a validation of this exact CNN–row-LSTM architecture. Its [original paper](https://arxiv.org/abs/1505.00393) describes four directional recurrent sweeps replacing convolution/pooling layers.

8. **Medium priority: state what the test protocol establishes.** The code ranks runs using validation macro-F1, which is appropriate. It nevertheless evaluates and logs the test set after every experiment, before the entire process finishes. That does not prove test-driven tuning occurred, and the saved code does not select by test score. It also cannot prove the stronger narrative claim that test results never influenced any human decision. Report the implemented procedure accurately. For a stronger future protocol, freeze the design and validation selection before exposing final test results. Retain validation predictions and per-run checkpoints if storage allows.

9. **Lower priority: retain evidence for provenance statements and clean up presentation.** The notebook's collection cell prints a fixed “Kaggle's published file list” total but does not query or compare the manifest. Its claim of full publisher verification was unsupported by that cell. The new manifest audit now supplies that evidence for this review date; link it rather than implying the original cell performed the check. Replace the corrupted replacement character before “20%” in the augmentation description with “±”. The training module's opening docstring should describe its actual `OneCycleLR` policy with cosine annealing, rather than leaving readers to infer a simple cosine-only schedule.

**Coverage of the supplied assignment instructions**

Natural Images is explicitly linked in the assignment and CIFAR-10 is described as an example. Using Natural Images is therefore consistent with the supplied instructions. Do not call this a CIFAR-10 experiment or describe its eight classes as CIFAR-10's ten classes. The [CIFAR-10 publisher page](https://cave.cs.toronto.edu/kriz/cifar.html) describes a different dataset: 60,000 images at 32×32 pixels and ten classes.

| Required workflow | Assessment |
| --- | --- |
| Data collection | Implemented; dataset and current publisher-manifest match verified. |
| Resize, normalization, augmentation | Implemented; normalization and cached images verified. Selected run disables augmentation and must be identified as such. |
| CNN–LSTM design in a deep learning framework | Implemented correctly in PyTorch, including meaningful row sequences and both directional variants. |
| Training with batch normalization/dropout techniques | Demonstrated in baseline and tuning runs. Selected run retains batch normalization and disables dropout. |
| Accuracy, precision, recall on test data | Implemented and reproduced; include the similarity-overlap limitation. |
| Hyperparameter and architecture optimization | Ten actual saved experiments with consistent evidence; combined ablation and unequal model capacity limit causal conclusions. |
| Comprehensive reporting | Incomplete. |

| Required report section | Current submission status |
| --- | --- |
| Abstract | Missing from the notebook; report has not been built. |
| Introduction | Present in notebook and report template; wording refinements noted above. |
| Methodology | Present; add similarity overlap and precise split limitations. |
| Model Implementation and Training | Code and summaries available; notebook result cells and observations unfinished. |
| Results and Evaluation | Saved results are valid; notebook evaluation cells unexecuted and narrative unfinished. |
| Optimization and Insights | Experiments exist; outputs and interpretation unfinished. |
| Conclusion | Placeholder only. |
| APA References | Citations and a reference list exist in the generator source; the notebook lacks a full reference list and no report has been generated. |

**References and factual support**

The inspected references correspond to identifiable scholarly works; I did not identify an invented paper title. Existence of a paper does not establish that every claim attributed to it is supported, or that every bibliographic field has been verified. The following primary or original-paper sources support the key identities and topics checked:

| Citation in the project | Source checked and scope |
| --- | --- |
| Roy dataset and Roy et al. | [Kaggle data card](https://www.kaggle.com/prasunroy/natural-images/notebooks), local/publisher manifest comparison, and [original dataset-associated paper](https://arxiv.org/abs/1807.10108). The latest paper revision is newer than its 2018 first submission; identify the version actually cited. |
| Hochreiter & Schmidhuber (1997) | [Original paper on the author's university site](https://ml.jku.at/publications/older/2604.pdf) confirms the authors, title, journal, volume, issue, pages, and original memory-cell formulation. |
| LeCun et al. (1998) | [Coauthor's publication page](https://bottou.org/papers/lecun-98h) confirms authors, title, journal, year, volume, issue, and pages. |
| Ioffe & Szegedy (2015) | [PMLR paper record](https://proceedings.mlr.press/v37/ioffe15.html). |
| Krizhevsky (2009) / CIFAR | [Publisher's dataset page](https://cave.cs.toronto.edu/kriz/cifar.html), including its technical-report link. |
| Donahue et al. (2015) | [Original CVPR paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Donahue_Long-Term_Recurrent_Convolutional_2015_CVPR_paper.pdf). |
| Loshchilov & Hutter | [Authors' paper](https://arxiv.org/abs/1711.05101) confirms identity and AdamW topic; cite the ICLR publication consistently if using 2019. |
| Paszke et al. (2019) | [NeurIPS paper record](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html) confirms all 21 authors and paper identity. APA's long-author-list format must preserve the first 19 authors and last author. |
| Pedregosa et al. (2011) | [JMLR paper record](https://jmlr.org/papers/v12/pedregosa11a.html). |
| Schuster & Paliwal (1997) | [Original IEEE paper hosted by CMU](https://deeplearning.cs.cmu.edu/F20/document/readings/Bidirectional%20Recurrent%20Neural%20Networks.pdf). |
| Shi et al. | [Authors' preprint](https://arxiv.org/abs/1507.05717) confirms the work and its scene-text recognition setting. The [published paper's abstract record](https://pubmed.ncbi.nlm.nih.gov/28055850/) confirms the 2017 journal date, authors, volume, issue, pages, and DOI. Do not mix the 2015 preprint date with a 2017 journal citation. |
| Smith & Topin | [Authors' preprint](https://arxiv.org/abs/1708.07120) confirms identity and the one-cycle topic. The supplied 2019 SPIE DOI page could not be read; exact SPIE metadata was not fully verified. Cite the verified preprint version if the proceedings record cannot be checked. |
| Srivastava et al. (2014) | [JMLR paper record](https://jmlr.org/papers/v15/srivastava14a.html). |
| Tompson et al. (2015) | [CVPR paper record](https://openaccess.thecvf.com/content_cvpr_2015/html/Tompson_Efficient_Object_Localization_2015_CVPR_paper.html), plus the original paper's spatial-dropout discussion. |
| Visin et al. (2015) | [Authors' ReNet paper](https://arxiv.org/abs/1505.00393). |

The 1997 original paper describes input and output gates around a memory cell with a constant self-connection; it does not establish the report's modern learned forget-gate formulation. Describe the displayed equations as the modern LSTM formulation and add an appropriate, verified forget-gate source rather than attributing every modern detail solely to 1997.

Before submission, add the reference list to the actual report/notebook, match every in-text citation to an entry, use sentence-case article titles, italicize journal titles and volume numbers, use hanging indents, and include persistent links where available. Cite any additional sources used to write the final discussion. A bibliography stored only in Python code does not satisfy the delivered report's references requirement.

**Suggested accurate results paragraph**

> On the original stratified seed-42 split of 6,897 distinct files, the validation-selected CNN–BiLSTM achieved 96.33% test accuracy (997/1,035), with macro precision 96.04%, macro recall 96.01%, and macro F1 0.9602. The selected configuration disabled augmentation, dropout, and weight decay while retaining batch normalization. Reloading the saved checkpoint reproduced the test predictions exactly. These results describe one image-level split; potential near-duplicate or related-view overlap, particularly in the fruit class, limits conclusions about performance on independent images.

**How to inspect or reproduce this review**

Run `python reports/audit_assignment.py` for local image, cache, prediction, and checkpoint checks. Run `python reports/audit_assignment.py --verify-manifest` for the read-only online publisher-manifest comparison. These commands write only under `reports/audit_evidence/` and do not retrain models. `consistency_checks.json` records the additional summary/log comparison performed during this review. `candidate_train_test_pairs.png` contains the visual comparison used here. SHA-256 inventories record the audited source/result/checkpoint files.

Complete the report and notebook using these verified values, correct the listed statements, and address or explicitly disclose the similarity-overlap issue. Any new experiment must produce its own measured results; do not carry these scores forward as if they came from a redesigned split or modified model.
