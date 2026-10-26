# DSA Land-Cover Feature Selection: Experimental Audit and Final Evaluation

**Date:** 26 October 2026
**Dataset:** UCI Statlog Landsat Satellite
**Project:** Graph-based feature selection, PCA, kNN, and PSO

## Scope and audit findings

This report records a direct audit of the repository, scripts, tests, result CSVs and logs, README, notes, data split protocol, and the existing Word report. The tracked Git branch was `main` at `origin/main`; all local experiment scripts, tests, data, and result files inspected were untracked user files. They have been preserved. New analysis files use V5/final-test names.

The data files contain 4,435 training rows (`sat.trn`) and 2,000 official test rows (`sat.tst`), each with 36 numeric predictors and six labels. The recent PSO V4 scripts use `sat.trn` only. However, this project has already used `sat.tst` for older correlation-threshold and PCA comparisons: the original scripts explicitly load it, `results/fs_experiment.csv` and `results/pca_experiment.csv` contain the resulting metrics, and the Word report describes them as official-test results. Those historical results remain in place and are not treated as validation evidence for the new PSO evaluation.

The V4 PSO runs also repeatedly optimized macro-F1 on the same 20% validation fold used for the reported performance. Therefore, those four V4 scores are useful for a preliminary seed/stability audit but are optimistic estimates of generalization. The standalone V4 script has fixed constants, so repeating it requires editing the seed in the source; the four historical runs themselves are preserved.

## Historical V4 seed and feature stability

The four saved V4 runs (seeds 7, 21, 42 and 84) each completed 110 objective evaluations with no failed evaluations. Their feature subsets contained 15, 15, 15 and 16 predictors. On the same stratified validation splits, the newly replayed full-feature custom-kNN baselines produce the paired comparison in `results/pso_paired_baseline_v5.csv`.

Across those four splits, PSO's mean paired macro-F1 change is **+0.012616** (1.26 percentage points), and PSO is higher on **3 of 4**. Mean pairwise Jaccard similarity of selected subsets is **0.276236** (range **0.153846–0.428571**), showing that predictive scores can be similar while the particular selected variables vary by seed. Per-feature selection frequencies are in `results/pso_feature_frequency_v5.csv`.

These are descriptive results from repeated single splits, not independent test estimates. The average gain has a wide uncertainty with only four splits, and PSO selection and evaluation share each split.

## Nested cross-validation protocol

The new `scripts/run_pso_nested_cv_v5.py` performs five-fold stratified outer cross-validation on `sat.trn`. For each fold:

1. The outer validation fold is withheld from PSO search and all fitted preprocessing.
2. PSO optimizes on an inner stratified split taken only from the outer training rows.
3. A scaler and custom kNN are refitted on all outer training rows.
4. The chosen feature subset and the full-feature baseline are evaluated on the same outer validation fold.

The fixed objective is `0.9 × (1 − macro-F1) + 0.1 × (selected feature count / 36)`, with kNN `k=5`. To make the nested run practical, it uses 5 PSO epochs and a population of 8 (48 objective evaluations per outer fold in the installed MEALPY version). This budget is lower than V4's 10×10 budget, so the cross-validation estimate evaluates the same objective and classifier under a smaller search budget. Seeds are deterministically derived from the outer seed and fold number. No official test data is loaded by this script.

**Results:** PSO macro-F1 averaged **0.862013 ± 0.015956** (sample SD) across the five outer folds, versus **0.884349** for the paired full-feature baseline. Mean paired difference was **−0.022336** (−2.23 percentage points); PSO was higher on **0 of 5** folds. The selected feature counts were 17, 10, 14, 12 and 18. Feature 15 was selected in all folds; features 8 and 25 were each selected in four of five folds. The current search budget did not match the baseline on outer validation data.

Fold-level rows: `results/pso_nested_cv_v5.csv`
Feature frequencies across outer folds: `results/pso_feature_stability_v5.csv`
Execution log: `results/pso_nested_cv_v5.log`

## Final official-test evaluation

After the nested analysis, freeze the method and run `scripts/run_pso_final_test_v1.py` once. It uses the V4 search budget (10 epochs, population 10), seed 42, 20% stratified inner validation, custom kNN `k=5`, and the same objective as above. Search, scaler fitting, feature-mask selection, and final training all use `sat.trn`. The script logs the frozen mask and settings before it loads `sat.tst`, then reports one full-feature reference and one selected-feature model from the same final evaluation stage. A header-only output guard prevents an accidental second official-test run.

**Final results (one frozen evaluation stage):** The 10×10 PSO search completed 110 objective evaluations with no failures and selected 15/36 features: `3;4;8;9;11;13;14;15;17;20;22;29;30;31;33` (zero-based). On `sat.tst`, all-features custom kNN reached **0.904500 accuracy**, **0.891554 macro-F1**, and **0.888575 balanced accuracy**. The PSO subset reached **0.894500**, **0.880541**, and **0.877842**, respectively. The selected model is lower by **1.00 accuracy point** and **1.10 macro-F1 points**, while using 58.3% fewer features. This does not support an accuracy improvement from the current PSO procedure.

Output: `results/pso_official_test_v1.csv`
Log: `results/pso_official_test_v1.log`

This test result is a single estimate from the official split and is not used to change feature-selection weights, PSO budget, random seed, or other method choices. Since earlier project work already reported results on `sat.tst`, it is held out from this new PSO selection procedure but is not a pristine, never-before-inspected test set for the project as a whole. Any future method changes require a new training/validation study and preferably a genuinely independent dataset for an independent final estimate.

## Repository checks and limitations

- Existing suite: **22 passed** after the changes (2.50 seconds).
- Python compilation succeeded for every `.py` file under `src/` and `scripts/`; both new runner `--help` checks succeeded.
- Historical repeated-seed recheck: paired mean macro-F1 gain **+0.012616** (3/4 splits); mean subset Jaccard **0.276236**.
- Nested five-fold run: PSO mean macro-F1 **0.862013 ± 0.015956**, full-feature baseline **0.884349**, mean paired change **−0.022336** (0/5 wins).
- The custom kNN implementation is intentionally retained for consistency with the DSA project, but it makes PSO evaluations slow.
- Five outer folds provide a more defensible estimate than a single split, but the folds are correlated because they come from one dataset. The study still covers one dataset and one official split.
- The reduced 5×8 nested search may find weaker masks than the final 10×10 search. It estimates the behavior of the defined procedure at the stated budget, not a guarantee of the final fitted subset's score.
- The existing Word report is an earlier project report and contains historical test-set comparisons and draft text. It is preserved; this versioned report documents the new protocol and supersedes it for the new PSO evidence.

## PSO selection of the PCA component count with SVD

On 27 October 2026, `scripts/run_pso_pca_svd.py` was added as a separate experiment for the component-count assignment. Mealpy PSO searches integer component counts from 1 to 36. PCA directions are computed using SVD after a scaler fitted on the current training partition. The classifier is scikit-learn kNN (`k=5`) for practical evaluation speed. The objective maximizes macro-F1; exact ties prefer fewer components. The official `sat.tst` split is never loaded.

The first larger run used seed 42, three stratified outer folds, five PSO epochs, and a population of eight. It completed in **1.63 seconds** on the measured machine. The fold searches selected **25, 23, and 12** components; outer-fold macro-F1 averaged **0.878790 ± 0.017458** (sample SD). A separate training-only search on the full training split proposed **18** components, with inner validation macro-F1 **0.886377**. That inner score was used during selection and is not an independent estimate. Variation in selected counts across folds shows that this initial single-seed run does not establish a stable universal component count or prove PSO found the global optimum. The run and candidate-level progress are appended to `results/pso_pca_svd_runs.txt`; reruns append rather than overwrite.

### Matched comparison against the 95% explained-variance threshold

`scripts/compare_pso_vs_pca_threshold.py` compared PSO selection and a fixed 95% explained-variance rule on the same three stratified outer folds of `sat.trn`. Both used SVD directions fit from training rows only, the same inner split for selecting the component count, the same scikit-learn kNN (`k=5`), and the same held-out outer rows for scoring. `sat.tst` was not accessed. The run used seed 42, PSO epochs 5, and population 8.

| Outer fold | Threshold components | Threshold macro-F1 | PSO components | PSO macro-F1 | Paired difference |
|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 0.894679 | 25 | 0.897492 | +0.002814 |
| 2 | 6 | 0.862318 | 23 | 0.875952 | +0.013634 |
| 3 | 6 | 0.860804 | 12 | 0.862925 | +0.002121 |
| Mean | 6 | 0.872600 | — | 0.878790 | +0.006190 |

PSO was higher on all three folds, by a mean of **0.006190 macro-F1** (about 0.62 percentage points), while using more components in each fold. This is an initial positive result for PSO when the objective is validation macro-F1, but three folds from one seed are not enough to establish a general winner. The 95% threshold is the more compact choice in this run. The combined comparison completed in **2.01 seconds** on the measured machine. Full per-candidate output appends to `results/pso_vs_threshold_runs.txt`.
