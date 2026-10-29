# Graph-Based Feature Selection & Dimensionality Reduction

Data Structures and Algorithms project, M.Tech AIML, Bennett University.

**Team:** Kshitiz, Yashovardhan Kushva

The project studies feature selection and dimensionality reduction for classification in three parts:

| Study | Dataset | Question |
|---|---|---|
| [1. Graph-based selection and PCA from scratch](#1-graph-based-selection-and-pca-from-scratch) | UCI Statlog Landsat (6,435 samples, 36 features, 6 classes) | Can redundancy removal with our own heap, graph and PCA code keep accuracy while cutting features? |
| [2. Metaheuristic comparison](#2-metaheuristic-comparison-landsat) | Landsat (training file only) | How do ten nature-inspired optimisers compare at selecting features and tuning the pipeline? |
| [3. Malware family classification](#3-malware-family-classification-pcap) | Network traffic, `pcap.csv` (737,651 packets, 39 features, 8 families) | Which scaling, reduction and classifier combination works best, and can metaheuristics find it without trying every combination? |

All core data structures (heap, merge sort, mutual information, graph search, Union-Find, KD-tree, kNN, PCA) are written from scratch and validated against numpy, scipy and scikit-learn.

## Repository layout

```
src/
  heap.py              indexed max-heap (push, pop, update, O(n) build)
  sorting.py           merge sort
  mi.py                binned mutual information, relevance scores
  graph.py             adjacency list, BFS, DFS, connected components
  unionfind.py         Union-Find (path halving, union by rank)
  selection.py         heap + BFS selection, per-component variant, naive baseline
  kdtree.py            KD-tree with k-nearest-neighbour query
  knn.py               kNN classifier on the KD-tree
  pca.py               PCA from scratch (covariance, power iteration, deflation)
  dataset_loader.py    configurable CSV loader with validation
  validation_pipeline.py  leakage-safe train/validation splits
  meta_fs.py           metaheuristic wrapper feature selection (10 Mealpy algorithms)
  meta_tune.py         metaheuristic tuning of the threshold t and component count k
  pcap_data.py         pcap cleaning, hash-grouped splits, saved row indices
  pcap_grid.py         exhaustive scaler x reduction x k x classifier grid
  pcap_mh_search.py    metaheuristic search over the grid
  pcap_final_test.py   refit and single evaluation on the held-out test set
tests/                 unit tests (106 in total)
scripts/               experiment runners for studies 1 and 2, pcap pilot and timing
results/               all experiment outputs (csv, txt, logs)
report/                written reports
data/                  Landsat files, pcap splits; pcap.csv itself is not stored
```

## Setup

```
pip install -r requirements.txt
python -m pytest tests/ -q
```

Run every command from the repository root. The experiments were run in a conda environment with Python 3.11, NumPy 1.26.0, scikit-learn 1.9.1, XGBoost 3.2.0 and Mealpy 3.0.3. Mealpy requires NumPy ≤ 1.26.0, so do not upgrade NumPy in this environment.

---

## 1. Graph-based selection and PCA from scratch

### Method

1. Rank each feature by mutual information with the class label.
2. Build a redundancy graph: features are nodes, with an edge when |correlation| is above a threshold *t*.
3. Put features in an indexed max-heap by relevance. Pop the best, keep it, and drop its neighbours found by BFS.
4. Reduce the kept features with PCA (implemented from scratch).
5. Classify with a KD-tree based kNN (implemented from scratch).

### Validation of the from-scratch code

| Component | Checked against |
|---|---|
| Indexed max-heap | `heapq` |
| Merge sort | `sorted` |
| Mutual information | scikit-learn |
| Graph, BFS/DFS, Union-Find | scipy connected components |
| Heap + BFS selection | naive selection, identical at every threshold |
| KD-tree and kNN | scikit-learn kNN, identical predictions |
| PCA | numpy and scikit-learn |

### Results on the official Landsat test split

These were computed on `sat.tst`, which earlier analyses had already used, so they are reported as project results rather than an untouched test estimate. Our kNN (k = 5) on raw features matches scikit-learn exactly (0.9045 accuracy).

| Setting | Features | Accuracy | Macro-F1 | Query time |
|---|---|---|---|---|
| Raw | 36 | 0.9045 | 0.8916 | 12.05 s |
| Heap + BFS, t = 0.95 | 25 | 0.8950 | 0.8800 | 9.28 s |
| Heap + BFS, t = 0.9 | 11 | 0.8730 | 0.8531 | 4.65 s |
| Heap + BFS, t = 0.8 | 5 | 0.8520 | 0.8291 | 1.30 s |
| Heap + BFS, t = 0.7 | 2 | 0.7845 | 0.7440 | 0.21 s |

| Setting (separate run, slower machine) | Dims | Accuracy | Macro-F1 | Query time |
|---|---|---|---|---|
| Raw | 36 | 0.9045 | 0.8916 | 33.14 s |
| PCA, 95% variance | 6 | 0.8950 | 0.8768 | 5.59 s |
| t = 0.95, then PCA | 6 | 0.8935 | 0.8765 | 5.13 s |
| t = 0.9, then PCA | 5 | 0.8640 | 0.8417 | 4.21 s |

Feature selection trades some accuracy for much faster queries; lower thresholds remove more features and lose more accuracy. PCA alone keeps 6 of 36 dimensions at 0.8950 accuracy, and selecting features before PCA gave no further gain.

```
python scripts/run_selection.py
python scripts/run_fs_experiment.py
python scripts/run_complexity.py
python scripts/run_pca_experiment.py
```

### PSO studies and leakage audit

The early PSO runs (V4) scored candidate masks on the same validation fold later used for reporting, so their scores are optimistic and kept only as history. The corrected studies are:

- **Nested CV** (`scripts/run_pso_nested_cv_v5.py`): every outer fold is excluded from PSO search, scaling and fitting. PSO was lower than all features on all five folds (mean macro-F1 0.8620 vs 0.8843).
- **Single frozen evaluation on `sat.tst`** (`scripts/run_pso_final_test_v1.py`, guarded against reruns): PSO 0.8945 accuracy / 0.8805 macro-F1 with 15 of 36 features; all features 0.9045 / 0.8916.
- **PSO choice of the PCA component count** (`scripts/run_pso_pca_svd.py`, `scripts/compare_pso_vs_pca_threshold.py`): on 3 matched folds PSO-chosen counts scored 0.8788 macro-F1 against 0.8726 for the 95% variance rule. One three-fold run is too small to name a general winner.

```
python scripts/run_pso_nested_cv_v5.py --folds 5 --seed 42 --epochs 5 --population 8
python scripts/run_pso_final_test_v1.py --seed 42 --epochs 10 --population 10
python scripts/run_pso_pca_svd.py --folds 3 --epochs 5 --population 8
```

The full audit and limitations are in `report/DSA_Final_Experimental_Report_v5.md`.

---

## 2. Metaheuristic comparison (Landsat)

Ten Mealpy algorithms (GA, DE, PSO, GWO, WOA, HHO, ABC, MFO, SSA = Sparrow Search, SMA) are compared under the same rules:

- **Wrapper feature selection** (`src/meta_fs.py`): search a binary feature mask.
- **Pipeline tuning** (`src/meta_tune.py`): choose the heap + BFS threshold *t*, the PCA component count *k*, or both, using the from-scratch MI, graph, heap and PCA code.
- **Fitness** (minimised): 0.9 × (1 − macro-F1) + 0.1 × (features or components used / 36).
- **Equal budgets**: every run gets the same number of fitness evaluations, since some algorithms evaluate more often per iteration.
- **Exact optimum**: *t* only changes the selected subset at observed correlation values, so an exhaustive grid gives the true optimum for each tuning task.
- **Protocol**: nested 5-fold CV on the training file, 3 seeds, budget 200; `sat.tst` is never read.

| Task | Main result (outer-fold macro-F1; all features = 0.884) |
|---|---|
| PCA component count *k* | 0.877 with about 7 components; a 36-point grid finds it as well as any metaheuristic |
| Threshold *t* | SSA found the exact optimum in 100% of runs, PSO in 93%, GA in 7% |
| Joint (*t*, *k*) | Hardest search: best hit rate 27% (PSO), yet PSO's score was within 0.001 of the optimum's |
| Feature masks | PSO best, 0.868 with about 12 of 36 features |

```
python scripts/run_mh_comparison.py --smoke
python scripts/run_mh_comparison.py
```

Outputs: `results/mh_comparison_v1.csv`, `results/mh_convergence_v1.csv`.

---

## 3. Malware family classification (pcap)

### Data and protocol

Each row of `pcap.csv` is one network packet; packets from the same malware sample share a `Hash`. The dataset is not stored in this repository; place it at `data/pcap.csv`.

- `Hash`, `Arch` and the label are not features. 563 infinite `Rate` values become missing values, imputed with the training median. 3 constant columns are dropped, leaving 36 features.
- `Agent` (1 hash) and `Tsunami` (3 hashes) cannot appear in both training and test under grouping, so their 812 rows (0.1%) are excluded. Six families remain.
- **Test set**: 145,355 packets (19.7%) from 1,101 hashes, held out by a stratified, hash-grouped split and used once.
- **Grid sample**: at most 5 packets per development hash (22,467 rows), with stratified, hash-grouped 4-fold CV. Four folds, because Gafgyt has only 4 development hashes.
- All row indices are saved in `data/pcap_splits.npz`, so every experiment uses identical rows.
- Mirai is 70% of the test packets, so results are reported as **macro-F1** and balanced accuracy alongside accuracy.

### Pilot experiment

Random Forest and XGBoost on 5 original columns (3rd to 7th) cost only 0.01–0.02 test macro-F1 against all 36 features; 5 PCA components performed about the same as the 5 columns.

### Exhaustive grid

3 scalers (Standard, MinMax, Normalizer) × 3 reductions (PCA and KernelPCA with k = 1–36, LDA with k = 1–5) × 4 classifiers (Random Forest, XGBoost, Decision Tree, SVM) = **924 configurations, 3,696 fits**, completed in 63 minutes with no failures.

| Classifier | Best by CV accuracy | Best by CV macro-F1 |
|---|---|---|
| C1 · Random Forest | Normalizer + PCA, k = 35 | MinMax + KernelPCA, k = 9 (0.249) |
| C2 · XGBoost | MinMax + PCA, k = 36 (0.629) | MinMax + PCA, k = 12 (0.252) |
| C3 · Decision Tree | Normalizer + KernelPCA, k = 1 | **Normalizer + PCA, k = 14 (0.269)** |
| C4 · SVM | Standard + PCA, k = 24 | Standard + PCA, k = 24 (0.222) |

### Metaheuristic search over the grid

The same ten algorithms search the 924 configurations, with each candidate's CV score looked up from the grid, and random search as the baseline (20 seeds per setting). With a budget of 400 evaluations, GWO found the exact best configuration in 80% of runs while trying 219 of 924 configurations; random search at that cost would succeed about 24% of the time. SSA (65%) and PSO (60%) also beat random search clearly; GA, DE, WOA, HHO and SMA converged early and rarely reached the top.

### Held-out test results

Each classifier's best configuration (by CV accuracy and by CV macro-F1) was refitted on all 591,484 development rows (SVM on a 60,000-row subsample; KernelPCA fitted on 8,000 rows) and evaluated once.

| Classifier | Configuration | Test macro-F1 | Balanced accuracy | Accuracy |
|---|---|---|---|---|
| Decision Tree | Normalizer + PCA, k = 14 | **0.299** | **0.332** | 0.648 |
| Random Forest | Normalizer + PCA, k = 35 | 0.284 | 0.322 | 0.703 |
| Random Forest | MinMax + KernelPCA, k = 9 | 0.279 | 0.308 | 0.695 |
| Decision Tree | Normalizer + KernelPCA, k = 1 | 0.270 | 0.321 | 0.683 |
| XGBoost | MinMax + PCA, k = 36 | 0.257 | 0.297 | **0.721** |
| XGBoost | MinMax + PCA, k = 12 | 0.254 | 0.295 | 0.720 |
| SVM | Standard + PCA, k = 24 | 0.252 | 0.318 | 0.720 |

Predicting Mirai for every packet already scores 0.699 accuracy.

### Findings

- CV selection held up on unseen malware samples: the CV accuracy leader has the highest test accuracy, and the CV macro-F1 leader the highest test macro-F1.
- Ranking by accuracy picks models that mostly predict Benign and Mirai; ranking by macro-F1 picks a model that recognises more families.
- A decision tree on 14 PCA components matches Random Forest on all 36 raw features (0.299 vs 0.298 test macro-F1).
- The top of the grid is a plateau: the 10 best configurations lie within 0.012 CV macro-F1.
- Gafgyt and Generic are not recognised by any configuration; scaling, reduction or classifier choice does not change this.

### Reproduce

```
python src/pcap_data.py --data data/pcap.csv --out data/pcap_splits.npz
python src/pcap_grid.py --smoke
python src/pcap_grid.py
python src/pcap_mh_search.py
python src/pcap_final_test.py
python scripts/run_pcap_small.py --pca 5
```

The grid run is resumable: if interrupted, run the same command again. The final-test script refuses to run a second time so the test set is used once.

Outputs: `results/pcap_small_v2/`, `results/pcap_grid_v1/`, `results/pcap_mh_v1/`, `results/pcap_final_v1/` (including `best_model.joblib`, the configuration with the highest CV accuracy).

---

`data/framework_wine_smoke.csv` is a small CSV fixture used to test the dataset loader on a second schema. It is derived from the UCI Wine dataset by Aeberhard and Forina (1992), licensed CC BY 4.0: [UCI dataset record](https://doi.org/10.24432/C5PC7J).
