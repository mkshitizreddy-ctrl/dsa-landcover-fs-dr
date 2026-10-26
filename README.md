# Graph-Based Feature Selection & PCA for Land-Cover Classification

Data Structures and Algorithms project, M.Tech AIML, Bennett University.

**Team:** Kshitiz, Yashovardhan Kushva

**Topics covered:** Feature Selection and Dimensionality Reduction, on the UCI Statlog (Landsat Satellite) land-cover dataset (6435 samples, 36 features, 6 classes).

## Idea

1. Rank each feature by mutual information with the class label.
2. Build a redundancy graph: features are nodes, with an edge when |correlation| is above a threshold t.
3. Put features in an indexed max-heap by relevance. Pop the best, keep it, and drop its neighbours found by BFS.
4. Reduce the kept features with PCA/SVD (implemented from scratch).
5. Classify with a KD-tree based kNN (implemented from scratch).

All data structures are written from scratch. Libraries are used only for loading data, metrics, and validating our code against sklearn and scipy.

## Structure

```
src/
  heap.py         indexed max-heap (push, pop, update, O(n) build)
  sorting.py      merge sort
  mi.py           binned mutual information, relevance scores
  graph.py        adjacency list, BFS, DFS, components
  unionfind.py    Union-Find (path halving, union by rank)
  selection.py    heap + BFS selection, per-component variant, naive baseline
  kdtree.py       KD-tree with k-nearest-neighbour query
  knn.py          kNN classifier on the KD-tree
  pca.py          PCA from scratch (covariance, power iteration, deflation)
tests/            one test file per module
scripts/          experiment scripts
results/          experiment outputs (csv)
```

`data/framework_wine_smoke.csv` is a small, headered CSV fixture used to
exercise the configurable dataset pipeline on a second schema. It is derived
from the UCI Wine dataset by Aeberhard and Forina (1992), licensed CC BY 4.0:
[UCI dataset record](https://doi.org/10.24432/C5PC7J).

## Run

```
pip install -r requirements.txt
python tests/test_heap.py
python tests/test_mi_sorting.py
python tests/test_graph.py
python tests/test_selection.py
python tests/test_knn.py
python tests/test_pca.py
python -m pytest -q
python scripts/run_selection.py
python scripts/run_fs_experiment.py
python scripts/run_complexity.py
python scripts/run_pca_experiment.py
```

The original experiment scripts use the local `data/sat.trn` and `data/sat.tst` files; they attempt to download missing files from UCI. The newer validation and PSO runners use the local training file and do not need internet access. Run commands from the repository root in the `landcover_opt` environment. If pytest cannot create its default temporary folder on your machine, pass a fresh directory under `work/`, for example `python -m pytest -q --basetemp=work/pytest-run`.

## Progress

| Part | Status |
|---|---|
| Indexed max-heap | Done, tested against heapq |
| Merge sort | Done, tested against sorted |
| Mutual information scoring | Done, tested against sklearn |
| Graph, BFS/DFS, Union-Find | Done, tested against scipy |
| Heap + BFS selection, Union-Find variant, naive baseline | Done, matches naive at all thresholds |
| KD-tree and kNN | Done, matches sklearn kNN |
| Feature selection experiment | Done |
| PCA from scratch | Done, tested against numpy and sklearn |
| Runtime comparison against naive filter | Done |
| PCA rows in the pipeline comparison | Done |
| Experimental report | Updated in `report/DSA_Final_Experimental_Report_v5.md` |
| Final presentation | Not included in this repository snapshot |

## Reproducible PSO evaluation (2026-10-26)

The V4 PSO summary files are historical validation-only runs. Each run selected features by repeatedly scoring the same 20% validation fold later used for its reported score, so those scores are exploratory and optimistic. V4 was run with seeds 7, 21, 42 and 84; the V4 script currently defaults to seed 42. Do not interpret the four scores as independent test estimates.

Use `scripts/run_pso_nested_cv_v5.py` for an outer-fold estimate. Each outer validation fold is kept out of PSO search, feature scaling, and model fitting. The default is five stratified folds, with a 5-epoch, 8-member PSO budget in each fold. The full-feature custom kNN baseline uses the same outer folds. Results and feature frequencies go to new `*_v5.csv` files.

After reviewing and freezing the method, `scripts/run_pso_final_test_v1.py` runs the 10-epoch, 10-member V4 search on a training-only split, fixes the mask, fits on all `sat.trn`, and evaluates the selected and full-feature models in one final stage on `sat.tst`. A header-only output guard prevents accidental reruns after the final-test stage starts. Its output is `results/pso_official_test_v1.csv`. The project has prior historical results on this official split, so it is held out from this new PSO selection but is not a pristine test set for the project as a whole.

```powershell
python scripts\summarize_pso_v4_stability_v5.py
python scripts\run_pso_nested_cv_v5.py --folds 5 --seed 42 --epochs 5 --population 8
python scripts\run_pso_final_test_v1.py --seed 42 --epochs 10 --population 10
```

The older threshold and PCA results in `results/fs_experiment.csv`, `results/pca_experiment.csv`, and `report/DSA_Project_Report.docx` were calculated using the official test set. They are retained as historical project results and must not be used to tune or select the new PSO procedure. See `report/DSA_Final_Experimental_Report_v5.md` for the audit, new results, and limitations.

## Comparative study and conclusion

The comparative study uses training-only stratified validation splits to compare feature-ranking and graph-based selectors at 6, 12, and 18 features. On the saved seed-42 split, all-features custom kNN scored 0.8711 macro-F1; the best reported 12-feature comparison, RFE, scored 0.8546. This is a single split, not an independent final test estimate. The paired PSO nested-CV study is lower than all-features on all five outer folds: mean macro-F1 0.8620 versus 0.8843 (mean paired difference −0.0223). In the one frozen official-split evaluation, PSO scored 0.8945 accuracy and 0.8805 macro-F1, while all-features scored 0.9045 and 0.8916. PSO reduced the input from 36 to 15 features, but did not improve predictive performance. Earlier project analyses had already used the official test split, so it is not a pristine project-wide holdout.

The final experimental report is `report/DSA_Final_Experimental_Report_v5.md`. Reproduce the newer studies with `scripts/run_comparative_v3.py`, `scripts/run_pso_nested_cv_v5.py`, and, only after the method is frozen, `scripts/run_pso_final_test_v1.py`. The final-test runner intentionally guards against a second run to avoid repeated use of `sat.tst`.

## PSO selection of a fixed PCA component count

`scripts/run_pso_pca_svd.py` is a separate experiment: Mealpy PSO searches integer component counts from 1 to 36, while PCA directions are computed explicitly with SVD. Preprocessing is fit only on each training partition. Nested stratified CV estimates the search procedure, and a final training-only search records one proposed component count. The script uses scikit-learn kNN for practical runtime; this classifier choice is logged. The official `sat.tst` file is never loaded. All candidate progress, fold metrics, chosen counts, and elapsed times append to `results/pso_pca_svd_runs.txt`.

Start with a quick timing run, then increase the search budget if needed:

```powershell
python scripts\run_pso_pca_svd.py --folds 2 --epochs 1 --population 5
python scripts\run_pso_pca_svd.py --folds 3 --epochs 5 --population 8
```

The PSO-selected validation score is used for search and is not an independent test estimate. Use the nested outer-fold scores when describing performance. Each additional run is appended, not overwritten.

The first larger run (seed 42; 3 outer folds; 5 epochs; population 8) completed in 1.63 seconds on the current machine. PSO selected 25, 23, and 12 components in the outer folds; their held-out macro-F1 mean was 0.8788 (sample SD 0.0175). The final training-only search proposed 18 components, with inner validation macro-F1 0.8864. These are an initial run at one seed and budget, not proof of a globally best component count; the differing fold selections show that the chosen count depends on the training split. Full candidate-level details are in the appended text log.

### Matched comparison with the 95% variance threshold

`scripts/compare_pso_vs_pca_threshold.py` compares both approaches on identical training-only outer folds, with the same SVD preprocessing and kNN classifier. In the first run (seed 42; 3 folds; PSO 5 epochs × 8 particles), the threshold method selected 6 components in every fold; PSO selected 25, 23, and 12. Mean outer macro-F1 was 0.8726 for the threshold and 0.8788 for PSO, a paired mean difference of +0.0062 (about 0.62 percentage points); PSO scored higher on all 3 folds. This is encouraging for PSO's classification score, but one three-fold run is too limited to establish a general winner. The combined run took 2.01 seconds here. Details append to `results/pso_vs_threshold_runs.txt`.

## Results so far

The historical metrics below were calculated on `sat.tst`, which had already been used in earlier project analyses. Treat them as retained project results, not as an untouched independent test estimate.

Test accuracy on the official split, using our kNN with k = 5. Raw-feature accuracy matches sklearn kNN (0.9045).

| Setting | Features | Accuracy | Macro-F1 | Query time |
|---|---|---|---|---|
| Raw | 36 | 0.9045 | 0.8916 | 12.05 s |
| Heap + BFS, t = 0.95 | 25 | 0.8950 | 0.8800 | 9.28 s |
| Heap + BFS, t = 0.9 | 11 | 0.8730 | 0.8531 | 4.65 s |
| Heap + BFS, t = 0.8 | 5 | 0.8520 | 0.8291 | 1.30 s |
| Heap + BFS, t = 0.7 | 2 | 0.7845 | 0.7440 | 0.21 s |

PCA runs (second run, slower machine, raw took 33.14 s, so compare times only within this table):

| Setting | Dims | Accuracy | Macro-F1 | Query time |
|---|---|---|---|---|
| Raw | 36 | 0.9045 | 0.8916 | 33.14 s |
| PCA, 95% variance | 6 | 0.8950 | 0.8768 | 5.59 s |
| t = 0.95, then PCA | 6 | 0.8935 | 0.8765 | 5.13 s |
| t = 0.9, then PCA | 5 | 0.8640 | 0.8417 | 4.21 s |

Feature selection alone trades some accuracy for much faster queries. Lower thresholds remove more features and lose more accuracy. PCA alone keeps 6 of 36 dims at 0.8950 accuracy, and selecting features before PCA gave no further gain.
