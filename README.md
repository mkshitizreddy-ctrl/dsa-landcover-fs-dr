\# Graph and Heap-Based Redundancy-Aware Feature Selection with From-Scratch PCA/SVD



Data Structures and Algorithms project, M.Tech AIML, Bennett University.

Team: Kshitiz, Yashovardhan Kushva.



Topics covered: Feature Selection and Dimensionality Reduction, on the UCI Statlog (Landsat Satellite) land-cover dataset (6435 samples, 36 features, 6 classes).



\## Idea

1\. Rank each feature by mutual information with the class label.

2\. Build a redundancy graph: features are nodes, with an edge when |correlation| is above a threshold t.

3\. Put features in an indexed max-heap by relevance. Pop the best, keep it, and drop its neighbours found by BFS.

4\. Reduce the kept features with PCA/SVD (implemented from scratch).

5\. Classify with a KD-tree based kNN (implemented from scratch).



All data structures are written from scratch. Libraries are used only for loading data, metrics, and validating our code against sklearn/scipy.



\## Structure

```

src/

&#x20; heap.py         indexed max-heap (push, pop, update, O(n) build)

&#x20; sorting.py      merge sort

&#x20; mi.py           binned mutual information, relevance scores

&#x20; graph.py        adjacency list, BFS, DFS, components

&#x20; unionfind.py    Union-Find (path halving, union by rank)

&#x20; selection.py    heap + BFS selection, per-component variant, naive O(d^2) baseline

&#x20; kdtree.py       KD-tree with k-nearest-neighbour query

&#x20; knn.py          kNN classifier on the KD-tree

tests/            one test file per module

scripts/          experiment scripts

results/          experiment outputs (csv)

```



\## Run

```

pip install -r requirements.txt

python tests/test\_heap.py

python tests/test\_mi\_sorting.py

python tests/test\_graph.py

python tests/test\_selection.py

python tests/test\_knn.py

python scripts/run\_selection.py

python scripts/run\_fs\_experiment.py

```

The scripts download the dataset from the UCI repository, so they need internet access.



\## Progress



| Part | Status |

|---|---|

| Indexed max-heap | Done, tested against heapq |

| Merge sort | Done, tested against sorted |

| Mutual information scoring | Done, tested against sklearn |

| Graph, BFS/DFS, Union-Find | Done, tested against scipy |

| Heap + BFS selection, Union-Find variant, naive baseline | Done, matches naive at all thresholds |

| KD-tree and kNN | Done, matches sklearn kNN |

| Feature selection experiment | Done |

| PCA/SVD from scratch | In progress |

| Runtime comparison against naive filter | Planned |

| PCA rows in the pipeline comparison | Planned |

| Report and final presentation | In progress |



\## Results so far

Test accuracy on the official split, our kNN with k = 5. Raw-feature accuracy matches sklearn kNN (0.9045).



| Setting | Features | Accuracy | Macro-F1 | Query time |

|---|---|---|---|---|

| Raw | 36 | 0.9045 | 0.8916 | 12.05 s |

| Heap + BFS, t = 0.95 | 25 | 0.8950 | 0.8800 | 9.28 s |

| Heap + BFS, t = 0.9 | 11 | 0.8730 | 0.8531 | 4.65 s |

| Heap + BFS, t = 0.8 | 5 | 0.8520 | 0.8291 | 1.30 s |

| Heap + BFS, t = 0.7 | 2 | 0.7845 | 0.7440 | 0.21 s |



Feature selection alone trades some accuracy for much faster queries. Lower thresholds remove more features and lose more accuracy.

