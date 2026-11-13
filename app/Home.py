from common import ROOT, read_result, setup_page

import streamlit as st

STUDIES = """
| Study | Dataset | Question |
|---|---|---|
| 1. Graph-based selection and PCA from scratch | UCI Statlog Landsat: 6,435 samples, 36 features, 6 classes | Can redundancy removal with our own heap, graph and PCA code keep accuracy while cutting features? |
| 2. Metaheuristic comparison | Landsat training file | How do ten nature-inspired optimisers compare at selecting features and tuning the pipeline? |
| 3. Malware family classification | pcap.csv: 737,651 packets, 36 usable features, 6 families | Which scaling, reduction and classifier works best, can metaheuristics find it cheaply, and does our selection compete with PCA? |
"""

PAGES = """
- **Feature Selection (Landsat)**: move the correlation threshold and watch mutual information, the correlation graph and heap + BFS choose the features, then score them with our kNN.
- **Metaheuristic Search**: run any of the ten algorithms on the pcap configuration spaces and compare its result with the true best.
"""

setup_page("Graph-Based Feature Selection and Dimensionality Reduction",
           "Data Structures and Algorithms project, M.Tech AIML, Bennett University. Team: Kshitiz, Yashovardhan Kushva.")

st.markdown("Feature selection keeps a subset of the original features; dimensionality reduction (PCA, KernelPCA, "
            "LDA) builds new ones. This project implements the selection side with its own data structures and "
            "compares both on two datasets, with metaheuristics choosing the settings.")

st.subheader("How the algorithms link")
st.image(str(ROOT / "app" / "assets" / "workflow.svg"), width=520)
st.markdown("Every study runs this pipeline. The threshold *t* decides which features survive heap + BFS, *k* "
            "decides how many components PCA keeps, and the metaheuristics choose these settings using the "
            "cross-validated macro-F1 as their fitness.")

st.subheader("The three studies")
st.markdown(STUDIES)

st.subheader("Key results")
landsat = read_result("fs_experiment.csv")
pcap_test = read_result("pcap_fs_final_v1/comparison_with_grid_test.csv")
columns = st.columns(3)
if landsat is not None:
    raw = landsat.iloc[0]
    heap = landsat[landsat.setting == "heap+BFS t=0.9"].iloc[0]
    columns[0].metric("Landsat, 36 features", f"{raw.accuracy:.4f}")
    columns[1].metric(f"Landsat, {heap.features} features", f"{heap.accuracy:.4f}",
                      f"{heap.accuracy - raw.accuracy:+.4f}")
if pcap_test is not None:
    pca = pcap_test[pcap_test.method == "PCA"].iloc[0]
    ours = pcap_test[pcap_test.method.isin(["MI_top", "HeapBFS", "Mask"])].iloc[0]
    columns[2].metric(f"pcap, {ours.n_features} selected features", f"{ours.test_f1_macro:.4f}",
                      f"{ours.test_f1_macro - pca.test_f1_macro:+.4f} vs PCA")
st.caption("Landsat: accuracy of our kNN (k = 5) on the official test split, with all features and after heap + "
           "BFS at t = 0.9. pcap: macro-F1 on the held-out test packets of our best selection setting chosen by "
           "cross-validation, compared with the best PCA configuration.")

st.subheader("Pages")
st.markdown(PAGES)
