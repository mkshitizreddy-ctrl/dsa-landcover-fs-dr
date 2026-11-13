from common import setup_page

import altair as alt
import numpy as np
import streamlit as st

from landsat_explorer import METHODS, evaluate, explore_statistics, score_table, select, threshold_sweep, \
    validation_split

setup_page("Feature Selection (Landsat)",
           "Mutual information ranks the 36 features, the correlation graph links features whose |correlation| "
           "exceeds t, and the selection keeps the most relevant feature of each linked group. Scores use a 20% "
           "validation split of the training file; the official test file is not used here.")


@st.cache_data
def load_data():
    X_train, X_valid, y_train, y_valid = validation_split()
    scores, C = explore_statistics(X_train, y_train)
    return X_train, X_valid, y_train, y_valid, scores, C


@st.cache_data
def score(columns, k, pca_ratio):
    X_train, X_valid, y_train, y_valid, _, _ = load_data()
    return evaluate(X_train, y_train, X_valid, y_valid, list(columns), k, pca_ratio)


with st.sidebar:
    st.header("Settings")
    t = st.slider("Correlation threshold t", 0.50, 1.00, 0.90, 0.01)
    method = METHODS[st.radio("Selection method", list(METHODS))]
    use_pca = st.checkbox("Apply our PCA after selection")
    pca_ratio = st.slider("Variance kept by PCA", 0.80, 0.99, 0.95, 0.01, disabled=not use_pca)
    k = st.slider("Neighbours k (our kNN with a KD-tree)", 1, 15, 5)

with st.spinner("Computing mutual information and correlations..."):
    X_train, X_valid, y_train, y_valid, scores, C = load_data()
result = select(scores, C, t, method)
selected = result["features"]
with st.spinner("Scoring with our kNN..."):
    chosen = score(tuple(selected), k, pca_ratio if use_pca else None)
    baseline = score(tuple(range(X_train.shape[1])), k, None)

row = st.columns(5)
row[0].metric("Features kept", f"{len(selected)} of {X_train.shape[1]}")
row[1].metric("Dimensions used", chosen["dimensions"])
row[2].metric("Validation accuracy", f"{chosen['accuracy']:.4f}", f"{chosen['accuracy'] - baseline['accuracy']:+.4f}")
row[3].metric("Macro-F1", f"{chosen['macro_f1']:.4f}", f"{chosen['macro_f1'] - baseline['macro_f1']:+.4f}")
row[4].metric("kNN query time", f"{chosen['query_seconds']:.2f} s",
              f"{chosen['query_seconds'] - baseline['query_seconds']:+.2f} s", delta_color="inverse")
st.caption(f"Changes are against all 36 features (accuracy {baseline['accuracy']:.4f}). Correlation graph at "
           f"t = {t:.2f}: {result['edges']} edges, {result['components']} connected components.")

left, right = st.columns(2)
with left:
    st.subheader("Mutual information of each feature")
    table = score_table(scores, selected)
    chart = alt.Chart(table).mark_bar().encode(
        x=alt.X("mutual_information:Q", title="Mutual information (nats)"),
        y=alt.Y("feature:N", sort=None, title=None),
        color=alt.Color("status:N", scale=alt.Scale(domain=["kept", "removed"], range=["#2f6db5", "#c9ced6"]),
                        title=None),
        tooltip=["feature", alt.Tooltip("mutual_information:Q", format=".3f"), "status"])
    st.altair_chart(chart.properties(height=620), width="stretch")
with right:
    st.subheader("How t controls redundancy")
    sweep = threshold_sweep(scores, C, np.round(np.arange(0.50, 1.001, 0.01), 2), method)
    line = alt.Chart(sweep).mark_line(point=True).encode(
        x=alt.X("threshold:Q", title="Correlation threshold t"), y=alt.Y("features_kept:Q", title="Features kept"),
        tooltip=["threshold", "features_kept", "edges", "components"])
    marker = alt.Chart(sweep[sweep.threshold == round(t, 2)]).mark_rule(color="#d9822b").encode(x="threshold:Q")
    st.altair_chart((line + marker).properties(height=280), width="stretch")
    st.subheader("Selected features")
    st.dataframe(table[table.status == "kept"][["feature", "mutual_information"]].style.format(
        {"mutual_information": "{:.3f}"}), hide_index=True, width="stretch")
