from common import setup_page

import altair as alt
import streamlit as st

from results_explorer import best_by_k, filter_grid, load_grid, top_configs

METRICS = {"Macro-F1": "f1_macro_mean", "Balanced accuracy": "balanced_accuracy_mean", "Accuracy": "accuracy_mean"}
HEADERS = {"scaler": "Scaler", "reduction": "Reduction", "k": "k", "classifier": "Classifier",
           "f1_macro_mean": "Macro-F1", "f1_macro_sd": "SD", "balanced_accuracy_mean": "Balanced acc.",
           "accuracy_mean": "Accuracy"}

setup_page("pcap Grid Results",
           "Every combination of 3 scalers, 3 reduction methods at every component count k and 4 classifiers: "
           "924 configurations, each scored by 4-fold cross-validation grouped by Hash. Saved scores only; the "
           "test set is not used here.")

grid = load_grid()

with st.sidebar:
    st.header("Filters")
    scalers = st.multiselect("Scalers", sorted(grid.scaler.unique()))
    reductions = st.multiselect("Reduction methods", sorted(grid.reduction.unique()))
    classifiers = st.multiselect("Classifiers", sorted(grid.classifier.unique()))
    metric_name = st.radio("Rank by", list(METRICS))
    n = st.slider("Configurations to list", 5, 50, 10, 5)
    st.caption("An empty filter keeps every option.")

metric = METRICS[metric_name]
chosen = filter_grid(grid, scalers, reductions, classifiers)
if chosen.empty:
    st.warning("No configuration matches these filters.")
    st.stop()

top = top_configs(chosen, n, metric)
best = top.iloc[0]
row = st.columns([1, 1, 2])
row[0].metric("Configurations shown", f"{len(chosen)} of {len(grid)}")
row[1].metric(f"Best CV {metric_name}", f"{best[metric]:.4f}")
row[2].metric("Best configuration", f"{best.scaler} + {best.reduction} ({best.k}) + {best.classifier}")

left, right = st.columns([3, 2])
with left:
    st.subheader(f"Top {len(top)} by CV {metric_name}")
    shown = top.rename(columns=HEADERS)
    st.dataframe(shown.style.format({c: "{:.4f}" for c in list(HEADERS.values())[4:]}),
                 hide_index=True, width="stretch")
with right:
    st.subheader("Best score at each k")
    curve = best_by_k(chosen, metric)
    chart = alt.Chart(curve).mark_line(point=True).encode(
        x=alt.X("k:Q", title="Components k"),
        y=alt.Y(f"{metric}:Q", title=f"Best CV {metric_name}", scale=alt.Scale(zero=False)),
        color=alt.Color("reduction:N", title=None),
        tooltip=["reduction", "k", alt.Tooltip(f"{metric}:Q", format=".4f")])
    st.altair_chart(chart.properties(height=360), width="stretch")
    st.caption("Each point is the best configuration with that reduction and k, over the scalers and "
               "classifiers kept by the filters. LDA allows at most 5 components for 6 families.")

st.caption("The top of the grid is a plateau: the 10 best configurations lie within 0.012 CV macro-F1, so small "
           "differences between them should not be read as real.")
