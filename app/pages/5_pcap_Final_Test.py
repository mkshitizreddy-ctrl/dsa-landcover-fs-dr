from common import setup_page

import altair as alt
import streamlit as st

from results_explorer import (BEST_CONFUSION_PATH, confusion_files, final_test_comparison, majority_accuracy,
                              read_confusion, row_percentages, sample_level_summary, weighting_table)

HEADERS = {"family": "Family", "scaler": "Scaler", "setting": "Setting", "classifier": "Classifier",
           "n_features": "Inputs", "chosen_as": "Chosen as", "cv_f1_macro": "CV macro-F1",
           "test_f1_macro": "Test macro-F1", "test_balanced_accuracy": "Test balanced acc.",
           "test_accuracy": "Test accuracy"}
SCORES = ["CV macro-F1", "Test macro-F1", "Test balanced acc.", "Test accuracy"]
COLORS = alt.Scale(domain=["selection", "extraction"], range=["#2f6db5", "#d9822b"])

setup_page("pcap Final Test",
           "Configurations chosen by cross-validation, refitted on all 591,484 development packets and scored "
           "once on the 145,355 held-out test packets. Selection keeps original features (our MI ranking, heap + "
           "BFS, or a metaheuristic mask); extraction builds new ones (PCA, KernelPCA).")

results = final_test_comparison()
best_confusion = read_confusion(BEST_CONFUSION_PATH)
best = {family: part.iloc[0] for family, part in results.groupby("family")}

row = st.columns(3)
row[0].metric("Best test macro-F1, selection", f"{best['selection'].test_f1_macro:.4f}",
              f"{best['selection'].test_f1_macro - best['extraction'].test_f1_macro:+.4f} vs extraction")
row[1].metric("Best test macro-F1, extraction", f"{best['extraction'].test_f1_macro:.4f}")
row[2].metric("Accuracy of always predicting Mirai", f"{majority_accuracy(best_confusion):.4f}")

st.subheader("Held-out test results")
shown = results.rename(columns=HEADERS)
st.dataframe(shown.style.format({c: "{:.4f}" for c in SCORES}), hide_index=True, width="stretch")
bars = results.assign(name=results.scaler + " + " + results.setting + " + " + results.classifier)
chart = alt.Chart(bars).mark_bar().encode(
    x=alt.X("test_f1_macro:Q", title="Test macro-F1"),
    y=alt.Y("name:N", sort="-x", title=None, axis=alt.Axis(labelLimit=400)),
    color=alt.Color("family:N", scale=COLORS, title=None),
    tooltip=["name", "family", "n_features", alt.Tooltip("test_f1_macro:Q", format=".4f"),
             alt.Tooltip("test_accuracy:Q", format=".4f")])
st.altair_chart(chart.properties(height=420), width="stretch")

st.subheader("Which families are recognised")
files = confusion_files()
names = list(files)
choice = st.selectbox("Configuration", names, index=names.index("Normalizer PCA k14 DT"))
confusion = read_confusion(files[choice])
cells = (row_percentages(confusion).rename_axis("true").reset_index()
         .melt(id_vars="true", var_name="predicted", value_name="percent"))
cells["packets"] = confusion.rename_axis("true").reset_index().melt(id_vars="true").value.to_numpy()
order = list(confusion.index)
base = alt.Chart(cells).encode(x=alt.X("predicted:N", sort=order, title="Predicted family"),
                               y=alt.Y("true:N", sort=order, title="True family"))
heat = base.mark_rect().encode(color=alt.Color("percent:Q", scale=alt.Scale(scheme="blues"), title="% of row"),
                               tooltip=["true", "predicted", alt.Tooltip("packets:Q", format=","),
                                        alt.Tooltip("percent:Q", format=".1f")])
text = base.mark_text(fontSize=12).encode(
    text=alt.Text("percent:Q", format=".0f"),
    color=alt.condition(alt.datum.percent > 50, alt.value("white"), alt.value("black")))
st.altair_chart((heat + text).properties(height=360), width="stretch")
st.caption("Each row shows where the test packets of one family went, as a percentage of that family. "
           "Gafgyt and Generic are almost never recognised: no configuration gets more than 2.5% of their test "
           "packets right, and most of them are predicted as Mirai, the largest family.")

st.subheader("Does class weighting help the rare families?")
weights = weighting_table().rename(columns={"family": "Family", "configuration": "Configuration", "none": "No weights",
                                            "sqrt": "Square-root weights", "balanced": "Balanced weights"})
st.dataframe(weights.style.format({c: "{:.4f}" for c in ["No weights", "Square-root weights", "Balanced weights"]}),
             hide_index=True, width="stretch")
st.caption("Cross-validation macro-F1 of the best configurations with and without class weights. The changes are "
           "small and mixed, so weighting does not fix the rare families.")

st.subheader("Whole malware samples instead of packets")
st.markdown("All packets of one Hash come from one malware sample, so they can be summarised into one row: the mean, "
            "standard deviation, minimum and maximum of each feature and the packet count. The best classifier by "
            "cross-validation on these samples (RF) was scored once on the 1,101 test samples, using the same "
            "development and test hashes as above.")
comparison, recall = sample_level_summary()
packet, sample = comparison.set_index("level").loc["packet"], comparison.set_index("level").loc["sample"]
row = st.columns(3)
row[0].metric("Test macro-F1, samples", f"{sample.test_f1_macro:.4f}",
              f"{sample.test_f1_macro - packet.test_f1_macro:+.4f} vs packets")
row[1].metric("Test accuracy, samples", f"{sample.test_accuracy:.4f}",
              f"{sample.test_accuracy - packet.test_accuracy:+.4f} vs packets")
row[2].metric("Test samples", f"{int(recall.test_samples.sum()):,}")
left, right = st.columns([2, 3])
with left:
    st.dataframe(comparison.rename(columns={"level": "Unit", "configuration": "Configuration",
                                            "cv_f1_macro": "CV macro-F1", "test_f1_macro": "Test macro-F1",
                                            "test_balanced_accuracy": "Test balanced acc.",
                                            "test_accuracy": "Test accuracy"})
                 .style.format({c: "{:.4f}" for c in SCORES}), hide_index=True, width="stretch")
    st.dataframe(recall.rename(columns={"family": "Family", "test_samples": "Test samples",
                                        "sample_recall": "Recall, samples", "packet_recall": "Recall, packets"})
                 .style.format({"Recall, samples": "{:.2f}", "Recall, packets": "{:.2f}"}),
                 hide_index=True, width="stretch")
with right:
    bars = recall.melt(id_vars=["family", "test_samples"], value_vars=["sample_recall", "packet_recall"],
                       var_name="unit", value_name="recall")
    bars["unit"] = bars.unit.map({"sample_recall": "samples", "packet_recall": "packets"})
    chart = alt.Chart(bars).mark_bar().encode(
        x=alt.X("family:N", sort=list(recall.family), title=None),
        xOffset=alt.XOffset("unit:N", sort=["packets", "samples"]),
        y=alt.Y("recall:Q", title="Test recall", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("unit:N", scale=alt.Scale(domain=["packets", "samples"], range=["#d9822b", "#2f6db5"]),
                        title=None),
        tooltip=["family", "unit", alt.Tooltip("recall:Q", format=".2f"), "test_samples"])
    st.altair_chart(chart.properties(height=320), width="stretch")
st.caption("Whole samples give the best test macro-F1 and accuracy in the project, but Gafgyt (4 development and "
           "1 test sample) and Generic (22 and 7) are still never recognised: there are too few malware samples of "
           "these families to learn from. Unknown is recognised less often as whole samples (0.02) than as packets "
           "(0.17).")
