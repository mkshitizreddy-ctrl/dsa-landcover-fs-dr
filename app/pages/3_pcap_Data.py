from common import setup_page

import altair as alt
import streamlit as st

from results_explorer import BEST_CONFUSION_PATH, class_counts, majority_accuracy, read_confusion, split_summary

setup_page("pcap Data",
           "Each row of pcap.csv is one network packet, and packets from the same malware sample share a Hash. "
           "This page reads the saved splits and test results, so pcap.csv itself is not needed.")

summary = split_summary()
confusion = read_confusion(BEST_CONFUSION_PATH)
counts = class_counts(confusion)
sizes = dict(zip(summary["table"].part, summary["table"].packets))

row = st.columns(4)
row[0].metric("Packets used", f"{sizes['Packets after cleaning']:,}")
row[1].metric("Features", len(summary["features"]))
row[2].metric("Families", len(summary["families"]))
row[3].metric("Test packets", f"{sizes['Test, used once']:,}")

left, right = st.columns(2)
with left:
    st.subheader("How the packets are split")
    st.dataframe(summary["table"].style.format({"packets": "{:,}"}), hide_index=True, width="stretch")
    st.caption("The test split is grouped by Hash, so test packets come from malware samples never seen in "
               "training. Agent (1 hash) and Tsunami (3 hashes) are removed because a grouped split cannot place "
               "them on both sides. Cross-validation uses at most 5 packets per hash and 4 folds, because Gafgyt "
               "has only 4 development hashes.")
with right:
    st.subheader("Test packets per family")
    table = counts.rename_axis("family").reset_index()
    table["share"] = table.packets / table.packets.sum()
    chart = alt.Chart(table).mark_bar(color="#2f6db5").encode(
        x=alt.X("packets:Q", title="Test packets"),
        y=alt.Y("family:N", sort="-x", title=None),
        tooltip=["family", alt.Tooltip("packets:Q", format=","), alt.Tooltip("share:Q", format=".1%")])
    st.altair_chart(chart.properties(height=260), width="stretch")
    st.caption(f"Always predicting {counts.idxmax()} already gives {majority_accuracy(confusion):.3f} accuracy, "
               "so macro-F1, where every family counts equally, is the main metric.")

st.subheader("Features")
st.write(", ".join(summary["features"]))
st.caption(f"Removed because they are constant: {', '.join(summary['constant_columns'])}. "
           "Hash, Arch and MalwareFamily are identifiers and the label, not features.")
