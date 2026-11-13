from common import make_show, setup_page

import altair as alt
import pandas as pd
import streamlit as st

from search_explorer import CHOICES, SPACES, compare_algorithms, describe, load_space, search_with_curve

setup_page("Metaheuristic Search",
           "Every configuration of the space was scored once with 4-fold cross-validation (macro-F1), so a search "
           "here looks scores up instead of training models, and its result can be compared with the true best. "
           "In a real search each new configuration would cost four model fits.")


@st.cache_resource
def space_for(kind):
    return load_space(kind)


with st.sidebar:
    st.header("Settings")
    kind = SPACES[st.radio("Search space", list(SPACES))]
    budget = st.slider("Budget (fitness evaluations)", 25, 400, 200, 25)
    population = st.slider("Population size", 10, 50, 20, 5)
    near = st.select_slider("Near-best margin (macro-F1)", [0.002, 0.005, 0.01, 0.02], value=0.01)

space = space_for(kind)
st.markdown(f"**{space.n:,} configurations.** True best: **{describe(space.best_cfg, kind)}**, CV macro-F1 "
            f"{space.best:.4f}. {space.count_within(near)} configurations lie within {near} of it.")

single, compare = st.tabs(["One run", "Compare all algorithms"])

with single:
    left, right = st.columns(2)
    name = left.selectbox("Algorithm", CHOICES, index=CHOICES.index("GWO"))
    seed = right.number_input("Seed", min_value=1, max_value=1000, value=1, step=1)
    if st.button("Run search", type="primary"):
        result, curve = search_with_curve(name, space, budget, population, int(seed), near)
        found = (result["scaler"], result["reduction"], result["k"], result["classifier"])
        row = st.columns(4)
        row[0].metric("Best CV macro-F1 found", f"{result['best_f1']:.4f}", f"{-result['gap_to_best']:+.4f} vs best")
        row[1].metric("True rank of the result", f"{result['best_rank']} of {space.n}")
        row[2].metric("Distinct configurations tried", result["unique_configs"])
        row[3].metric("Fitness calls", result["calls"])
        st.markdown(f"Best found: **{describe(found, kind)}**. "
                    + ("It is the true best. " if result["hit_best"] else "")
                    + (f"First within {near} of the best after {result['unique_to_near']} configurations."
                       if result["hit_near"] else f"Never came within {near} of the best."))
        best_line = pd.DataFrame({"best_score": [space.best]})
        chart = alt.Chart(curve).mark_line(interpolate="step-after").encode(
            x=alt.X("configs_tried:Q", title="Distinct configurations tried"),
            y=alt.Y("best_score:Q", title="Best CV macro-F1 so far", scale=alt.Scale(zero=False)))
        rule = alt.Chart(best_line).mark_rule(color="#d9822b", strokeDash=[6, 4]).encode(y="best_score:Q")
        st.altair_chart((chart + rule).properties(height=320), width="stretch")
        st.caption("Orange dashed line: the true best configuration of the space.")

with compare:
    n_seeds = st.slider("Seeds per algorithm", 3, 20, 10)
    if st.button("Run every algorithm"):
        with st.spinner("Running all algorithms..."):
            summary = compare_algorithms(kind, budget, n_seeds, population, near, show=make_show(6))
        st.dataframe(summary.style.format({"hit_best_%": "{:.0f}", "hit_near_%": "{:.0f}", "mean_best_f1": "{:.4f}",
                                           "mean_rank": "{:.1f}", "median_rank": "{:.1f}", "unique_configs": "{:.0f}",
                                           "median_unique_to_near": "{:.0f}"}), hide_index=True, width="stretch")
        bars = alt.Chart(summary).mark_bar().encode(
            x=alt.X("hit_best_%:Q", title="Runs that found the true best (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("algorithm:N", sort="-x", title=None),
            color=alt.condition(alt.datum.algorithm == "RANDOM", alt.value("#c9ced6"), alt.value("#2f6db5")))
        st.altair_chart(bars.properties(height=320), width="stretch")
        st.caption(f"{n_seeds} seeds per algorithm, budget {budget}. Grey: random search with the same budget.")
