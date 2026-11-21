from common import setup_page

import streamlit as st

from results_explorer import cd_figures, stats_tables

TASKS = {"fs": "Feature selection (feature mask)", "threshold": "Tuning the threshold t",
         "components": "Tuning the components k", "joint": "Tuning t and k together"}
SPACES = {"grid": "pcap grid: scaler, reduction, k, classifier", "selection": "pcap selection settings"}

setup_page("Statistical Tests",
           "Are the differences between the metaheuristics real, or chance? Saved results only; nothing is "
           "recomputed here. The tables are made by run_statistical_tests in src/stats_tests.py.")

tables = stats_tables()
figures = {path.stem.removeprefix("cd_landsat_"): path for path in cd_figures()}

st.header("Landsat: ten algorithms on the same folds")
st.markdown("Each algorithm ran 15 times (3 seeds x 5 outer folds) on the same folds, so the runs are paired. "
            "The **Friedman test** asks whether the algorithms differ at all, using their ranks within each run; "
            "the **Nemenyi test** then finds which pairs differ.")
friedman = tables["landsat_friedman"].assign(task=lambda t: t.task.map(TASKS))
st.dataframe(friedman.rename(columns={"task": "Task", "n_blocks": "Runs", "n_algorithms": "Algorithms",
                                      "statistic": "Friedman statistic", "p_value": "p-value",
                                      "best_algorithm": "Best average rank", "best_mean_rank": "Its rank"})
             .style.format({"Friedman statistic": "{:.2f}", "p-value": "{:.4f}", "Its rank": "{:.2f}"}),
             hide_index=True, width="stretch")
st.caption("p below 0.05 means at least some algorithms really differ. For tuning k most algorithms reach the "
           "same optimum, so their ranks differ only by chance.")

task = st.selectbox("Task", list(TASKS), format_func=TASKS.get)
left, right = st.columns([3, 2])
with left:
    st.image(str(figures[task]))
    st.caption("Critical-difference diagram: average rank of each algorithm (1 = best). Algorithms joined by a red "
               "bar are not significantly different; the bar at the top shows the critical difference.")
with right:
    pairs = tables["landsat_nemenyi"]
    pairs = pairs[pairs.task == task]
    significant = pairs[pairs.significant]
    st.metric("Significantly different pairs", f"{len(significant)} of {len(pairs)}")
    if significant.empty:
        st.info("No pair of algorithms differs significantly on this task.")
    else:
        st.dataframe(significant[["better", "worse", "rank_better", "rank_worse", "rank_gap"]]
                     .rename(columns={"better": "Better", "worse": "Worse", "rank_better": "Better rank",
                                      "rank_worse": "Worse rank", "rank_gap": "Gap"})
                     .style.format({"Better rank": "{:.2f}", "Worse rank": "{:.2f}", "Gap": "{:.2f}"}),
                     hide_index=True, width="stretch")

st.header("pcap: ten algorithms against random search")
st.markdown("Each algorithm ran 20 times with independent seeds on one fixed search space, so the runs are not "
            "paired. The **Kruskal-Wallis test** replaces Friedman, and each algorithm is compared with random "
            "search by a **Mann-Whitney U test**, with the **Holm correction** for testing ten pairs at once.")
kruskal = tables["pcap_kruskal"].assign(space=lambda t: t.space.map(SPACES))
st.dataframe(kruskal.rename(columns={"space": "Search space", "budget": "Budget", "n_algorithms": "Methods",
                                     "runs_per_algorithm": "Runs each", "statistic": "Kruskal-Wallis statistic",
                                     "p_value": "p-value", "best_algorithm": "Best mean", "best_mean": "Its best F1"})
             .style.format({"Kruskal-Wallis statistic": "{:.2f}", "p-value": "{:.2e}", "Its best F1": "{:.4f}"}),
             hide_index=True, width="stretch")

one, two = st.columns(2)
space = one.radio("Search space", list(SPACES), format_func=SPACES.get, horizontal=True)
budgets = sorted(tables["pcap_vs_random"].budget.unique())
budget = two.radio("Budget (fitness evaluations)", budgets, index=len(budgets) // 2, horizontal=True)
versus = tables["pcap_vs_random"]
versus = versus[(versus.space == space) & (versus.budget == budget)]
better = versus[versus.significant & (versus.difference > 0)]
worse = versus[versus.significant & (versus.difference < 0)]
row = st.columns(2)
row[0].metric("Significantly better than random search", len(better))
row[1].metric("Significantly worse than random search", len(worse))
st.dataframe(versus[["algorithm", "mean", "baseline_mean", "difference", "p_value", "p_holm", "significant"]]
             .rename(columns={"algorithm": "Algorithm", "mean": "Mean best F1", "baseline_mean": "Random search",
                              "difference": "Difference", "p_value": "p-value", "p_holm": "Holm p-value",
                              "significant": "Significant"})
             .style.format({"Mean best F1": "{:.4f}", "Random search": "{:.4f}", "Difference": "{:+.4f}",
                            "p-value": "{:.4f}", "Holm p-value": "{:.4f}"}),
             hide_index=True, width="stretch")
st.caption("On the best F1 found, no metaheuristic is significantly better than random search at any budget; "
           "several are significantly worse. This measures the score reached, not how often the exact best "
           "configuration is hit.")

st.subheader("How often is the exact best configuration found?")
st.markdown("Each run either finds the best configuration of its search space or does not. **Fisher's exact test** "
            "compares each algorithm's hits with random search's hits in the same 20 runs, with the **Holm "
            "correction** again for ten comparisons. Same search space and budget as above.")
hits = tables["pcap_hit_rates"]
hits = hits[(hits.space == space) & (hits.budget == budget)]
row = st.columns(2)
row[0].metric("Hit the best significantly more often", len(hits[hits.significant & (hits.difference > 0)]))
row[1].metric("Hit the best significantly less often", len(hits[hits.significant & (hits.difference < 0)]))
shown = hits.assign(found=hits.hits.astype(str) + " of " + hits.runs.astype(str),
                    random=hits.baseline_hits.astype(str) + " of " + hits.baseline_runs.astype(str))
st.dataframe(shown[["algorithm", "found", "hit_rate", "random", "baseline_rate", "p_value", "p_holm", "significant"]]
             .rename(columns={"algorithm": "Algorithm", "found": "Runs that hit the best", "hit_rate": "Hit rate",
                              "random": "Random search", "baseline_rate": "Random hit rate", "p_value": "p-value",
                              "p_holm": "Holm p-value", "significant": "Significant"})
             .style.format({"Hit rate": "{:.0%}", "Random hit rate": "{:.0%}", "p-value": "{:.4f}",
                            "Holm p-value": "{:.4f}"}),
             hide_index=True, width="stretch")
st.caption("A higher hit rate is not proven by a few extra hits in 20 runs: on the grid at budget 400, GWO hit the "
           "best in 16 runs and random search in 8, which is not significant after the Holm correction. Only SSA on "
           "the selection settings at budget 400 hits the best significantly more often than random search.")
