import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from write_pcap_report import selection_tag, selection_vs_extraction


def test_selection_against_extraction_per_pair():
    comparison = pd.DataFrame([
        {"scaler": "Standard", "classifier": "DT", "method": "HeapBFS", "setting": "t = 0.80",
         "features_or_components": 18.0, "cv_f1_macro": 0.25},
        {"scaler": "Standard", "classifier": "DT", "method": "PCA", "setting": "k = 10",
         "features_or_components": 10.0, "cv_f1_macro": 0.24},
        {"scaler": "Normalizer", "classifier": "DT", "method": "MI_top", "setting": "k = 33",
         "features_or_components": 33.0, "cv_f1_macro": 0.25},
        {"scaler": "Normalizer", "classifier": "DT", "method": "PCA", "setting": "k = 14",
         "features_or_components": 14.0, "cv_f1_macro": 0.27}])
    table = selection_vs_extraction(comparison)
    assert table.Higher.tolist() == ["selection", "extraction"]
    assert table.iloc[0]["Best selection"] == "HeapBFS t = 0.80 (18 feat.)"


def test_selection_tags_match_the_saved_report_names():
    rows = pd.DataFrame([
        {"scaler": "Normalizer", "method": "HeapBFS", "param": "0.84", "classifier": "DT"},
        {"scaler": "Standard", "method": "MI_top", "param": "22.0", "classifier": "XGB"},
        {"scaler": "Normalizer", "method": "Mask", "param": "GA seed 2", "classifier": "DT"}])
    assert [selection_tag(r) for r in rows.itertuples()] == [
        "Normalizer_HeapBFS_t0.84_DT", "Standard_MI_top_k22_XGB", "Normalizer_Mask_GA_seed_2_DT"]


def test_weighting_section_reads_the_summary(tmp_path, monkeypatch):
    import write_pcap_report
    folder = tmp_path / "pcap_weighting_v1"
    folder.mkdir()
    rows = [{"family": "selection", "scaler": "Standard", "method": "HeapBFS", "param": 0.06, "classifier": "RF",
             "scheme": scheme, "f1_macro": f1, "balanced_accuracy": 0.3, "accuracy": 0.6, "f1_Gafgyt": 0.0}
            for scheme, f1 in (("none", 0.2), ("sqrt", 0.25), ("balanced", 0.22))]
    pd.DataFrame(rows).to_csv(folder / "summary.csv", index=False)
    monkeypatch.setattr(write_pcap_report, "RESULTS", tmp_path)
    text = "\n".join(write_pcap_report.weighting_section())
    assert "7. CLASS WEIGHTING" in text and "Standard + HeapBFS 0.06 + RF" in text and "0.25" in text
