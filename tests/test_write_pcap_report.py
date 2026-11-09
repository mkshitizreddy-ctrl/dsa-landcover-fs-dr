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
