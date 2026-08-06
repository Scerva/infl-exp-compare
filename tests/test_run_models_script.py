import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "stage6", Path(__file__).resolve().parents[1] / "scripts" / "06_run_models.py"
)
stage6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage6)


def test_mention_counts_by_measure_id_aggregates_shared_labels():
    mentions_df = pd.DataFrame(
        {"specific_measure": ["TIPS / Breakevens"] * 10 + ["Michigan Survey"] * 5 + ["SPF"] * 3}
    )
    out = stage6._mention_counts_by_measure_id(mentions_df)
    # T5YIE, T10YIE both map only to "TIPS / Breakevens" -> both get the full 10
    assert out["T5YIE"] == 10
    assert out["T10YIE"] == 10
    # T5YIFR maps to both "TIPS / Breakevens" and "5-Year, 5-Year Forward" (0 here)
    assert out["T5YIFR"] == 10
    assert out["MICH"] == 5
    assert out["SPF_CPI10"] == 3


def test_mention_counts_by_measure_id_unmapped_measure_gets_zero():
    mentions_df = pd.DataFrame({"specific_measure": ["Michigan Survey"] * 5})
    out = stage6._mention_counts_by_measure_id(mentions_df)
    # ATSIX/Cleveland-Fed measures have no MEASURE_ID_TO_MENTION_LABELS entry
    assert "ATSIX_12m" not in out.index


def test_well_covered_measures_filters_by_coverage_threshold():
    panel = pd.DataFrame(
        {
            "monetary_policy_relevance": [0.5] * 100,
            "delta_stance": range(100),
            "event_date": pd.date_range("2020-01-01", periods=100),
            "speaker": ["A"] * 100,
            "document_type": ["speech"] * 100,
            "T5YIE_value_zscore": [1.0] * 100,  # 100% coverage
            "MICH_value_zscore": [1.0] * 10 + [None] * 90,  # 10% coverage
        }
    )
    result = stage6._well_covered_measures(panel, min_coverage=0.30)
    ids = [m.measure_id for m in result]
    assert "T5YIE" in ids
    assert "MICH" not in ids
