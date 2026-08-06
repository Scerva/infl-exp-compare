import importlib.util
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "stage8", Path(__file__).resolve().parents[1] / "scripts" / "08_export_paper_outputs.py"
)
stage8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage8)


def test_fmt_handles_none_and_nan():
    assert stage8._fmt(None) == "n/a"
    assert stage8._fmt(float("nan")) == "n/a"


def test_fmt_rounds_to_requested_digits():
    assert stage8._fmt(1.23456, digits=2) == "1.23"
    assert stage8._fmt(1.0, digits=3) == "1.000"


def test_write_research_summary_produces_nonempty_file(tmp_path, monkeypatch):
    monkeypatch.setattr(stage8, "OUTPUT_DIR", tmp_path)
    summary = pd.DataFrame(
        {
            "measure_id": ["X", "Y"],
            "display_name": ["Measure X", "Measure Y"],
            "source_family": ["market", "household"],
            "horizon_bucket": ["long", "near"],
            "univariate_t": [2.5, -1.0],
            "univariate_pvalue": [0.02, 0.3],
            "incremental_oos_r2": [0.01, -0.02],
            "mention_share": [0.6, 0.4],
            "event_response": [1.5, 0.5],
            "robustness_pass_rate": [0.9, 0.5],
            "convergence_rank_score": [0.8, 0.3],
        }
    )
    stage8.write_research_summary(summary)
    out_file = tmp_path / "research_summary.md"
    assert out_file.exists()
    text = out_file.read_text()
    assert "Measure X" in text
    assert "Explicit distinctions" in text
