import numpy as np
import pandas as pd

from src.models.importance import build_summary_matrix, compute_permutation_importance


def _synthetic_sample(n=300, seed=2):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    y = 0.2 * a + rng.normal(0, 0.05, n)  # only 'a' matters
    return pd.DataFrame({"dv": y, "a_zscore": a, "b_zscore": b})


def test_compute_permutation_importance_ranks_informative_regressor_higher():
    df = _synthetic_sample()
    out = compute_permutation_importance(df, "dv", ["a_zscore", "b_zscore"], n_repeats=10)
    assert out.iloc[0]["regressor"] == "a_zscore"
    assert out.set_index("regressor").loc["a_zscore", "permutation_importance_mean"] > \
           out.set_index("regressor").loc["b_zscore", "permutation_importance_mean"]


def test_build_summary_matrix_handles_all_none_optional_inputs():
    univariate = pd.DataFrame(
        {
            "measure_id": ["X", "Y"],
            "display_name": ["Measure X", "Measure Y"],
            "source_family": ["market", "household"],
            "horizon_bucket": ["long", "near"],
            "n_obs": [100, 100],
            "coef": [0.01, -0.02],
            "t": [2.0, -1.5],
            "pvalue": [0.05, 0.13],
            "oos_r2": [-0.1, -0.2],
        }
    )
    out = build_summary_matrix(univariate)
    assert len(out) == 2
    assert "convergence_rank_score" in out.columns
    assert out["incremental_oos_r2"].isna().all()


def test_build_summary_matrix_sorts_by_convergence_rank_descending():
    univariate = pd.DataFrame(
        {
            "measure_id": ["X", "Y", "Z"],
            "display_name": ["Measure X", "Measure Y", "Measure Z"],
            "source_family": ["market", "household", "professional"],
            "horizon_bucket": ["long", "near", "medium"],
            "n_obs": [100, 100, 100],
            "coef": [0.01, -0.02, 0.05],
            "t": [3.0, -0.5, 1.0],
            "pvalue": [0.01, 0.6, 0.3],
            "oos_r2": [-0.05, -0.4, -0.2],
        }
    )
    out = build_summary_matrix(univariate)
    scores = out["convergence_rank_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_build_summary_matrix_incorporates_mention_counts():
    univariate = pd.DataFrame(
        {
            "measure_id": ["X"],
            "display_name": ["Measure X"],
            "source_family": ["market"],
            "horizon_bucket": ["long"],
            "n_obs": [100],
            "coef": [0.01],
            "t": [2.0],
            "pvalue": [0.05],
            "oos_r2": [-0.1],
        }
    )
    mentions = pd.Series({"X": 25})  # indexed by measure_id, not display_name
    out = build_summary_matrix(univariate, mention_counts=mentions)
    assert out.iloc[0]["mention_count"] == 25
    assert out.iloc[0]["mention_share"] == 1.0
