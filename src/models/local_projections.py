"""Distributed-lag / local-projection analysis (spec section 19).

Estimates, for each horizon h, whether an expectation-measure innovation
at communication i predicts a stance change over the following h
communications:

    Hawkishness(i+h) - Hawkishness(i-1) = a_h + b_h * innovation_i
                                            + controls + e

Horizons here are EVENT-NUMBER horizons (h-th subsequent Fed
communication), the spec's primary choice for irregularly-spaced
documents; a calendar-day variant would require resampling to a regular
grid and is left as a documented extension rather than force-fit here.
"""
from __future__ import annotations

import pandas as pd

from src.models.baseline import fit_baseline_model
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_HORIZONS = range(1, 6)


def local_projection_coefficients(
    panel: pd.DataFrame,
    innovation_col: str,
    horizons: range | list[int] | None = None,
    dv_base_col: str = "stance_score",
    control_cols: list[str] | None = None,
    min_relevance: float = 0.10,
) -> pd.DataFrame:
    """One row per horizon h: beta_h and its confidence interval.

    Sample is restricted to monetary-policy-relevant documents BEFORE
    horizon-shifting so h-step-ahead targets always land on another
    relevant document, not an arbitrary off-topic one.
    """
    horizons = horizons if horizons is not None else DEFAULT_HORIZONS
    control_cols = control_cols or []

    df = panel[panel["monetary_policy_relevance"] >= min_relevance].sort_values("event_date").reset_index(drop=True)
    if innovation_col not in df.columns:
        raise KeyError(f"Panel is missing {innovation_col!r}.")

    rows = []
    for h in horizons:
        target_col = f"_lp_target_h{h}"
        df[target_col] = df[dv_base_col].shift(-h) - df[dv_base_col].shift(1)
        needed = [target_col, innovation_col] + [c for c in control_cols if c in df.columns]
        sample = df.dropna(subset=needed)
        if len(sample) < 30:
            logger.warning("Local projection h=%d: only %d observations; skipping.", h, len(sample))
            continue
        try:
            result = fit_baseline_model(
                sample, target_col, [innovation_col], control_cols,
                speaker_fe=False, doctype_fe=True, year_fe=False, compute_oos=False,
            )
        except Exception as exc:
            # logger.error (not .exception): a full traceback format
            # crashes on this Python/patsy combination when a PatsyError
            # propagates (a stdlib traceback-suggestion bug), so log the
            # message only.
            logger.error("Local projection failed at horizon %d: %s", h, exc)
            continue
        coef = result.coef_row(innovation_col)
        if coef is None:
            continue
        rows.append(
            {
                "horizon": h,
                "coef": coef["coef"],
                "se": coef["se"],
                "t": coef["t"],
                "pvalue": coef["pvalue"],
                "ci_low": coef["ci_low"],
                "ci_high": coef["ci_high"],
                "n_obs": result.n_obs,
            }
        )
    return pd.DataFrame(rows)
