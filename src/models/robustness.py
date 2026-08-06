"""Robustness suite (spec section 23).

Rather than a single robust/not-robust verdict, this refits the baseline
univariate specification for each top-ranked measure across a battery of
sample and specification variants, and reports whether the coefficient's
SIGN survives (the minimal bar for "this measure's association with Fed
stance isn't an artifact of one particular slice of the data or one
particular era"). Feeds Table 8 and the robustness_pass_rate column of
the section-22 importance summary matrix.

Variants implemented (a practical subset of section 23's full list,
chosen for genuine informativeness rather than exhaustive coverage):
regime splits (pre-2020 / pandemic 2020-21 / post-2022 tightening /
excluding pandemic), document-type splits (speech-only / FOMC-only),
level vs. standardized-change specification, and alternative inference
(cluster-by-speaker vs. Newey-West/HAC).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.baseline import build_regression_sample, fit_baseline_model
from src.models.registry import MeasureRegistryEntry
from src.utils.logging import get_logger

logger = get_logger(__name__)


REGIME_FILTERS = {
    "pre_2020": lambda df: df[df["event_date"] < "2020-01-01"],
    "pandemic_2020_2021": lambda df: df[(df["event_date"] >= "2020-01-01") & (df["event_date"] < "2022-01-01")],
    "post_2022_tightening": lambda df: df[df["event_date"] >= "2022-01-01"],
    "excluding_pandemic": lambda df: df[(df["event_date"] < "2020-01-01") | (df["event_date"] >= "2022-01-01")],
}
DOCTYPE_FILTERS = {
    "speech_only": lambda df: df[df["document_type"] == "speech"],
    "fomc_only": lambda df: df[df["document_type"].isin(["fomc_minutes", "fomc_statement"])],
}


def run_robustness_checks(
    panel: pd.DataFrame,
    measures: list[MeasureRegistryEntry],
    dv_col: str = "delta_stance",
    control_cols: list[str] | None = None,
    min_obs: int = 30,
) -> pd.DataFrame:
    control_cols = control_cols or []
    rows = []

    for m in measures:
        reg_col = m.level_zscore_col
        base_sample = build_regression_sample(panel, dv_col=dv_col, extra_required_cols=[reg_col] + control_cols)
        if len(base_sample) < min_obs:
            logger.warning("Skipping robustness for %s: only %d baseline observations.", m.measure_id, len(base_sample))
            continue
        try:
            baseline_result = fit_baseline_model(base_sample, dv_col, [reg_col], control_cols, compute_oos=False)
        except Exception as exc:
            logger.error("Baseline fit failed for %s: %s", m.measure_id, exc)
            continue
        baseline_coef = baseline_result.coef_row(reg_col)
        if baseline_coef is None:
            continue
        baseline_sign = np.sign(baseline_coef["coef"])
        rows.append(
            {
                "measure_id": m.measure_id, "variant": "baseline_full_sample",
                "coef": baseline_coef["coef"], "se": baseline_coef["se"], "t": baseline_coef["t"],
                "pvalue": baseline_coef["pvalue"], "n_obs": baseline_result.n_obs,
                "same_sign_as_baseline": True,
            }
        )

        all_filters = {**REGIME_FILTERS, **DOCTYPE_FILTERS}
        for variant_name, filter_fn in all_filters.items():
            sub = filter_fn(base_sample)
            if len(sub) < min_obs:
                logger.info("%s / %s: only %d obs, skipping.", m.measure_id, variant_name, len(sub))
                continue
            try:
                result = fit_baseline_model(sub, dv_col, [reg_col], control_cols, compute_oos=False)
            except Exception as exc:
                logger.error("Robustness variant %s failed for %s: %s", variant_name, m.measure_id, exc)
                continue
            coef = result.coef_row(reg_col)
            if coef is None:
                continue
            rows.append(
                {
                    "measure_id": m.measure_id, "variant": variant_name,
                    "coef": coef["coef"], "se": coef["se"], "t": coef["t"],
                    "pvalue": coef["pvalue"], "n_obs": result.n_obs,
                    "same_sign_as_baseline": bool(np.sign(coef["coef"]) == baseline_sign),
                }
            )

        # Level vs. standardized-change specification.
        if m.change_zscore_col in base_sample.columns:
            change_sample = build_regression_sample(panel, dv_col=dv_col, extra_required_cols=[m.change_zscore_col] + control_cols)
            if len(change_sample) >= min_obs:
                try:
                    change_result = fit_baseline_model(change_sample, dv_col, [m.change_zscore_col], control_cols, compute_oos=False)
                    coef = change_result.coef_row(m.change_zscore_col)
                    if coef is not None:
                        rows.append(
                            {
                                "measure_id": m.measure_id, "variant": "change_specification",
                                "coef": coef["coef"], "se": coef["se"], "t": coef["t"],
                                "pvalue": coef["pvalue"], "n_obs": change_result.n_obs,
                                "same_sign_as_baseline": bool(np.sign(coef["coef"]) == baseline_sign),
                            }
                        )
                except Exception as exc:
                    logger.error("Change-specification robustness failed for %s: %s", m.measure_id, exc)

        # Alternative inference: HAC/Newey-West instead of speaker-clustered.
        try:
            hac_result = fit_baseline_model(
                base_sample, dv_col, [reg_col], control_cols, cluster_col="__no_such_column__", compute_oos=False
            )
            coef = hac_result.coef_row(reg_col)
            if coef is not None:
                rows.append(
                    {
                        "measure_id": m.measure_id, "variant": "newey_west_se",
                        "coef": coef["coef"], "se": coef["se"], "t": coef["t"],
                        "pvalue": coef["pvalue"], "n_obs": hac_result.n_obs,
                        "same_sign_as_baseline": bool(np.sign(coef["coef"]) == baseline_sign),
                    }
                )
        except Exception as exc:
            logger.error("Newey-West robustness failed for %s: %s", m.measure_id, exc)

    return pd.DataFrame(rows)


def summarize_robustness_pass_rates(robustness_df: pd.DataFrame) -> pd.Series:
    """Share of non-baseline variants where the coefficient sign matched
    the full-sample baseline, per measure -- the robustness_pass_rate
    column spec section 22's summary matrix asks for.
    """
    non_baseline = robustness_df[robustness_df["variant"] != "baseline_full_sample"]
    return non_baseline.groupby("measure_id")["same_sign_as_baseline"].mean()
