"""Release event studies (spec section 18).

For a discrete survey/model release (Michigan, NY Fed SCE, SPF, ATSIX,
Atlanta BIE), does Fed communication occurring shortly AFTER that release
show a stance shift associated with the release's surprise (standardized
change from the prior release)? This reuses the {measure}_days_since_release
and {measure}_value_change_zscore columns already produced by the
information-set-aligned panel (src/features/align_information_set.py):
"within the event window" is just ``days_since_release <= window``, and
the change column IS the innovation/surprise proxy the spec calls for
when an external consensus-expectation series isn't available.

NAMING DISCIPLINE (spec section 18's explicit warning): these are
labeled "release-window associations," never "causal impulse responses"
-- no identification strategy here rules out reverse causality or
confounding by whatever else happened to be going on that week.
"""
from __future__ import annotations

import pandas as pd

from src.models.baseline import fit_baseline_model
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_WINDOWS = [3, 5, 10]


def release_event_window_association(
    panel: pd.DataFrame,
    measure_id: str,
    windows: list[int] | None = None,
    dv_col: str = "stance_score",
    control_cols: list[str] | None = None,
    min_relevance: float = 0.10,
) -> pd.DataFrame:
    """One row per window: association between a measure's release
    surprise and Fed stance within that many days after the release.
    """
    windows = windows or DEFAULT_WINDOWS
    control_cols = control_cols or []
    days_col = f"{measure_id}_days_since_release"
    change_col = f"{measure_id}_value_change_zscore"
    if days_col not in panel.columns or change_col not in panel.columns:
        raise KeyError(f"Panel is missing {days_col!r} or {change_col!r}; check measure_id.")

    relevant = panel[panel["monetary_policy_relevance"] >= min_relevance]
    rows = []
    for w in windows:
        subset = relevant[(relevant[days_col] >= 0) & (relevant[days_col] <= w)]
        subset = subset.dropna(subset=[dv_col, change_col] + [c for c in control_cols if c in subset.columns])
        if len(subset) < 20:
            logger.warning("%s window=%d: only %d observations; skipping.", measure_id, w, len(subset))
            continue
        try:
            result = fit_baseline_model(
                subset, dv_col, [change_col], control_cols,
                speaker_fe=True, doctype_fe=True, year_fe=False, compute_oos=False,
            )
        except Exception as exc:
            # logger.error (not .exception): see local_projections.py's
            # comment on the same pattern -- full traceback formatting
            # crashes on this Python/patsy combination.
            logger.error("Event-window model failed for %s window=%d: %s", measure_id, w, exc)
            continue
        coef = result.coef_row(change_col)
        if coef is None:
            continue
        rows.append(
            {
                "measure_id": measure_id,
                "window_days": w,
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


def release_event_study_all_measures(
    panel: pd.DataFrame,
    measure_ids: list[str],
    windows: list[int] | None = None,
    dv_col: str = "stance_score",
    control_cols: list[str] | None = None,
) -> pd.DataFrame:
    frames = []
    for measure_id in measure_ids:
        try:
            frames.append(release_event_window_association(panel, measure_id, windows, dv_col, control_cols))
        except KeyError:
            logger.warning("Skipping %s: required columns not in panel.", measure_id)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
