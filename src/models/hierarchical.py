"""Hierarchical (partial-pooling) speaker model (spec section 17).

Fed officials may respond differently to different expectations measures;
pooling everyone into one coefficient hides that heterogeneity, but
fitting a fully separate regression per speaker overfits officials with
only a handful of speeches. A random-slope mixed-effects model partially
pools each speaker's coefficient toward the committee-wide average,
shrinking harder for speakers with less data -- exactly the "do not
over-interpret officials with very few speeches" instruction in the spec.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HierarchicalModelResult:
    committee_avg_coef: float
    committee_avg_se: float
    speaker_effects: pd.DataFrame  # speaker, n_documents, random_slope, shrunk_coef
    converged: bool
    n_obs: int
    n_groups: int


def fit_hierarchical_speaker_model(
    sample: pd.DataFrame,
    dv_col: str,
    regressor_col: str,
    control_cols: list[str] | None = None,
    min_speaker_docs: int = 5,
) -> HierarchicalModelResult:
    """Random-intercept + random-slope-on-``regressor_col`` mixed model,
    grouped by speaker. Speakers with fewer than ``min_speaker_docs``
    monetary-policy-relevant documents are pooled into the fixed effect
    only (their random slope is not separately reported, though they
    still contribute to estimating the committee-average).
    """
    control_cols = control_cols or []
    df = sample.dropna(subset=[dv_col, regressor_col, "speaker"] + control_cols).copy()
    df = df[df["speaker"].notna()]

    doc_counts = df["speaker"].value_counts()
    eligible_speakers = doc_counts[doc_counts >= min_speaker_docs].index
    if len(eligible_speakers) < 3:
        raise ValueError(f"Too few speakers with >= {min_speaker_docs} documents ({len(eligible_speakers)}) to fit a hierarchical model.")

    rhs = " + ".join([regressor_col] + control_cols)
    formula = f"{dv_col} ~ {rhs}"

    model = smf.mixedlm(formula, data=df, groups=df["speaker"], re_formula=f"~{regressor_col}")
    try:
        result = model.fit(reml=True, method="lbfgs", maxiter=200)
        converged = result.converged
    except Exception as exc:
        # logger.error (not .exception): full traceback formatting
        # crashes on this Python/patsy combination when a PatsyError is
        # in the chain -- log the message only.
        logger.error("Mixed model failed to converge, falling back to random-intercept-only: %s", exc)
        model = smf.mixedlm(formula, data=df, groups=df["speaker"])
        result = model.fit(reml=True, method="lbfgs", maxiter=200)
        converged = result.converged

    committee_avg = result.params.get(regressor_col, np.nan)
    committee_se = result.bse.get(regressor_col, np.nan)

    rows = []
    random_effects = result.random_effects
    for speaker, re in random_effects.items():
        slope_re = re.get(regressor_col, 0.0) if hasattr(re, "get") else 0.0
        rows.append(
            {
                "speaker": speaker,
                "n_documents": int(doc_counts.get(speaker, 0)),
                "random_slope_deviation": slope_re,
                "shrunk_coef": committee_avg + slope_re,
                "eligible_for_own_slope": speaker in eligible_speakers,
            }
        )
    speaker_effects = pd.DataFrame(rows).sort_values("shrunk_coef", ascending=False)

    return HierarchicalModelResult(
        committee_avg_coef=committee_avg,
        committee_avg_se=committee_se,
        speaker_effects=speaker_effects,
        converged=bool(converged),
        n_obs=len(df),
        n_groups=df["speaker"].nunique(),
    )
