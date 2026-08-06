#!/usr/bin/env python
"""Stage 8: final paper-ready outputs and research summary (spec section 32).

Produces:
  output/tables/machine_readable_summary.csv -- one row per measure with
    every metric from every empirical test, spec section 32's requested
    schema (measure, family, horizon, N, univariate_beta, univariate_t,
    multivariate_beta, incremental_oos_r2, permutation_importance,
    mention_share, event_response, robustness_pass_rate).
  output/tables/*.tex -- LaTeX exports of Tables 4, 7, 8.
  output/research_summary.md -- the qualitative synthesis, built from
    the actual numbers in the tables above (not hand-waved), explicitly
    distinguishing "most frequently mentioned" from "most predictive"
    from "largest conditional coefficient" from "strongest release-event
    response" per spec section 32's closing instruction that these are
    not necessarily the same measure.

Usage: python scripts/08_export_paper_outputs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "output" / "tables"
OUTPUT_DIR = PROJECT_ROOT / "output"


def build_machine_readable_summary() -> pd.DataFrame:
    importance = pd.read_csv(TABLES_DIR / "table7_importance_ranking.csv")
    multivariate = pd.read_csv(TABLES_DIR / "table5_multivariate_models.csv")

    ols_coefs = multivariate[multivariate["method"] == "ols"].set_index("term")["coef"]

    out = importance.copy()
    out["multivariate_beta"] = out["measure_id"].map(
        lambda mid: next((ols_coefs.get(c) for c in ols_coefs.index if c.startswith(mid + "_") or c == mid), None)
    )

    cols = [
        "measure_id", "display_name", "source_family", "horizon_bucket", "N",
        "univariate_beta", "univariate_t", "univariate_pvalue", "multivariate_beta",
        "incremental_oos_r2", "permutation_importance_mean", "mention_count", "mention_share",
        "event_response_max_abs_t", "robustness_pass_rate", "convergence_rank_score",
    ]
    out = out[[c for c in cols if c in out.columns]]
    out = out.rename(columns={"permutation_importance_mean": "permutation_importance", "event_response_max_abs_t": "event_response"})
    out_path = TABLES_DIR / "machine_readable_summary.csv"
    out.to_csv(out_path, index=False)
    logger.info("Wrote machine-readable summary: %d measures -> %s", len(out), out_path)
    return out


def export_latex_tables() -> None:
    for name in ["table4_univariate_models", "table7_importance_ranking", "table8_robustness"]:
        csv_path = TABLES_DIR / f"{name}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].round(4)
        tex_path = TABLES_DIR / f"{name}.tex"
        try:
            df.to_latex(tex_path, index=False, longtable=False, escape=True)
            logger.info("Wrote LaTeX table -> %s", tex_path)
        except Exception as exc:
            logger.warning("Could not export %s to LaTeX: %s", name, exc)


def _fmt(x, digits=3) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def write_research_summary(summary: pd.DataFrame) -> None:
    by_univariate_t = summary.reindex(summary["univariate_t"].abs().sort_values(ascending=False).index)
    by_incremental_r2 = summary.dropna(subset=["incremental_oos_r2"]).sort_values("incremental_oos_r2", ascending=False)
    by_mentions = summary.dropna(subset=["mention_share"]).sort_values("mention_share", ascending=False)
    by_event = summary.dropna(subset=["event_response"]).sort_values("event_response", ascending=False)
    by_convergence = summary.sort_values("convergence_rank_score", ascending=False)

    family_group = summary.groupby("source_family")["convergence_rank_score"].mean().sort_values(ascending=False)
    horizon_group = summary.groupby("horizon_bucket")["convergence_rank_score"].mean().sort_values(ascending=False)

    top_conv = by_convergence.iloc[0]
    top_t = by_univariate_t.iloc[0]
    top_r2 = by_incremental_r2.iloc[0] if not by_incremental_r2.empty else None
    top_mention = by_mentions.iloc[0] if not by_mentions.empty else None
    top_event = by_event.iloc[0] if not by_event.empty else None

    lines = []
    lines.append("# Research Summary: Which Inflation-Expectations Measure Does the Fed Respond to Most?\n")
    lines.append(
        "This summary is generated directly from the pipeline's output tables "
        "(output/tables/table4-8*.csv, machine_readable_summary.csv) -- every number "
        "below is a live computed value, not a hand-written claim.\n"
    )

    lines.append("## Answering the spec's specific questions\n")

    lines.append(
        f"**1. Which INDIVIDUAL measure appears most closely associated with Fed stance changes?**\n\n"
        f"By convergence across all empirical tests (the section-22 summary matrix, not a single metric): "
        f"**{top_conv['display_name']}** ({top_conv['source_family']}, {top_conv['horizon_bucket']}-horizon), "
        f"convergence rank score {_fmt(top_conv['convergence_rank_score'])}. "
        f"By univariate significance alone: **{top_t['display_name']}** "
        f"(t = {_fmt(top_t['univariate_t'])}, p = {_fmt(top_t['univariate_pvalue'])}). "
        f"These are not always the same measure -- see the explicit distinctions below.\n"
    )

    lines.append(
        f"**2. Which SOURCE FAMILY appears most important?**\n\n"
        f"Ranked by mean convergence score across member measures:\n\n"
        + "\n".join(f"- {fam}: {_fmt(score)}" for fam, score in family_group.items()) + "\n"
    )

    lines.append(
        f"**3. Does the Fed respond more strongly to short- or long-horizon expectations?**\n\n"
        f"Ranked by mean convergence score across horizon buckets:\n\n"
        + "\n".join(f"- {h}: {_fmt(score)}" for h, score in horizon_group.items())
        + (
            "\n\nThe data " + ("support" if horizon_group.index[0] == "long" else "do NOT clearly support")
            + " the spec's stated prior that long-run anchoring measures matter most; "
            "treat this as a hypothesis the data were allowed to reject, not a predetermined conclusion.\n"
        )
    )

    doc_type_note = (
        "See output/tables/table8_robustness.csv's speech_only / fomc_only variant rows for the "
        "per-measure comparison of speeches vs. collective FOMC communications; robustness_pass_rate "
        "in the summary table already reflects whether a measure's association survives this split."
    )
    lines.append(f"**4. Do results differ between speeches and collective FOMC communications?**\n\n{doc_type_note}\n")

    lines.append(
        "**5. Do policymakers respond differently when household, professional, and market expectations diverge?**\n\n"
        "See Figure 8 (household-minus-market and professional-minus-market standardized spreads over time) "
        "for the descriptive divergence series; a formal interaction-term test of this hypothesis "
        "(spec section 25) is left as a documented extension beyond this pipeline's current model set.\n"
    )

    if top_mention is not None:
        lines.append(
            f"**6. Which measures are most often directly cited?**\n\n"
            f"**{top_mention['display_name']}** leads by mention share ({_fmt(top_mention['mention_share'])}). "
            f"Full ranking in output/figures/figure4_mention_frequency.png and expectation_mentions.parquet.\n"
        )

    lines.append(
        "**7. How much does the conclusion change after macro controls?**\n\n"
        "Every regression in this pipeline (Tables 4-8) already includes the macro control panel "
        "(headline/core CPI and PCE, unemployment, payrolls, real GDP, 2y/10y Treasury yields, "
        "the effective funds rate, oil prices, and financial conditions) by construction -- there is "
        "no \"before controls\" version to compare against in the current design; a no-controls "
        "ablation is a natural extension.\n"
    )

    lines.append(
        "**8. How much does the conclusion change across monetary-policy regimes?**\n\n"
        "See Table 8's pre_2020 / pandemic_2020_2021 / post_2022_tightening / excluding_pandemic rows "
        "per measure. robustness_pass_rate in the summary table is the share of ALL tested variants "
        "(regime splits, document-type splits, change-vs-level, alternative inference) whose "
        "coefficient sign matched the full-sample baseline.\n"
    )

    lines.append("## Explicit distinctions (spec section 32's closing instruction)\n")
    lines.append(
        f"- **Most frequently mentioned:** {top_mention['display_name'] if top_mention is not None else 'n/a'}\n"
        f"- **Most predictive (incremental out-of-sample R^2):** {top_r2['display_name'] if top_r2 is not None else 'n/a'}\n"
        f"- **Largest conditional coefficient (|univariate t-stat|):** {top_t['display_name']}\n"
        f"- **Strongest release-event response:** {top_event['display_name'] if top_event is not None else 'n/a'}\n"
        f"- **Most robust across specifications:** "
        f"{summary.dropna(subset=['robustness_pass_rate']).sort_values('robustness_pass_rate', ascending=False).iloc[0]['display_name'] if summary['robustness_pass_rate'].notna().any() else 'n/a'}\n"
    )
    lines.append(
        "\nThese are deliberately NOT collapsed into one another. A measure can be heavily discussed "
        "in speeches without being the strongest statistical predictor, or vice versa -- see spec "
        "section 33's point that Fed officials themselves say they look at a range of measures rather "
        "than treating any one as definitive.\n"
    )

    lines.append("## Caveat on causality\n")
    lines.append(
        "Every result in this project is a predictive ASSOCIATION conditional on lagged information "
        "and macro controls, not a causal effect. Reverse causality (expectations reacting to anticipated "
        "Fed rhetoric), confounding by common macro shocks, and the non-random timing of speeches are all "
        "live concerns discussed throughout the codebase (see src/features/align_information_set.py and "
        "src/models/event_study.py's docstrings) and not resolved by any specification here.\n"
    )

    out_path = OUTPUT_DIR / "research_summary.md"
    out_path.write_text("\n".join(lines))
    logger.info("Wrote research summary -> %s", out_path)


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_machine_readable_summary()
    export_latex_tables()
    write_research_summary(summary)
    logger.info("Stage 8 completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
