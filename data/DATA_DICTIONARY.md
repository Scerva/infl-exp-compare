# Data Dictionary

Describes every processed dataset in `data/processed/` and the naming
convention for the (very wide) model panel, rather than listing all 361
model-panel columns individually -- the convention is systematic and applies
uniformly, see [Column naming convention](#column-naming-convention-model_panelparquet).

## documents_scored.parquet

One row per Fed communication (speech, FOMC minutes, or FOMC statement).

| Column | Type | Description |
|---|---|---|
| `document_id` | str | Unique ID: speech URL slug (e.g. `powell20240131a`) or `fomc_minutes_YYYYMMDD` / `fomc_statement_YYYYMMDD`. |
| `document_type` | str | `speech`, `fomc_minutes`, or `fomc_statement`. |
| `event_date` | date | For speeches: the speech date. For minutes/statements: the **policy decision date** (meeting end), not the publication date. |
| `publication_date` | date | When the document became public. For minutes this is ~3 weeks after `event_date`; for speeches/statements it equals `event_date`. |
| `meeting_date` | date | FOMC meeting end date (minutes/statements only; null for speeches). |
| `speaker` | str/null | Speaker name; null for FOMC-collective documents (minutes/statements). |
| `speaker_role` | str/null | Title at time of speech (e.g. "Vice Chair for Supervision"), or `"FOMC"` for collective documents. |
| `title`, `venue`, `source_url` | str | As scraped. |
| `raw_text` | str | Full raw HTML of the source page (provenance). |
| `clean_text` | str | Extracted body paragraphs only (navigation/footnotes/share-widget text removed). |
| `monetary_policy_relevance` | float [0,1] | Share of sentences hitting >=1 of 9 topic patterns (inflation, inflation expectations, labor market, growth, monetary policy, rates, financial conditions, balance sheet, dual mandate). Documents below 0.10 are flagged, not dropped. |
| `inflation_relevance`, `expectations_relevance` | float [0,1] | Same construction, restricted to the `inflation` / `inflation_expectations` topic only. |
| `hawk_score`, `dove_score` | float >=0 | Weighted-mean positive/negative parts of sentence-level stance (Model A dictionary). |
| `neutral_score` | float [0,1] | Share of relevant sentences with zero hawk/dove phrase hits. |
| `stance_score` | float [-1,1] | Document-level weighted-mean stance = hawk_score - dove_score. THE primary dependent-variable input. |
| `stance_uncertainty` | float >=0 | Standard deviation of sentence-level stance within the document. |
| `n_sentences`, `n_relevant_sentences` | int | Denominators behind the relevance/stance scores. |

## expectation_mentions.parquet

One row per direct textual reference to a named expectations measure.

| Column | Description |
|---|---|
| `document_id`, `date`, `speaker`, `document_type` | Join keys back to documents_scored. |
| `sentence_index` | Position of the mentioning sentence within the document. |
| `measure_family` | household / professional / market / business / survey_generic / horizon_generic. |
| `specific_measure` | e.g. "Michigan Survey", "TIPS / Breakevens", "SPF". Coarser than the 21-series model registry by design -- text mentions rarely specify T5YIE vs T10YIE vs T5YIFR, for instance. |
| `sentence`, `context` | The mentioning sentence and its +/-2-sentence window. |
| `local_stance_score` | Model A stance score computed on `context` only. |
| `concern_score` | Independent lexicon-based "concern" intensity (elevated/persistent/risk/de-anchor/etc.), direction-agnostic. |

## Per-source raw panels (fred_series_long, nyfed_sce, philly_spf, livingston,
## atsix_full_curve / atsix_selected_horizons, atlanta_bie, macro_controls)

Each carries, at minimum: a reference-period column (`date` /
`reference_month` / `reference_quarter` / `reference_date` /
`survey_period_end`), a `release_date` column, a `release_date_estimated`
boolean (True where reconstructed rather than sourced from an official
calendar -- see each `src/data/*.py` module's docstring for the exact rule),
and a `value` column (or named value columns for wide files like nyfed_sce).
`series_id` distinguishes series within long-format files.

## model_panel.parquet

One row per document (same base rows as documents_scored.parquet), widened
with every information-set-aligned expectations/macro series plus derived
stance features.

### Speaker-relative stance features (spec section 8)

| Column | Description |
|---|---|
| `stance_group` | `speaker` if present, else `document_type` (groups FOMC-collective documents together). |
| `delta_stance` | `stance_score` minus the SAME group's previous document's `stance_score`. Null for each group's first document. The primary regression dependent variable. |
| `speaker_rolling_mean_stance` | Mean of that group's stance history strictly BEFORE this document (no look-ahead). |
| `stance_vs_speaker_rolling_mean` | `stance_score` minus the above -- controls for a policymaker's structural hawkishness/dovishness. |

### Column naming convention (model_panel.parquet)

For each aligned series with prefix `{S}` (e.g. `T5YIE`, `SPF_CPI10`,
`MACRO_CPI_HEADLINE`, `SCE_median_1yr`, `ATSIX_60m`):

- `{S}_value` -- raw level as of the asof-merge (macro controls only, which
  aren't standardized).
- `{S}_value_level` -- raw level (expectations series).
- `{S}_value_change_1d` / `_5d` / `_20d` (daily) or `_1m`/`_3m`/`_12m`
  (monthly) or `_1q`/`_4q` (quarterly) or `_1p` (semiannual) -- change over
  that many periods at the series' native frequency.
- `{S}_value_zscore` -- rolling z-score (trailing ~1 year of native-frequency
  observations); THE standardized regressor used in every regression table.
- `{S}_value_change_zscore` -- z-score of the primary (shortest) change
  column; used as the "innovation/surprise" proxy in event studies and local
  projections.
- `{S}_value_dev_from_trailing_mean` -- level minus trailing rolling mean.
- `{S}_value_pctile` -- trailing rolling percentile rank in [0,1].
- `{S}_value_dev_from_target` -- level minus the series' target anchor
  (`config/sources.yaml`'s `target_anchor`; null where no anchor is
  economically well-defined, e.g. Michigan's "prices in general" concept).
- `{S}_release_date` -- the actual release-date value used for that row's
  asof-merge (auditable: confirms no look-ahead).
- `{S}_days_since_release` -- event_date minus that release date, in days.

See `src/models/registry.py` for the canonical mapping from each of the 21
primary measures' short `measure_id` (e.g. `SPF_CPI5YR`) to its panel column
prefix and `display_name`.

## output/tables/machine_readable_summary.csv

The final per-measure result table (spec section 32's exact requested
schema): `measure_id`, `display_name`, `source_family`, `horizon_bucket`, `N`,
`univariate_beta`, `univariate_t`, `univariate_pvalue`, `multivariate_beta`,
`incremental_oos_r2`, `permutation_importance`, `mention_count`,
`mention_share`, `event_response`, `robustness_pass_rate`,
`convergence_rank_score` (a transparent average-percentile-rank sort key
across the other numeric columns -- not a claimed "importance weight").
