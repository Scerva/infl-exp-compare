"""Registry of primary inflation-expectations measures used across every
econometric model in this project (univariate/multivariate horse races,
source-family models, importance ranking).

Each entry names the model_panel.parquet z-score column (standardized
level, spec section 13's requirement that coefficients be directly
comparable as "response per one-standard-deviation move") and the
standardized-change column, plus metadata (source_family, horizon
bucket) used to build family-level and short/medium/long-horizon
groupings for sections 16 and 26.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeasureRegistryEntry:
    measure_id: str
    display_name: str
    source_family: str  # household | professional | market | business | model
    horizon_bucket: str  # near | medium | long
    level_zscore_col: str
    change_zscore_col: str


PRIMARY_MEASURES: list[MeasureRegistryEntry] = [
    # --- market ---
    MeasureRegistryEntry("T5YIE", "5-Year Breakeven", "market", "medium", "T5YIE_value_zscore", "T5YIE_value_change_zscore"),
    MeasureRegistryEntry("T10YIE", "10-Year Breakeven", "market", "long", "T10YIE_value_zscore", "T10YIE_value_change_zscore"),
    MeasureRegistryEntry("T5YIFR", "5y5y Forward", "market", "long", "T5YIFR_value_zscore", "T5YIFR_value_change_zscore"),
    # --- model / synthetic ---
    MeasureRegistryEntry("EXPINF1YR", "Cleveland Fed 1Y", "model", "near", "EXPINF1YR_value_zscore", "EXPINF1YR_value_change_zscore"),
    MeasureRegistryEntry("EXPINF5YR", "Cleveland Fed 5Y", "model", "medium", "EXPINF5YR_value_zscore", "EXPINF5YR_value_change_zscore"),
    MeasureRegistryEntry("EXPINF10YR", "Cleveland Fed 10Y", "model", "long", "EXPINF10YR_value_zscore", "EXPINF10YR_value_change_zscore"),
    MeasureRegistryEntry("EXPINF30YR", "Cleveland Fed 30Y", "model", "long", "EXPINF30YR_value_zscore", "EXPINF30YR_value_change_zscore"),
    MeasureRegistryEntry("ATSIX_12m", "ATSIX 1Y", "model", "near", "ATSIX_12m_value_zscore", "ATSIX_12m_value_change_zscore"),
    MeasureRegistryEntry("ATSIX_60m", "ATSIX 5Y", "model", "medium", "ATSIX_60m_value_zscore", "ATSIX_60m_value_change_zscore"),
    MeasureRegistryEntry("ATSIX_120m", "ATSIX 10Y", "model", "long", "ATSIX_120m_value_zscore", "ATSIX_120m_value_change_zscore"),
    # --- household ---
    MeasureRegistryEntry("MICH", "Michigan 1Y", "household", "near", "MICH_value_zscore", "MICH_value_change_zscore"),
    MeasureRegistryEntry("SCE_1yr", "NY Fed SCE 1Y", "household", "near", "SCE_median_1yr_value_zscore", "SCE_median_1yr_value_change_zscore"),
    MeasureRegistryEntry("SCE_3yr", "NY Fed SCE 3Y", "household", "medium", "SCE_median_3yr_value_zscore", "SCE_median_3yr_value_change_zscore"),
    MeasureRegistryEntry("SCE_5yr", "NY Fed SCE 5Y", "household", "medium", "SCE_median_5yr_value_zscore", "SCE_median_5yr_value_change_zscore"),
    # --- professional ---
    MeasureRegistryEntry("SPF_CPI1YR", "SPF CPI 1Y", "professional", "near", "SPF_CPI1YR_value_zscore", "SPF_CPI1YR_value_change_zscore"),
    MeasureRegistryEntry("SPF_CPI5YR", "SPF CPI 5Y", "professional", "medium", "SPF_CPI5YR_value_zscore", "SPF_CPI5YR_value_change_zscore"),
    MeasureRegistryEntry("SPF_CPI10", "SPF CPI 10Y", "professional", "long", "SPF_CPI10_value_zscore", "SPF_CPI10_value_change_zscore"),
    MeasureRegistryEntry("SPF_PCE1YR", "SPF PCE 1Y", "professional", "near", "SPF_PCE1YR_value_zscore", "SPF_PCE1YR_value_change_zscore"),
    MeasureRegistryEntry("SPF_PCE5YR", "SPF PCE 5Y", "professional", "medium", "SPF_PCE5YR_value_zscore", "SPF_PCE5YR_value_change_zscore"),
    MeasureRegistryEntry("SPF_PCE10", "SPF PCE 10Y", "professional", "long", "SPF_PCE10_value_zscore", "SPF_PCE10_value_change_zscore"),
    MeasureRegistryEntry("LIVINGSTON", "Livingston CPI 10Y", "professional", "long", "LIVINGSTON_CPI10Y_value_zscore", "LIVINGSTON_CPI10Y_value_change_zscore"),
    # --- business ---
    MeasureRegistryEntry("ATL_BIE", "Atlanta Fed BIE 1Y", "business", "near", "ATL_BIE_value_zscore", "ATL_BIE_value_change_zscore"),
]

MEASURES_BY_ID = {m.measure_id: m for m in PRIMARY_MEASURES}

MACRO_CONTROL_COLUMNS = [
    "MACRO_CPI_HEADLINE_value",
    "MACRO_CPI_CORE_value",
    "MACRO_PCE_HEADLINE_value",
    "MACRO_PCE_CORE_value",
    "MACRO_UNRATE_value",
    "MACRO_PAYROLLS_value",
    "MACRO_REAL_GDP_value",
    "MACRO_TREASURY_2Y_value",
    "MACRO_TREASURY_10Y_value",
    "MACRO_FED_FUNDS_EFFECTIVE_value",
    "MACRO_OIL_WTI_value",
    "MACRO_FIN_CONDITIONS_NFCI_value",
]
