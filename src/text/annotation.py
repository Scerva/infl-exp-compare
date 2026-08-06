"""Human-annotation sample construction and scoring (spec section 7).

The dictionary (Model A), zero-shot NLI (Model B), and FinBERT-robustness
(Model C) stance approaches must ultimately be chosen by validation
performance against human judgment, not by convenience. This module
builds the stratified sentence sample for annotators and, once labels
come back, scores every model against them.

Stratification dimensions (spec section 7): decade, speaker, document
type, model-predicted stance bucket. We sample (approximately) evenly
across the cross of these so no single era/speaker/document type/stance
bucket dominates the human-labeled set -- a validation set that was 90%
Powell FOMC statements from 2023 would tell us little about whether the
model generalizes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from src.text.sentence_split import split_sentences
from src.text.stance import score_sentence_stance

ANNOTATION_COLUMNS = [
    "sentence_id", "document_id", "document_type", "speaker", "date", "decade",
    "sentence", "model_predicted_label", "model_stance_score",
    "human_label", "human_confidence",
]

STANCE_LABELS = ["HAWKISH", "NEUTRAL", "DOVISH"]


def _predicted_label(stance_score: float) -> str:
    if stance_score > 0:
        return "HAWKISH"
    if stance_score < 0:
        return "DOVISH"
    return "NEUTRAL"


def build_sentence_pool(documents_df: pd.DataFrame) -> pd.DataFrame:
    """Explode scored documents into policy-relevant sentence records
    (relevance_score > 0), each with the model's predicted stance.
    """
    rows = []
    for _, doc in documents_df.iterrows():
        text = doc.get("clean_text") or ""
        if not text:
            continue
        sentences = split_sentences(text)
        event_date = pd.to_datetime(doc["event_date"])
        decade = f"{(event_date.year // 10) * 10}s"
        for i, sent in enumerate(sentences):
            ss = score_sentence_stance(sent)
            if ss.relevance_score <= 0:
                continue
            rows.append(
                {
                    "sentence_id": f"{doc['document_id']}_s{i}",
                    "document_id": doc["document_id"],
                    "document_type": doc["document_type"],
                    "speaker": doc.get("speaker") or "FOMC",
                    "date": event_date,
                    "decade": decade,
                    "sentence": sent,
                    "model_predicted_label": _predicted_label(ss.stance),
                    "model_stance_score": ss.stance,
                }
            )
    return pd.DataFrame(rows)


def build_annotation_sample(
    documents_df: pd.DataFrame,
    n_target: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """Stratified sample of policy-relevant sentences for human annotation.

    Strata: decade x document_type x model_predicted_label. Speaker
    diversity is enforced as a secondary constraint (at most ~15% of the
    sample from any one speaker) rather than a full stratum, since many
    speakers have too few sentences to cross with every other dimension.
    """
    pool = build_sentence_pool(documents_df)
    if pool.empty:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)

    rng = np.random.default_rng(seed)
    pool["stratum"] = pool["decade"] + "_" + pool["document_type"] + "_" + pool["model_predicted_label"]
    strata = pool["stratum"].unique()
    per_stratum = max(1, n_target // max(len(strata), 1))

    speaker_cap = max(1, int(0.15 * n_target))
    speaker_counts: dict[str, int] = {}
    selected_idx = []

    for stratum in strata:
        stratum_df = pool[pool["stratum"] == stratum]
        idx = rng.permutation(stratum_df.index.to_numpy())
        taken = 0
        for i in idx:
            if taken >= per_stratum:
                break
            speaker = pool.loc[i, "speaker"]
            if speaker_counts.get(speaker, 0) >= speaker_cap:
                continue
            selected_idx.append(i)
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            taken += 1

    sample = pool.loc[selected_idx].copy()
    if len(sample) > n_target:
        sample = sample.sample(n=n_target, random_state=seed)
    elif len(sample) < n_target:
        remaining = pool.drop(index=sample.index)
        n_more = min(n_target - len(sample), len(remaining))
        if n_more > 0:
            extra = remaining.sample(n=n_more, random_state=seed)
            sample = pd.concat([sample, extra], ignore_index=False)

    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle order for annotators
    sample["human_label"] = ""
    sample["human_confidence"] = ""
    return sample[ANNOTATION_COLUMNS]


@dataclass
class ValidationReport:
    n_labeled: int
    accuracy: float
    macro_f1: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    confusion: pd.DataFrame
    spearman_rho: float
    spearman_pvalue: float


_LABEL_TO_NUMERIC = {"HAWKISH": 1, "NEUTRAL": 0, "DOVISH": -1}


def score_annotations(annotated_df: pd.DataFrame) -> ValidationReport:
    """Score model predictions against completed human labels.

    Expects ``human_label`` filled in with one of STANCE_LABELS for the
    rows an annotator completed; unlabeled rows (blank human_label) are
    dropped before scoring.
    """
    labeled = annotated_df[annotated_df["human_label"].isin(STANCE_LABELS)].copy()
    if labeled.empty:
        raise ValueError("No completed annotations found (human_label must be one of HAWKISH/NEUTRAL/DOVISH).")

    y_true = labeled["human_label"]
    y_pred = labeled["model_predicted_label"]

    accuracy = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, labels=STANCE_LABELS, average="macro", zero_division=0))
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=STANCE_LABELS, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=STANCE_LABELS)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in STANCE_LABELS], columns=[f"pred_{l}" for l in STANCE_LABELS])

    y_true_numeric = y_true.map(_LABEL_TO_NUMERIC)
    rho, pvalue = stats.spearmanr(y_true_numeric, labeled["model_stance_score"])

    return ValidationReport(
        n_labeled=len(labeled),
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_precision=dict(zip(STANCE_LABELS, precision)),
        per_class_recall=dict(zip(STANCE_LABELS, recall)),
        confusion=cm_df,
        spearman_rho=float(rho),
        spearman_pvalue=float(pvalue),
    )
