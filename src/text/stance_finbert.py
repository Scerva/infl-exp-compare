"""MODEL C: FinBERT robustness benchmark.

FinBERT is a financial-domain sentiment model (positive/negative/neutral
TONE of financial text), not a monetary-policy stance model. The project
spec is explicit that this must be used only as a robustness benchmark,
never as a direct hawkish/dovish signal -- "financial positive/negative
tone is not the same as monetary-policy stance" (spec section 7, Model
C). A hawkish statement ("we will keep raising rates until inflation is
defeated") can read as financially NEGATIVE in tone even though it is
unambiguously hawkish, and a dovish statement ("the economy is
strengthening, supporting our decision to ease") can read as
POSITIVE. The comparison of FinBERT's positive-minus-negative score
against Model A/B's actual stance scores is therefore itself the
informative check: strong agreement would be a little suspicious
(suggesting the dictionary/NLI models are just recovering sentiment,
not stance); the expected result is a weak/noisy relationship.
"""
from __future__ import annotations

from functools import lru_cache

from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_FINBERT_MODEL = "ProsusAI/finbert"


@lru_cache(maxsize=4)
def _load_pipeline(model_name: str):
    from transformers import pipeline

    logger.info("Loading FinBERT model %r (first call downloads if not cached)...", model_name)
    return pipeline("text-classification", model=model_name, top_k=None)


def score_sentence_finbert(sentence: str, model_name: str = DEFAULT_FINBERT_MODEL, pipe=None) -> dict:
    """Return {"p_positive", "p_negative", "p_neutral", "finbert_tone"}
    where finbert_tone = p_positive - p_negative. This is financial TONE,
    explicitly not a hawkish/dovish stance measure -- see module docstring.
    """
    if pipe is None:
        pipe = _load_pipeline(model_name)
    result = pipe(sentence)
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        result = result[0]
    probs = {r["label"].lower(): r["score"] for r in result}
    p_pos = probs.get("positive", 0.0)
    p_neg = probs.get("negative", 0.0)
    p_neu = probs.get("neutral", 0.0)
    return {"p_positive": p_pos, "p_negative": p_neg, "p_neutral": p_neu, "finbert_tone": p_pos - p_neg}


def score_sentences_finbert(sentences: list[str], model_name: str = DEFAULT_FINBERT_MODEL, batch_size: int = 16) -> list[dict]:
    pipe = _load_pipeline(model_name)
    out = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        results = pipe(batch)
        for result in results:
            probs = {r["label"].lower(): r["score"] for r in result}
            p_pos = probs.get("positive", 0.0)
            p_neg = probs.get("negative", 0.0)
            p_neu = probs.get("neutral", 0.0)
            out.append({"p_positive": p_pos, "p_negative": p_neg, "p_neutral": p_neu, "finbert_tone": p_pos - p_neg})
        logger.info("FinBERT scored %d/%d sentences", min(i + batch_size, len(sentences)), len(sentences))
    return out
