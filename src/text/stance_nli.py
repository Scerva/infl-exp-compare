"""MODEL B: zero-shot NLI hawkish/dovish/neutral classification.

Uses a general-purpose natural-language-inference model in HuggingFace's
``zero-shot-classification`` pipeline mode: the sentence is the premise,
and each stance hypothesis below is tested as an entailment target. This
is a genuinely different modeling approach from Model A (src.text.stance,
literal phrase dictionary) -- it can pick up hawkish/dovish meaning
expressed without ever using one of the dictionary's exact phrases -- and
serves as the second of the three stance approaches the spec requires for
benchmarking before a primary method is chosen against human labels.

Model name is configurable (spec section 7: "Make model name/config
configurable"); the default is ``facebook/bart-large-mnli``, the
long-standing reference model for this exact HuggingFace pipeline mode.

COMPUTE NOTE: this is a ~1.6GB transformer run on CPU in this project's
environment (no GPU; see pyproject.toml's numpy/torch platform note).
Scoring the full ~1,984-document corpus sentence-by-sentence would take
hours. scripts/run_nli_benchmark.py therefore runs this model over the
same human-annotation sample (data/annotation/stance_annotation_sample.csv)
rather than the full corpus -- enough to benchmark Model B against Model A
and (eventually) human labels, without an overnight CPU job. Nothing
about the code below is corpus-size-limited; a GPU environment could run
it over everything.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_NLI_MODEL = "facebook/bart-large-mnli"

HYPOTHESES = {
    "hawkish": "This sentence advocates or implies tighter U.S. monetary policy.",
    "dovish": "This sentence advocates or implies easier U.S. monetary policy.",
    "neutral": "This sentence is neutral regarding monetary policy.",
}
_LABELS = list(HYPOTHESES.values())
_LABEL_TO_KEY = {v: k for k, v in HYPOTHESES.items()}


@lru_cache(maxsize=4)
def _load_pipeline(model_name: str):
    """Lazily load (and cache) a zero-shot-classification pipeline.

    Imports torch/transformers inside the function so the rest of the
    project never pays their import cost unless Model B is actually
    invoked.
    """
    from transformers import pipeline

    logger.info("Loading zero-shot NLI model %r (first call downloads if not cached)...", model_name)
    return pipeline("zero-shot-classification", model=model_name)


def score_sentence_nli(sentence: str, model_name: str = DEFAULT_NLI_MODEL, pipe=None) -> dict:
    """Score one sentence's hawkish/dovish/neutral probabilities via NLI.

    Returns {"p_hawkish", "p_dovish", "p_neutral", "stance"} where
    stance = p_hawkish - p_dovish, in [-1, 1], directly comparable to
    Model A's stance_sentence.
    """
    if pipe is None:
        pipe = _load_pipeline(model_name)
    result = pipe(sentence, candidate_labels=_LABELS, multi_label=False)
    probs = {_LABEL_TO_KEY[label]: score for label, score in zip(result["labels"], result["scores"])}
    p_hawk = probs.get("hawkish", 0.0)
    p_dove = probs.get("dovish", 0.0)
    p_neutral = probs.get("neutral", 0.0)
    return {
        "p_hawkish": p_hawk,
        "p_dovish": p_dove,
        "p_neutral": p_neutral,
        "stance": p_hawk - p_dove,
    }


def score_sentences_nli(sentences: list[str], model_name: str = DEFAULT_NLI_MODEL, batch_size: int = 8) -> list[dict]:
    """Batch-score a list of sentences. Returns a list aligned to input order."""
    pipe = _load_pipeline(model_name)
    out = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        results = pipe(batch, candidate_labels=_LABELS, multi_label=False)
        if isinstance(results, dict):
            results = [results]
        for result in results:
            probs = {_LABEL_TO_KEY[label]: score for label, score in zip(result["labels"], result["scores"])}
            p_hawk = probs.get("hawkish", 0.0)
            p_dove = probs.get("dovish", 0.0)
            p_neutral = probs.get("neutral", 0.0)
            out.append({"p_hawkish": p_hawk, "p_dovish": p_dove, "p_neutral": p_neutral, "stance": p_hawk - p_dove})
        logger.info("NLI scored %d/%d sentences", min(i + batch_size, len(sentences)), len(sentences))
    return out
