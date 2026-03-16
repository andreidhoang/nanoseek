"""
Quality classifier using FineWeb-Edu fastText model.

Downloads and caches the classifier on first use. Scores documents 0-5
where higher = more educational/high-quality content.

Reference: Lozhkov et al., "FineWeb-Edu" (HuggingFace, 2024)
"""

import os
import logging

from nanoseek.common import get_base_dir

logger = logging.getLogger(__name__)

# FineWeb-Edu classifier hosted on HuggingFace
_CLASSIFIER_URL = "https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier/resolve/main/model.bin"
_CLASSIFIER_FILENAME = "fineweb_edu_classifier.bin"
_model = None


def _get_classifier():
    """Load fastText classifier, downloading if needed."""
    global _model
    if _model is not None:
        return _model

    try:
        import fasttext
    except ImportError:
        raise ImportError(
            "fasttext is required for quality classification. "
            "Install with: pip install fasttext-wheel"
        )

    base_dir = get_base_dir()
    model_path = os.path.join(base_dir, _CLASSIFIER_FILENAME)

    if not os.path.exists(model_path):
        logger.info(f"Downloading FineWeb-Edu classifier to {model_path}...")
        import urllib.request
        from filelock import FileLock
        lock_path = model_path + ".lock"
        with FileLock(lock_path):
            if not os.path.exists(model_path):
                urllib.request.urlretrieve(_CLASSIFIER_URL, model_path)
                logger.info(f"Downloaded classifier ({os.path.getsize(model_path) / 1e9:.1f} GB)")

    # Suppress fasttext warnings about deprecated load_model
    _model = fasttext.load_model(model_path)
    logger.info("FineWeb-Edu classifier loaded")
    return _model


def classify_quality(text: str) -> float:
    """
    Score a document's quality using FineWeb-Edu fastText classifier.

    Args:
        text: Document text.

    Returns:
        Score in [0, 5] where higher = better quality.
    """
    model = _get_classifier()
    # fastText expects single-line input
    clean_text = text.replace('\n', ' ').strip()
    if not clean_text:
        return 0.0
    # Truncate to prevent OOM on very long docs (fastText handles full text but slowly)
    if len(clean_text) > 50000:
        clean_text = clean_text[:50000]
    labels, scores = model.predict(clean_text, k=1)
    # Label format: __label__X where X is the score (0-5)
    try:
        score = int(labels[0].replace('__label__', ''))
    except (ValueError, IndexError):
        score = 0
    return float(score)


def classify_batch(texts: list[str]) -> list[float]:
    """Score a batch of documents. Returns list of scores."""
    return [classify_quality(t) for t in texts]
