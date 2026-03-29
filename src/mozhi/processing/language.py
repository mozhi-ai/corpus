"""Language detection filter using fastText."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from mozhi.models import Document

logger = logging.getLogger(__name__)

# Tamil language label in fastText
_TAMIL_LABEL = "__label__ta"

# Cache the model globally to avoid reloading
_model = None


def _get_model():  # type: ignore[no-untyped-def]
    """Load the fastText language identification model (cached)."""
    global _model  # noqa: PLW0603
    if _model is not None:
        return _model

    import fasttext

    # Try standard locations for the model
    model_paths = [
        Path("corpus/models/lid.176.bin"),
        Path.home() / ".cache" / "mozhi" / "lid.176.bin",
    ]

    for path in model_paths:
        if path.exists():
            logger.info("Loading fastText model from %s", path)
            _model = fasttext.load_model(str(path))
            return _model

    # Download if not found
    download_path = model_paths[0]
    download_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading fastText language ID model...")

    import urllib.request

    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    urllib.request.urlretrieve(url, str(download_path))  # noqa: S310
    logger.info("Model saved to %s", download_path)

    _model = fasttext.load_model(str(download_path))
    return _model


def _predict_language(model: object, text: str) -> tuple[str, float]:
    """Predict language with numpy 2.x compatibility."""
    import numpy as np

    # fasttext's predict() uses np.array(probs, copy=False) which fails on numpy 2.x
    # Monkey-patch numpy temporarily to work around this
    _orig_array = np.array

    def _compat_array(*args: object, **kwargs: object) -> object:
        if "copy" in kwargs and kwargs["copy"] is False:
            kwargs["copy"] = None  # type: ignore[assignment]
        return _orig_array(*args, **kwargs)

    np.array = _compat_array  # type: ignore[assignment]
    try:
        predictions = model.predict(text)  # type: ignore[union-attr]
        label = predictions[0][0]
        score = float(predictions[1][0])
    finally:
        np.array = _orig_array  # type: ignore[assignment]

    return label, score


class LanguageFilter:
    """Filters documents to keep only Tamil text above a confidence threshold."""

    def __init__(self, min_score: float = 0.7) -> None:
        self.min_score = min_score
        self._filtered = 0

    @property
    def name(self) -> str:
        return "language_filter"

    def __call__(self, docs: Iterable[Document]) -> Iterator[Document]:
        model = _get_model()
        self._filtered = 0

        for doc in docs:
            # fastText expects single-line input
            text_oneline = doc.text.replace("\n", " ")[:1000]
            label, score = _predict_language(model, text_oneline)

            doc.language_score = score

            if label == _TAMIL_LABEL and score >= self.min_score:
                yield doc
            else:
                self._filtered += 1
                if self._filtered <= 10:
                    logger.debug(
                        "Filtered: lang=%s score=%.2f text=%s...",
                        label,
                        score,
                        doc.text[:50],
                    )

        logger.info("Language filter: removed %d non-Tamil documents", self._filtered)
