"""Singleton sentence-transformers model, shared across dynamic pipeline loads.

The autoresearch loop loads context_pipeline.py into a fresh module each
iteration via importlib.  A module-level cache inside that dynamic module
would reset every load.  By storing the model here (in the installed package,
which Python caches in sys.modules), we pay the ~2 s load cost only once per
process, not once per iteration.
"""

from __future__ import annotations

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model():
    """Return the cached SentenceTransformer, loading it on first call."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: uv sync"
            ) from exc
        _model = SentenceTransformer(_MODEL_NAME)
    return _model
