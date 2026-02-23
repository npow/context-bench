"""Result caching for resumable evaluation runs.

Saves per-row results to a JSONL file keyed by (system, dataset, example_id).
On resume, already-completed rows are loaded and skipped.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from context_bench.results import EvalRow


def _cache_key(system: str, dataset: str, example_id: str | int) -> str:
    """Deterministic cache key for a (system, dataset, example_id) triple."""
    return f"{system}::{dataset}::{example_id}"


def _row_to_dict(row: EvalRow) -> dict[str, Any]:
    """Serialize an EvalRow to a JSON-safe dict."""
    return {
        "system": row.system,
        "example_id": row.example_id,
        "dataset": row.dataset,
        "scores": row.scores,
        "input_tokens": row.input_tokens,
        "output_tokens": row.output_tokens,
        "latency": row.latency,
        "metadata": row.metadata,
    }


def _dict_to_row(d: dict[str, Any]) -> EvalRow:
    """Deserialize a dict back to an EvalRow."""
    return EvalRow(
        system=d["system"],
        example_id=d["example_id"],
        scores=d.get("scores", {}),
        input_tokens=d.get("input_tokens", 0),
        output_tokens=d.get("output_tokens", 0),
        metadata=d.get("metadata", {}),
        latency=d.get("latency", 0.0),
        dataset=d.get("dataset", ""),
    )


def _run_fingerprint(
    systems: list[str], datasets: list[str], evaluators: list[str],
) -> str:
    """Generate a short fingerprint for the run configuration."""
    key = json.dumps(
        {"systems": sorted(systems), "datasets": sorted(datasets),
         "evaluators": sorted(evaluators)},
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:12]


class ResultCache:
    """Persistent JSONL cache for evaluation rows.

    Args:
        cache_dir: Directory for cache files. A subdirectory named by run
            fingerprint is created automatically.
        systems: System names for this run.
        datasets: Dataset names for this run.
        evaluators: Evaluator names for this run.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        systems: list[str],
        datasets: list[str],
        evaluators: list[str],
    ) -> None:
        fp = _run_fingerprint(systems, datasets, evaluators)
        self._dir = Path(cache_dir) / fp
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "rows.jsonl"
        self._index: dict[str, EvalRow] = {}
        self._load()

    def _load(self) -> None:
        """Load existing cached rows from disk."""
        if not self._path.exists():
            return
        with self._path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    row = _dict_to_row(d)
                    key = _cache_key(row.system, row.dataset, row.example_id)
                    self._index[key] = row
                except (json.JSONDecodeError, KeyError):
                    continue

    def get(
        self, system: str, dataset: str, example_id: str | int,
    ) -> EvalRow | None:
        """Look up a cached row. Returns None on miss."""
        key = _cache_key(system, dataset, example_id)
        return self._index.get(key)

    def put(self, row: EvalRow) -> None:
        """Save a row to the cache (appends to JSONL file)."""
        key = _cache_key(row.system, row.dataset, row.example_id)
        self._index[key] = row
        with self._path.open("a") as f:
            f.write(json.dumps(_row_to_dict(row)) + "\n")

    @property
    def cached_count(self) -> int:
        """Number of cached rows."""
        return len(self._index)

    @property
    def cache_path(self) -> Path:
        """Path to the cache file."""
        return self._path
