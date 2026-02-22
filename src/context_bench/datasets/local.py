"""Load datasets from local JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(
    path: str | Path,
    n: int | None = None,
) -> list[dict[str, Any]]:
    """Load examples from a JSONL file.

    Each line must be valid JSON with at least "id" and "context" fields.

    Args:
        path: Path to the JSONL file.
        n: Maximum number of examples to load.

    Returns:
        List of example dicts.
    """
    path = Path(path)
    examples = []

    with path.open() as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            # Ensure id exists
            if "id" not in example:
                example["id"] = i
            examples.append(example)

    return examples
