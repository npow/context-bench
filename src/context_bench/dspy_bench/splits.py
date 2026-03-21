"""Dataset split manifest and conversion utilities for DSPy optimizer benchmarking.

Provides per-dataset split configuration, dataset-to-task-type mapping,
and conversion functions from context-bench dicts to dspy.Example objects.
"""

from __future__ import annotations

import random as _random
from typing import Any

import dspy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TRAIN_DEFAULT: int = 500
MAX_VAL_DEFAULT: int = 200
LABEL_FIELD: str = "answer"

# ---------------------------------------------------------------------------
# Dataset name normalization
# ---------------------------------------------------------------------------


def normalize_dataset_name(name: str) -> str:
    """Canonicalize dataset name: lowercase and replace hyphens with underscores."""
    return name.lower().replace("-", "_")


# ---------------------------------------------------------------------------
# Dataset-to-task-type mapping
# ---------------------------------------------------------------------------

DATASET_TASK_TYPE: dict[str, str] = {
    # qa
    "hotpotqa": "qa",
    "natural_questions": "qa",
    "musique": "qa",
    "narrativeqa": "qa",
    "triviaqa": "qa",
    "frames": "qa",
    "quality": "qa",
    "qasper": "qa",
    "truthfulqa": "qa",
    # math
    "gsm8k": "math",
    "math": "math",
    "mgsm": "math",
    "drop": "math",
    # mc
    "mmlu": "mc",
    "arc_challenge": "mc",
    "gpqa": "mc",
    "hellaswag": "mc",
    "winogrande": "mc",
    # summarization
    "multi_news": "summarization",
    "dialogsum": "summarization",
    "qmsum": "summarization",
    "summscreenfd": "summarization",
    "meetingbank": "summarization",
    "govreport": "summarization",
    # code
    "humaneval": "code",
    "mbpp": "code",
}

# ---------------------------------------------------------------------------
# Task input fields
# ---------------------------------------------------------------------------

TASK_INPUT_FIELDS: dict[str, list[str]] = {
    "qa": ["context", "question"],
    "math": ["question"],
    "mc": ["question", "choices"],
    "summarization": ["context", "question"],
    "code": ["context", "question"],
}

# ---------------------------------------------------------------------------
# Split manifest
# ---------------------------------------------------------------------------

SPLIT_MANIFEST: dict[str, dict[str, Any]] = {
    # train + validation (context-bench defaults to validation as eval)
    "hotpotqa": {"train_src": "train", "test_src": "validation", "val_from": "train_split"},
    "natural_questions": {"train_src": "train", "test_src": "validation", "val_from": "train_split"},
    "musique": {"train_src": "train", "test_src": "validation", "val_from": "train_split"},
    "qasper": {"train_src": "train", "test_src": "validation", "val_from": "train_split"},
    "quality": {"train_src": "train", "test_src": "validation", "val_from": "train_split"},
    "drop": {"train_src": "train", "test_src": "validation", "val_from": "train_split"},
    "qmsum": {"train_src": "train", "test_src": "validation", "val_from": "train_split"},
    "summscreenfd": {"train_src": "train", "test_src": "validation", "val_from": "train_split"},
    "hellaswag": {"train_src": "train", "test_src": "validation", "val_from": "train_split"},
    "winogrande": {"train_src": "train", "test_src": "validation", "val_from": "train_split"},
    # train + test (context-bench defaults to test as eval)
    "gsm8k": {"train_src": "train", "test_src": "test", "val_from": "train_split"},
    "math": {"train_src": "train", "test_src": "test", "val_from": "train_split"},
    "mgsm": {"train_src": "train", "test_src": "test", "val_from": "train_split"},
    "mbpp": {"train_src": "train", "test_src": "test", "val_from": "train_split"},
    # train + validation + test (all three HF splits available)
    "narrativeqa": {"train_src": "train", "test_src": "test", "val_from": "validation"},
    "triviaqa": {"train_src": "train", "test_src": "validation", "val_from": "train_split"},
    "arc_challenge": {"train_src": "train", "test_src": "test", "val_from": "validation"},
    "multi_news": {"train_src": "train", "test_src": "test", "val_from": "validation"},
    "dialogsum": {"train_src": "train", "test_src": "test", "val_from": "validation"},
    "meetingbank": {"train_src": "train", "test_src": "test", "val_from": "validation"},
    "govreport": {"train_src": "train", "test_src": "test", "val_from": "validation"},
    # MMLU: auxiliary_train, validation, test
    "mmlu": {"train_src": "auxiliary_train", "test_src": "test", "val_from": "train_split"},
    # SINGLE-SPLIT: zero-shot optimization only
    "truthfulqa": {"train_src": None, "test_src": "validation", "val_from": None, "zero_shot_only": True},
    "gpqa": {"train_src": None, "test_src": "train", "val_from": None, "zero_shot_only": True},
    "frames": {"train_src": None, "test_src": "test", "val_from": None, "zero_shot_only": True},
    "humaneval": {"train_src": None, "test_src": "test", "val_from": None, "zero_shot_only": True},
}

# ---------------------------------------------------------------------------
# Choice formatting
# ---------------------------------------------------------------------------


def format_choices(choices: list[str]) -> str:
    """Serialize a list of choices to 'A) ... B) ... C) ...' format."""
    letters = "ABCDEFGHIJ"
    return "  ".join(f"{letters[i]}) {c}" for i, c in enumerate(choices))


# ---------------------------------------------------------------------------
# Example conversion
# ---------------------------------------------------------------------------


def convert_to_dspy_example(cb_dict: dict, task_type: str) -> dspy.Example:
    """Convert a context-bench dict to a dspy.Example with correct input marking.

    The original dict is preserved as ``example._cb_original`` so evaluators
    can access the raw context-bench fields.

    For MC tasks, if ``choices`` is a list it is serialized to a string via
    :func:`format_choices`.
    """
    cb_dict_prepared = dict(cb_dict)
    if task_type == "mc" and isinstance(cb_dict.get("choices"), list):
        cb_dict_prepared["choices"] = format_choices(cb_dict["choices"])
    input_fields = TASK_INPUT_FIELDS[task_type]
    example = dspy.Example(**cb_dict_prepared).with_inputs(*input_fields)
    example._cb_original = cb_dict  # preserve original for evaluator
    return example


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------


def split_dataset(
    examples: list,
    train_frac: float,
    seed: int,
) -> tuple[list, list]:
    """Split a list into train and validation parts deterministically.

    Uses ``random.Random(seed).shuffle`` on a copy of *examples* and splits at
    ``int(len(examples) * train_frac)``.  The original list is not mutated.
    """
    data = list(examples)  # copy to avoid mutation
    rng = _random.Random(seed)
    rng.shuffle(data)
    n_train = int(len(data) * train_frac)
    return data[:n_train], data[n_train:]
