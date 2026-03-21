"""Optimizer registry — single source of truth for all optimizer-specific behavior."""

from inspect import signature as inspect_signature

try:
    from dspy.teleprompt import (
        LabeledFewShot,
        BootstrapFewShot,
        BootstrapFewShotWithRandomSearch,
        COPRO,
        MIPROv2,
        SIMBA,
        GEPA,
    )

    _DSPY_AVAILABLE = True
except ImportError:
    _DSPY_AVAILABLE = False

if _DSPY_AVAILABLE:
    REGISTRY = {
        "LabeledFewShot": {
            "class": LabeledFewShot,
            "accepts_metric": False,
            "init_kwargs": {
                "light": {"k": 4},
                "medium": {"k": 8},
                "heavy": {"k": 16},
            },
            "build_compile_kwargs": lambda train, val, seed: {
                "trainset": train + val,
                "sample": True,
            },
            "metric_factory": "standard",
            "health_checks": ["demos_added"],
            "prompt_model": None,
            "cost_heuristic": lambda budget, n_train, n_val: 0,
        },
        "BootstrapFewShot": {
            "class": BootstrapFewShot,
            "accepts_metric": True,
            "init_kwargs": {
                "light": {"max_rounds": 1, "max_bootstrapped_demos": 2},
                "medium": {"max_rounds": 1, "max_bootstrapped_demos": 4},
                "heavy": {"max_rounds": 2, "max_bootstrapped_demos": 8},
            },
            "build_compile_kwargs": lambda train, val, seed: {
                "trainset": train + val,
            },
            "metric_factory": "standard",
            "health_checks": ["demos_added"],
            "prompt_model": None,
            "cost_heuristic": lambda budget, n_train, n_val: {
                "light": n_train + n_val,
                "medium": n_train + n_val,
                "heavy": 2 * (n_train + n_val),
            }[budget],
        },
        "BootstrapFewShotWithRandomSearch": {
            "class": BootstrapFewShotWithRandomSearch,
            "accepts_metric": True,
            "init_kwargs": {
                "light": {"num_candidate_programs": 4},
                "medium": {"num_candidate_programs": 8},
                "heavy": {"num_candidate_programs": 16},
            },
            "build_compile_kwargs": lambda train, val, seed: {
                "trainset": train,
                "valset": val,
            },
            "metric_factory": "standard",
            "health_checks": ["demos_added"],
            "prompt_model": None,
            "cost_heuristic": lambda budget, n_train, n_val: {
                "light": 7 * (n_train + n_val),
                "medium": 11 * (n_train + n_val),
                "heavy": 19 * (n_train + n_val),
            }[budget],
        },
        "COPRO": {
            "class": COPRO,
            "accepts_metric": True,
            "init_kwargs": {
                "light": {"breadth": 5, "depth": 2},
                "medium": {"breadth": 10, "depth": 3},
                "heavy": {"breadth": 15, "depth": 4},
            },
            "build_compile_kwargs": lambda train, val, seed: {
                "trainset": val,
                "eval_kwargs": {"num_threads": 1},
            },
            "metric_factory": "standard",
            "health_checks": ["instruction_changed"],
            "prompt_model": "use_prompt_model",
            "cost_heuristic": lambda budget, n_train, n_val: {
                "light": 5 * 2 * n_val + 10,
                "medium": 10 * 3 * n_val + 30,
                "heavy": 15 * 4 * n_val + 60,
            }[budget],
        },
        "MIPROv2": {
            "class": MIPROv2,
            "accepts_metric": True,
            "init_kwargs": {
                "light": {"auto": "light"},
                "medium": {"auto": "medium"},
                "heavy": {"auto": "heavy"},
            },
            "build_compile_kwargs": lambda train, val, seed: {
                "trainset": train,
                "valset": val,
                "seed": seed,
            },
            "metric_factory": "standard",
            "health_checks": ["instruction_changed", "demos_added"],
            "prompt_model": "use_prompt_model",
            "cost_heuristic": lambda budget, n_train, n_val: {
                "light": 6 * n_train + 11 * 35 + 3 * n_val + 15,
                "medium": 12 * n_train + 18 * 35 + 4 * n_val + 25,
                "heavy": 18 * n_train + 27 * 35 + 6 * n_val + 25,
            }[budget],
        },
        "SIMBA": {
            "class": SIMBA,
            "accepts_metric": True,
            "init_kwargs": {
                "light": {"max_steps": 4, "num_candidates": 4},
                "medium": {"max_steps": 8, "num_candidates": 6},
                "heavy": {"max_steps": 12, "num_candidates": 8},
            },
            "build_compile_kwargs": lambda train, val, seed: {
                "trainset": train + val,
                "seed": seed,
            },
            "metric_factory": "standard",
            "health_checks": ["instruction_changed", "demos_added"],
            "prompt_model": "use_prompt_model",
            "cost_heuristic": lambda budget, n_train, n_val: {
                "light": 4 * (32 * 4 + 5 * 32 + 16),
                "medium": 8 * (32 * 6 + 7 * 32 + 48),
                "heavy": 12 * (32 * 8 + 9 * 32 + 96),
            }[budget],
        },
        "GEPA": {
            "class": GEPA,
            "accepts_metric": True,
            "init_kwargs": {
                "light": {"auto": "light"},
                "medium": {"auto": "medium"},
                "heavy": {"auto": "heavy"},
            },
            "build_compile_kwargs": lambda train, val, seed: {
                "trainset": train,
                "valset": val,
            },
            "metric_factory": "gepa_feedback",
            "health_checks": ["instruction_changed"],
            "prompt_model": None,
            "cost_heuristic": lambda budget, n_train, n_val: {
                "light": 6 * n_val + 30 + n_val,
                "medium": 12 * n_val + 90 + n_val,
                "heavy": 18 * n_val + 90 + n_val,
            }[budget],
        },
        "GEPA_ablation": {
            "class": GEPA,
            "accepts_metric": True,
            "init_kwargs": {
                "light": {"auto": "light"},
                "medium": {"auto": "medium"},
                "heavy": {"auto": "heavy"},
            },
            "build_compile_kwargs": lambda train, val, seed: {
                "trainset": train,
                "valset": val,
            },
            "metric_factory": "standard",
            "health_checks": ["instruction_changed"],
            "prompt_model": None,
            "cost_heuristic": lambda budget, n_train, n_val: {
                "light": 6 * n_val + 30 + n_val,
                "medium": 12 * n_val + 90 + n_val,
                "heavy": 18 * n_val + 90 + n_val,
            }[budget],
        },
    }
else:
    REGISTRY = {}


def validate_registry():
    """Validate all registry entries. Raises ValueError on any invalid entry."""
    if not _DSPY_AVAILABLE:
        return

    REQUIRED_KEYS = {
        "class",
        "accepts_metric",
        "init_kwargs",
        "build_compile_kwargs",
        "metric_factory",
        "health_checks",
        "prompt_model",
        "cost_heuristic",
    }
    VALID_METRIC_FACTORIES = {"standard", "gepa_feedback"}
    VALID_HEALTH_CHECKS = {"demos_added", "instruction_changed"}

    for name, entry in REGISTRY.items():
        missing = REQUIRED_KEYS - set(entry.keys())
        if missing:
            raise ValueError(f"Registry error: {name} missing keys: {missing}")

        if entry["metric_factory"] not in VALID_METRIC_FACTORIES:
            raise ValueError(
                f"Registry error: {name} has unknown metric_factory "
                f"'{entry['metric_factory']}'. Valid: {VALID_METRIC_FACTORIES}"
            )

        invalid_checks = set(entry["health_checks"]) - VALID_HEALTH_CHECKS
        if invalid_checks:
            raise ValueError(
                f"Registry error: {name} has unknown health_checks: {invalid_checks}"
            )

        # Validate __init__ kwargs for each budget tier
        init_sig = inspect_signature(entry["class"].__init__)
        init_params = set(init_sig.parameters.keys()) - {"self"}
        has_var_keyword = any(
            p.kind == p.VAR_KEYWORD for p in init_sig.parameters.values()
        )
        for budget_tier, kwargs in entry["init_kwargs"].items():
            if not has_var_keyword:
                for kwarg_name in kwargs:
                    if kwarg_name not in init_params:
                        raise ValueError(
                            f"Registry error: {name}.__init__() does not accept "
                            f"'{kwarg_name}' (budget={budget_tier}). "
                            f"Accepted: {sorted(init_params)}"
                        )

        # Validate compile() kwargs
        compile_sig = inspect_signature(entry["class"].compile)
        dummy_kwargs = entry["build_compile_kwargs"]([], [], 42)
        has_var_keyword_compile = any(
            p.kind == p.VAR_KEYWORD for p in compile_sig.parameters.values()
        )
        for kwarg_name in dummy_kwargs:
            if not has_var_keyword_compile and kwarg_name not in compile_sig.parameters:
                raise ValueError(
                    f"Registry error: {name}.compile() does not accept '{kwarg_name}'. "
                    f"Accepted: {sorted(compile_sig.parameters.keys())}"
                )


if _DSPY_AVAILABLE:
    validate_registry()
