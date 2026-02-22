"""Plugin registry for datasets, metrics, and reporters."""

from __future__ import annotations

from typing import Any, Callable


class Registry:
    """Simple name-based registry for pluggable components."""

    def __init__(self) -> None:
        self._stores: dict[str, dict[str, Any]] = {}

    def register(self, kind: str, name: str, obj: Any) -> None:
        """Register an object under kind/name."""
        if kind not in self._stores:
            self._stores[kind] = {}
        self._stores[kind][name] = obj

    def get(self, kind: str, name: str) -> Any:
        """Look up a registered object. Raises KeyError if not found."""
        try:
            return self._stores[kind][name]
        except KeyError:
            available = list(self._stores.get(kind, {}).keys())
            raise KeyError(
                f"No {kind} registered with name {name!r}. "
                f"Available: {available}"
            )

    def list(self, kind: str) -> list[str]:
        """List registered names for a kind."""
        return list(self._stores.get(kind, {}).keys())


# Global singleton
registry = Registry()


def register_dataset(name: str, loader: Callable[..., Any]) -> None:
    """Convenience: register a dataset loader."""
    registry.register("dataset", name, loader)


def load_dataset(name: str, **kwargs: Any) -> Any:
    """Convenience: load a registered dataset."""
    loader = registry.get("dataset", name)
    return loader(**kwargs)


def register_metric(name: str, metric: Any) -> None:
    """Convenience: register a metric."""
    registry.register("metric", name, metric)


def get_metric(name: str) -> Any:
    """Convenience: get a registered metric."""
    return registry.get("metric", name)
