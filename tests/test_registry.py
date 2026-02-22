"""Tests for the plugin registry."""

from __future__ import annotations

import pytest

from context_bench.registry import Registry, register_dataset, load_dataset


def test_register_and_get():
    reg = Registry()
    reg.register("dataset", "test_ds", lambda: [1, 2, 3])
    result = reg.get("dataset", "test_ds")
    assert result() == [1, 2, 3]


def test_get_missing_raises():
    reg = Registry()
    with pytest.raises(KeyError, match="No dataset registered"):
        reg.get("dataset", "nonexistent")


def test_list():
    reg = Registry()
    reg.register("metric", "m1", "a")
    reg.register("metric", "m2", "b")
    assert sorted(reg.list("metric")) == ["m1", "m2"]


def test_list_empty():
    reg = Registry()
    assert reg.list("nonexistent") == []


def test_convenience_functions():
    """register_dataset / load_dataset work with the global registry."""
    register_dataset("my_test", lambda n=None: list(range(n or 5)))
    assert load_dataset("my_test", n=3) == [0, 1, 2]
