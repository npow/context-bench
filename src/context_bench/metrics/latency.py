"""Latency metrics — per-example wall-clock timing statistics."""

from __future__ import annotations

from dataclasses import dataclass

from context_bench.results import EvalRow


@dataclass
class Latency:
    """Per-example latency statistics: mean, median, p95, p99.

    Reads the ``latency`` field on each EvalRow (seconds).
    """

    @property
    def name(self) -> str:
        return "latency"

    def compute(self, rows: list[EvalRow]) -> dict[str, float]:
        if not rows:
            return {
                "latency_mean": 0.0,
                "latency_median": 0.0,
                "latency_p95": 0.0,
                "latency_p99": 0.0,
            }

        latencies = sorted(r.latency for r in rows)
        n = len(latencies)
        mean = sum(latencies) / n
        median = latencies[n // 2] if n % 2 else (latencies[n // 2 - 1] + latencies[n // 2]) / 2
        p95 = latencies[int(n * 0.95)] if n > 1 else latencies[0]
        p99 = latencies[int(n * 0.99)] if n > 1 else latencies[0]

        return {
            "latency_mean": mean,
            "latency_median": median,
            "latency_p95": p95,
            "latency_p99": p99,
        }
