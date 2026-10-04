"""轻量指标注册表。"""

from __future__ import annotations

from collections import defaultdict


class MetricsRegistry:
    def __init__(self):
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)

    def increment(self, name: str, value: float = 1.0) -> None:
        self._counters[name] += value

    def observe(self, name: str, value: float) -> None:
        self._histograms[name].append(float(value))

    def snapshot(self) -> dict:
        return {
            "counters": dict(self._counters),
            "histograms": {
                name: {
                    "count": len(values),
                    "avg": round(sum(values) / len(values), 2),
                    "min": round(min(values), 2),
                    "max": round(max(values), 2),
                }
                for name, values in self._histograms.items()
                if values
            },
        }

    def reset(self) -> None:
        self._counters.clear()
        self._histograms.clear()


metrics = MetricsRegistry()


def classify_error(error: Exception) -> str:
    text = str(error).lower()
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "connection" in text or "connect" in text:
        return "connection"
    if "permission" in text or "forbidden" in text:
        return "permission"
    if "not found" in text or "不存在" in text:
        return "not_found"
    return "unknown"
