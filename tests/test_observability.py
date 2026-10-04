"""M7-7 指标与错误分类测试。"""

from knowresearch.core.observability import MetricsRegistry, classify_error


def test_metrics_registry_snapshot():
    registry = MetricsRegistry()
    registry.increment("requests")
    registry.increment("requests", 2)
    registry.observe("latency_ms", 10)
    registry.observe("latency_ms", 30)

    snapshot = registry.snapshot()
    assert snapshot["counters"]["requests"] == 3
    assert snapshot["histograms"]["latency_ms"]["avg"] == 20
    assert snapshot["histograms"]["latency_ms"]["max"] == 30


def test_classify_error():
    assert classify_error(TimeoutError("timed out")) == "timeout"
    assert classify_error(ConnectionError("connection reset")) == "connection"
    assert classify_error(PermissionError("forbidden")) == "permission"
    assert classify_error(ValueError("其他错误")) == "unknown"
