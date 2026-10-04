"""检查 Demo 服务状态。"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PID_FILE = ROOT / "data" / "demo_runtime" / "demo_processes.json"


def main() -> None:
    for name, url in (
        ("api", "http://127.0.0.1:8000/health"),
        ("frontend", "http://127.0.0.1:8501"),
    ):
        print(f"{name}: {'ok' if _http_ok(url) else 'down'} ({url})")
    if PID_FILE.exists():
        print("pids:", json.loads(PID_FILE.read_text(encoding="utf-8")))


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return 200 <= response.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


if __name__ == "__main__":
    main()
