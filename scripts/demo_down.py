"""停止 Demo 进程。"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PID_FILE = ROOT / "data" / "demo_runtime" / "demo_processes.json"


def main() -> None:
    if not PID_FILE.exists():
        print("没有找到 Demo PID 文件。")
        return
    pids = json.loads(PID_FILE.read_text(encoding="utf-8"))
    for name, pid in pids.items():
        _terminate_pid(pid)
        print(f"stopped: {name} ({pid})")
    PID_FILE.unlink(missing_ok=True)


def _terminate_pid(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


if __name__ == "__main__":
    main()
