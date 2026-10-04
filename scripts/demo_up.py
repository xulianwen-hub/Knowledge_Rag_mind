"""一键启动 Demo：Docker + 演示数据 + FastAPI + Streamlit。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "data" / "demo_runtime"
LOG_DIR = RUNTIME_DIR / "logs"
PID_FILE = RUNTIME_DIR / "demo_processes.json"
API_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://127.0.0.1:8501"


def main() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] 启动 PostgreSQL 和 Redis...")
    _run(["docker", "compose", "up", "-d"], check=True)

    print("[2/4] 准备演示数据...")
    env = _demo_env()
    _run([sys.executable, str(ROOT / "scripts" / "demo_seed.py")], env=env, check=True)

    processes = {}
    try:
        print("[3/4] 启动 FastAPI...")
        processes["api"] = _start_process(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "knowresearch.api.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            LOG_DIR / "api.log",
            env,
        )
        if not _wait_http(f"{API_URL}/health", timeout_seconds=60):
            raise RuntimeError(f"FastAPI 未就绪，请查看 {LOG_DIR / 'api.log'}")

        print("[4/4] 启动 Streamlit...")
        processes["frontend"] = _start_process(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "frontend/app.py",
                "--server.headless",
                "true",
                "--server.port",
                "8501",
            ],
            LOG_DIR / "frontend.log",
            env,
        )
        if not _wait_http(FRONTEND_URL, timeout_seconds=60):
            raise RuntimeError(
                f"Streamlit 未就绪，请查看 {LOG_DIR / 'frontend.log'}"
            )
    except Exception:
        _terminate_all(processes)
        raise

    _write_pid_file(processes)
    print("\nDemo 已启动：")
    print(f"  API:      {API_URL}")
    print(f"  Frontend: {FRONTEND_URL}")
    print(f"  日志目录: {LOG_DIR}")
    print("\n停止 Demo：")
    print("  python scripts/demo_down.py")


def _demo_env() -> dict[str, str]:
    env = os.environ.copy()
    env["DEMO_MODE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(args: list[str], env: dict[str, str] | None = None, check: bool = True):
    return subprocess.run(
        args,
        cwd=str(ROOT),
        env=env,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _start_process(args: list[str], log_path: Path, env: dict[str, str]):
    log_file = log_path.open("w", encoding="utf-8")
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(
        args,
        cwd=str(ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )


def _wait_http(url: str, timeout_seconds: int) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _http_ok(url):
            return True
        time.sleep(0.5)
    return False


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return 200 <= response.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _write_pid_file(processes: dict) -> None:
    PID_FILE.write_text(
        json.dumps(
            {
                "api": processes["api"].pid,
                "frontend": processes["frontend"].pid,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_pid_file() -> dict:
    if not PID_FILE.exists():
        return {}
    return json.loads(PID_FILE.read_text(encoding="utf-8"))


def _terminate_all(processes: dict) -> None:
    for process in processes.values():
        try:
            process.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
