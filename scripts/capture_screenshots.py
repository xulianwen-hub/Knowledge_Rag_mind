"""用 Edge CDP 截取 README 展示图片。"""

from __future__ import annotations

import base64
import json
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect


ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "docs" / "assets"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

SHOTS = [
    ("http://127.0.0.1:8501/?page=chat&demo=1", "01-demo-chat.png", 10000),
    ("http://127.0.0.1:8501/?page=memory", "02-memory.png", 8000),
    ("http://127.0.0.1:8501/?page=skills", "03-skills.png", 8000),
    ("http://127.0.0.1:8501/?page=knowledge", "04-knowledge.png", 8000),
    ("http://127.0.0.1:8000/docs", "05-api-docs.png", 3000),
]


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    user_data_dir = ROOT / "data" / "demo_runtime" / "edge-profile"
    process = subprocess.Popen(
        [
            str(EDGE),
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={user_data_dir}",
            "--window-size=1440,1000",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_cdp(port)
        websocket_url = _page_websocket_url(port)
        with connect(websocket_url, max_size=None) as ws:
            _send(ws, "Page.enable")
            _send(
                ws,
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": 1440,
                    "height": 1000,
                    "deviceScaleFactor": 1,
                    "mobile": False,
                },
            )
            for url, filename, wait_ms in SHOTS:
                print(f"capture: {url}")
                _send(ws, "Page.navigate", {"url": url})
                time.sleep(wait_ms / 1000)
                result = _send(
                    ws,
                    "Page.captureScreenshot",
                    {"format": "png", "captureBeyondViewport": True},
                )
                (ASSETS_DIR / filename).write_bytes(
                    base64.b64decode(result["data"])
                )
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _send(ws, method: str, params: dict | None = None) -> dict:
    request_id = int(time.time() * 1000000) % 2147483647
    ws.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
    while True:
        message = json.loads(ws.recv())
        if message.get("id") == request_id:
            if "error" in message:
                raise RuntimeError(message["error"])
            return message.get("result", {})


def _wait_for_cdp(port: int) -> None:
    deadline = time.time() + 20
    url = f"http://127.0.0.1:{port}/json/version"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.3)
    raise TimeoutError("Edge CDP 未就绪")


def _page_websocket_url(port: int) -> str:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/json/list",
        timeout=3,
    ) as response:
        targets = json.loads(response.read().decode("utf-8"))
    for target in targets:
        if target.get("type") == "page":
            return target["webSocketDebuggerUrl"]
    raise RuntimeError("没有找到 Edge page target")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


if __name__ == "__main__":
    main()
