"""Demo 启动 / 停止脚本辅助函数测试。"""

from types import SimpleNamespace

import scripts.demo_up as demo_up


def test_demo_env_sets_demo_mode():
    env = demo_up._demo_env()
    assert env["DEMO_MODE"] == "1"
    assert env["PYTHONUTF8"] == "1"


def test_pid_file_roundtrip(tmp_path, monkeypatch):
    pid_file = tmp_path / "pids.json"
    monkeypatch.setattr(demo_up, "PID_FILE", pid_file)
    demo_up._write_pid_file(
        {
            "api": SimpleNamespace(pid=111),
            "frontend": SimpleNamespace(pid=222),
        }
    )
    pids = demo_up._read_pid_file()
    assert pids == {"api": 111, "frontend": 222}
