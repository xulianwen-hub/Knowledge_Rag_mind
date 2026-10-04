"""Streamlit 前端冒烟测试。"""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_smoke():
    app_path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    at = AppTest.from_file(str(app_path)).run(timeout=30)
    assert not at.exception
    assert at.header[0].value == "问答"
