"""图像/表格转文本服务（方案一核心）。

- FigureVerbalizer：摄入时把图转成描述，带「结果缓存 + 有限重试」控制成本。
- TableTranscriber：表格结构识别失败时，把表格区域图片交给 VLM 转 Markdown 兜底。

两者都对 VLM 不可用 / 调用失败做静默降级，不阻断摄取。
"""

from __future__ import annotations

import hashlib
import os
import tempfile

from knowresearch.core.ports import VLMProvider


DEFAULT_FIGURE_PROMPT = (
    "你是科研论文配图说明生成器。请用中文简洁描述这张图，"
    "包括图的类型（示意图/曲线图/对比图/表格截图等）和关键信息，"
    "控制在 80 字以内，只输出描述本身。\n"
    "已知图题：{caption}\n"
    "图中上下文：{context}"
)

DEFAULT_TABLE_PROMPT = (
    "把这张表格图片转成 Markdown 表格，保留表头和所有行列数据，"
    "只输出 Markdown，不要输出任何解释。"
)


class _VlmTextService:
    """VLM 转文本的公共实现：缓存 + 有限重试 + 降级。"""

    def __init__(self, vlm: VLMProvider | None, prompt: str, retries: int = 1):
        self.vlm = vlm
        self.prompt = prompt
        self.retries = max(0, retries)
        self._cache: dict[str, str] = {}
        self.calls = 0

    def _run(self, image_bytes: bytes, prompt: str) -> str:
        if self.vlm is None or not image_bytes:
            return ""

        key = hashlib.sha1(image_bytes).hexdigest()
        if key in self._cache:
            return self._cache[key]

        result = ""
        for _ in range(self.retries + 1):
            try:
                result = self._call(image_bytes, prompt)
                self.calls += 1
                if result:
                    break
            except Exception:
                continue

        self._cache[key] = result
        return result

    def _call(self, image_bytes: bytes, prompt: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".png")
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(image_bytes)
            result = self.vlm.describe_image(path, prompt)
            return (result or "").strip()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


class FigureVerbalizer(_VlmTextService):
    """把图片区域转成文本描述。"""

    def __init__(
        self,
        vlm: VLMProvider | None = None,
        prompt: str = DEFAULT_FIGURE_PROMPT,
        retries: int = 1,
    ):
        super().__init__(vlm, prompt, retries)

    def describe(self, image_bytes: bytes, caption: str = "", context: str = "") -> str:
        prompt = self.prompt.format(
            caption=caption or "（无）", context=context or "（无）"
        )
        return self._run(image_bytes, prompt)


class TableTranscriber(_VlmTextService):
    """把表格区域图片转成 Markdown（表格结构识别失败时的兜底）。"""

    def __init__(
        self,
        vlm: VLMProvider | None = None,
        prompt: str = DEFAULT_TABLE_PROMPT,
        retries: int = 1,
    ):
        super().__init__(vlm, prompt, retries)

    def transcribe(self, image_bytes: bytes) -> str:
        return self._run(image_bytes, self.prompt)
