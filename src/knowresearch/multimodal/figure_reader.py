"""图解读器：把检索命中的图片交给 VLM，转成文字拼回生成上下文。

这是「方案一」的查询侧：命中 figure 块 → 从对象存储取原图 → VLM 解读
→ 把解读文本作为【图片解读】注入 prompt。任何一步失败都跳过该图，不影响主流程。
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from knowresearch.core.ports import ObjectStoragePort, VLMProvider


DEFAULT_READ_PROMPT = (
    "你是科研论文配图解读器。请结合下面的问题，用中文说明这张图与问题的关系"
    "以及图中的关键信息，控制在 120 字以内，只输出解读本身。\n问题：{question}"
)


class FigureReader:
    """把 figure 引用里的原图转成文字解读。"""

    def __init__(
        self,
        object_storage: ObjectStoragePort,
        vlm: VLMProvider,
        prompt: str = DEFAULT_READ_PROMPT,
        max_figures: int = 3,
    ) -> None:
        self.object_storage = object_storage
        self.vlm = vlm
        self.prompt = prompt
        self.max_figures = max_figures

    def read_citations(self, citations: list[Any], question: str = "") -> str:
        """读取 citations 中的 figure 引用，返回注入 prompt 的解读文本。"""
        lines: list[str] = []
        count = 0
        for cite in citations:
            if getattr(cite, "chunk_type", "") != "figure":
                continue
            image_key = getattr(cite, "image_key", "")
            if not image_key or count >= self.max_figures:
                continue
            try:
                image_bytes = self.object_storage.get(image_key)
            except Exception:
                continue
            try:
                description = self._describe(image_bytes, question)
            except Exception:
                continue
            if description:
                lines.append(f"- [图 {getattr(cite, 'index', '')}] {description}")
                count += 1
        return "\n".join(lines)

    def _describe(self, image_bytes: bytes, question: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".png")
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(image_bytes)
            prompt = self.prompt.format(question=question or "（无）")
            result = self.vlm.describe_image(path, prompt)
            return (result or "").strip()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
