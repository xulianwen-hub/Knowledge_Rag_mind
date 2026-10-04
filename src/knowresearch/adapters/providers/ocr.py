"""OCRProvider 适配器：占位实现。

M1 阶段不装 PaddleOCR，用占位实现跑通接口。
遇到扫描件时再装 PaddleOCR 替换。
"""

from typing import Any

from knowresearch.core.ports import OCRProvider


class MockOCR(OCRProvider):
    """占位 OCR：返回空列表，等 PaddleOCR 装上后替换。"""

    def recognize(self, image_path: str) -> list[dict[str, Any]]:
        return []