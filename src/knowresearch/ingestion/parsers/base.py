"""文档解析器端口与通用工具。

ParserPort：输入文件路径，输出 Document + list[SectionBlock]。
"""

from abc import ABC, abstractmethod

from knowresearch.core.schemas import Document, SectionBlock


class ParserPort(ABC):
    """文档解析端口。"""

    @abstractmethod
    def parse(self, file_path: str, doc_id: str = "") -> tuple[Document, list[SectionBlock]]:
        """解析文件。

        Args:
            file_path: 文件路径
            doc_id: 指定文档 ID，为空则自动生成

        Returns:
            (Document, list[SectionBlock])
        """