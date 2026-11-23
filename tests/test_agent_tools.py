"""新增 Agent 工具测试。"""

from knowresearch.core.schemas import Record
from knowresearch.tools.bibtex import BibTeXTool
from knowresearch.tools.memory_search import MemorySearchTool


class _FakeLongTermMemory:
    def __init__(self, records=None):
        self.records = records or []

    def search(self, query, top_k=5, user_id=None):
        return self.records[:top_k]


def test_memory_search_tool_returns_records():
    memory = _FakeLongTermMemory(
        [
            Record(
                id="m1",
                kind="reading",
                content="读过的 MoE 论文",
                metadata={"user_id": "u1"},
            )
        ]
    )
    tool = MemorySearchTool(memory)
    results = tool.run({"query": "MoE"})
    assert results[0]["memory_id"] == "m1"
    assert results[0]["kind"] == "reading"


def test_bibtex_tool_generates_entry():
    tool = BibTeXTool()
    output = tool.run(
        {
            "title": "Attention Is All You Need",
            "authors": "Vaswani, Ashish and Shazeer, Noam",
            "year": "2017",
            "venue": "NeurIPS",
        }
    )
    assert output.startswith("@inproceedings{")
    assert "Attention Is All You Need" in output
    assert "NeurIPS" in output
