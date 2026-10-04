"""M3 Step 3+4 测试：Citation 引用溯源 + AnswerGenerator 答案生成。"""

from knowresearch.core.ports import DocumentStorePort, LLMProvider
from knowresearch.core.generation.answer_generator import (
    NO_ANSWER_TEXT,
    AnswerGenerator,
)
from knowresearch.core.generation.prompt import build_messages, build_user_prompt
from knowresearch.core.retrieval.citation import Citation, CitationBuilder
from knowresearch.core.schemas import Chunk, Document, SectionBlock


# ================================================================
# Mock 实现
# ================================================================


class _MockDocumentStore(DocumentStorePort):
    def __init__(self, documents=None, sections=None):
        self._documents = {d.id: d for d in (documents or [])}
        self._sections = sections or []

    def save_document(self, doc):
        self._documents[doc.id] = doc

    def get_document(self, doc_id):
        return self._documents.get(doc_id)

    def get_document_by_hash(self, file_hash):
        for d in self._documents.values():
            if d.file_hash == file_hash:
                return d
        return None

    def list_documents(self):
        return list(self._documents.values())

    def save_sections(self, sections):
        self._sections = sections

    def get_sections(self, doc_id):
        return [s for s in self._sections if s.document_id == doc_id]

    def delete_document(self, doc_id):
        self._documents.pop(doc_id, None)
        self._sections = [s for s in self._sections if s.document_id != doc_id]


class _MockLLM(LLMProvider):
    def __init__(self, response: str = ""):
        self._response = response
        self.last_messages = None

    def generate(self, prompt, system_prompt="", temperature=0.7, max_tokens=2048):
        return self._response

    def generate_with_messages(self, messages, temperature=0.7, max_tokens=2048):
        self.last_messages = messages
        return self._response


def _make_chunk(cid="c1", doc_id="d1", sec_id="s1", page=3, content="BERT 是一种预训练语言模型。"):
    return Chunk(
        id=cid,
        content=content,
        metadata={
            "document_id": doc_id,
            "section_id": sec_id,
            "page": page,
        },
    )


def _make_document(doc_id="d1", file_name="论文.pdf", total_pages=10):
    return Document(
        id=doc_id,
        metadata={"file_name": file_name, "total_pages": total_pages},
    )


def _make_section(sec_id="s1", doc_id="d1", title="引言", level=1, page_start=1, page_end=5):
    return SectionBlock(
        id=sec_id,
        metadata={
            "document_id": doc_id,
            "section_title": title,
            "section_level": level,
            "page_start": page_start,
            "page_end": page_end,
        },
    )


# ================================================================
# Step 3: Citation 引用溯源
# ================================================================


def test_citation_builder_basic():
    """基本引用溯源：chunk 能反查到文档名和章节标题。"""
    doc = _make_document()
    section = _make_section()
    chunk = _make_chunk()
    store = _MockDocumentStore(documents=[doc], sections=[section])

    builder = CitationBuilder(store)
    citations = builder.build([chunk])

    assert len(citations) == 1
    cite = citations[0]
    assert cite.document_title == "论文.pdf"
    assert cite.section_title == "引言"
    assert cite.page == 3
    assert cite.index == 1
    assert cite.content == chunk.content


def test_citation_builder_index_increment():
    """多个 chunk 时 index 从 1 递增。"""
    doc = _make_document()
    store = _MockDocumentStore(documents=[doc], sections=[])
    chunks = [
        _make_chunk(cid="c1"),
        _make_chunk(cid="c2"),
        _make_chunk(cid="c3"),
    ]

    builder = CitationBuilder(store)
    citations = builder.build(chunks)

    assert [c.index for c in citations] == [1, 2, 3]


def test_citation_builder_missing_doc():
    """chunk 的 document_id 查不到时，document_title 为空。"""
    chunk = _make_chunk(doc_id="nonexistent")
    store = _MockDocumentStore()

    builder = CitationBuilder(store)
    citations = builder.build([chunk])

    assert citations[0].document_title == ""
    assert citations[0].section_title == ""


def test_citation_builder_cache():
    """同一个文档被多个 chunk 引用时，只查询一次。"""
    doc = _make_document()
    section = _make_section()
    chunks = [_make_chunk(cid="c1"), _make_chunk(cid="c2")]
    store = _MockDocumentStore(documents=[doc], sections=[section])

    call_count = {"n": 0}
    original_get = store.get_document

    def counting_get(doc_id):
        call_count["n"] += 1
        return original_get(doc_id)

    store.get_document = counting_get

    builder = CitationBuilder(store)
    builder.build(chunks)

    assert call_count["n"] == 1


# ================================================================
# Step 4: Prompt 模板
# ================================================================


def test_build_user_prompt_contains_citations():
    """user prompt 包含参考资料编号和内容。"""
    citations = [
        Citation(
            chunk_id="c1",
            document_id="d1",
            document_title="论文.pdf",
            section_title="引言",
            page=3,
            content="BERT 是一种预训练语言模型。",
            index=1,
        )
    ]

    prompt = build_user_prompt("BERT 是什么？", citations)

    assert "[1]" in prompt
    assert "BERT 是一种预训练语言模型" in prompt
    assert "论文.pdf" in prompt
    assert "第3页" in prompt


def test_build_user_prompt_truncates_long_content():
    """超过 max_chunk_chars 的内容被截断。"""
    long_content = "x" * 1000
    citations = [
        Citation(
            chunk_id="c1",
            document_id="d1",
            document_title="论文.pdf",
            content=long_content,
            index=1,
        )
    ]

    prompt = build_user_prompt("q", citations, max_chunk_chars=100)

    assert "x" * 100 in prompt
    assert "x" * 101 not in prompt
    assert "..." in prompt


def test_build_messages_has_system_and_user():
    """messages 包含 system 和 user 两个角色。"""
    citations = [
        Citation(
            chunk_id="c1",
            document_id="d1",
            document_title="论文.pdf",
            content="内容",
            index=1,
        )
    ]

    messages = build_messages("问题", citations)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


# ================================================================
# Step 4: AnswerGenerator
# ================================================================


def test_answer_generator_empty_citations():
    """没有检索结果时返回"未找到相关内容"。"""
    llm = _MockLLM(response="不会被调用")
    generator = AnswerGenerator(llm)

    answer = generator.generate("问题", [], [])

    assert answer.text == NO_ANSWER_TEXT
    assert answer.citations == []


def test_answer_generator_returns_llm_response():
    """正常情况下返回 LLM 的回复。"""
    llm = _MockLLM(response="BERT 是一种预训练语言模型[1]。")
    generator = AnswerGenerator(llm)

    citations = [
        Citation(
            chunk_id="c1",
            document_id="d1",
            document_title="论文.pdf",
            content="BERT 是一种预训练语言模型。",
            index=1,
        )
    ]

    answer = generator.generate("BERT 是什么？", [], citations)

    assert answer.text == "BERT 是一种预训练语言模型[1]。"


def test_answer_generator_extracts_used_citations():
    """只返回答案中实际引用到的 citations。"""
    llm = _MockLLM(response="这是答案[1][3]。")
    generator = AnswerGenerator(llm)

    citations = [
        Citation(chunk_id="c1", document_id="d1", document_title="A", content="", index=1),
        Citation(chunk_id="c2", document_id="d1", document_title="B", content="", index=2),
        Citation(chunk_id="c3", document_id="d1", document_title="C", content="", index=3),
    ]

    answer = generator.generate("问题", [], citations)

    assert len(answer.citations) == 2
    assert {c.index for c in answer.citations} == {1, 3}


def test_answer_generator_no_citation_marks_returns_all():
    """答案中没有引用标记时，返回全部 citations（保守策略）。"""
    llm = _MockLLM(response="这是答案，没有引用标记。")
    generator = AnswerGenerator(llm)

    citations = [
        Citation(chunk_id="c1", document_id="d1", document_title="A", content="", index=1),
        Citation(chunk_id="c2", document_id="d1", document_title="B", content="", index=2),
    ]

    answer = generator.generate("问题", [], citations)

    assert len(answer.citations) == 2