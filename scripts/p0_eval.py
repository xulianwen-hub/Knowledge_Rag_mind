"""P0 轻量评估基线。

当前测试范围固定为 1 个文档，因此这里的“检索命中”和“引用准确率”
只能作为单文档阶段的基线，不能替代后续多文档评估集。

评估口径：
- 检索命中：top-k 引用片段中是否包含该问题预期出现的关键词。
- 引用准确率：返回的 citation 是否带出了正确的文档标题。
- 可回答率：是否给出了非兜底答案。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from knowresearch.adapters import (
    BM25FullTextStore,
    InMemoryVectorStore,
    LocalObjectStorage,
    SQLiteDocumentStore,
)
from knowresearch.bootstrap import (
    build_embedding,
    build_ingest_pipeline,
    build_qa_engine,
    build_reranker,
    resolve_project_path,
)
from knowresearch.config import settings
from knowresearch.core.generation.answer_generator import NO_ANSWER_TEXT
from knowresearch.core.qa_engine import SERVICE_UNAVAILABLE_TEXT


EVAL_CASES = [
    {
        "question": "Myring 型回转体阻力优化主要改变哪些参数？",
        "expected_keywords": ["a", "n", "θ"],
    },
    {
        "question": "阻力计算中为什么需要考虑摩擦阻力？",
        "expected_keywords": ["摩擦阻力"],
    },
    {
        "question": "多岛遗传算法在这个优化问题中的作用是什么？",
        "expected_keywords": ["多岛遗传算法"],
    },
]


def _build_engine(processed_dir: Path):
    temp_dir = Path(tempfile.mkdtemp(prefix="knowresearch_p0_eval_"))
    db_path = temp_dir / "eval.db"
    file_dir = temp_dir / "files"

    embedding = build_embedding()
    vector_store = InMemoryVectorStore(dimension=embedding.dimension)
    fulltext_store = BM25FullTextStore()
    document_store = SQLiteDocumentStore(str(db_path))
    object_storage = LocalObjectStorage(str(file_dir))

    pipeline = build_ingest_pipeline(
        document_store=document_store,
        vector_store=vector_store,
        fulltext_store=fulltext_store,
        object_storage=object_storage,
        embedding_provider=embedding,
    )

    files = sorted(
        p
        for p in processed_dir.iterdir()
        if p.suffix.lower() in pipeline.SUPPORTED_EXTENSIONS
    )
    target = files[0]
    result = pipeline.ingest_file(str(target))
    if result.error:
        raise RuntimeError(f"摄取失败: {result.error}")

    engine = build_qa_engine(
        vector_store=vector_store,
        fulltext_store=fulltext_store,
        document_store=document_store,
        embedding_provider=embedding,
        reranker_provider=build_reranker(),
    )
    return engine, target.name, temp_dir


def main() -> None:
    processed_dir = resolve_project_path(settings.data.processed_dir)
    engine, target_name, temp_dir = _build_engine(processed_dir)

    answered = 0
    retrieval_hits = 0
    correct_citations = 0
    total_citations = 0

    for case in EVAL_CASES:
        answer = engine.answer(case["question"])
        print("\n" + "=" * 70)
        print(f"问题：{case['question']}")
        print(f"答案是否兜底：{answer.text in (NO_ANSWER_TEXT, SERVICE_UNAVAILABLE_TEXT)}")

        citation_contents = [c.content for c in answer.citations]
        joined = "".join(citation_contents)
        hit = any(keyword in joined for keyword in case["expected_keywords"])
        print(f"关键词命中：{hit}")

        for citation in answer.citations:
            total_citations += 1
            if citation.document_title == target_name:
                correct_citations += 1

        if answer.text not in (NO_ANSWER_TEXT, SERVICE_UNAVAILABLE_TEXT):
            answered += 1
        if hit:
            retrieval_hits += 1

    print("\n" + "=" * 70)
    print(f"评估文档：{target_name}")
    print(f"问题总数：{len(EVAL_CASES)}")
    print(f"可回答率：{answered}/{len(EVAL_CASES)}")
    print(f"检索关键词命中：{retrieval_hits}/{len(EVAL_CASES)}")
    if total_citations:
        print(f"引用标题准确率：{correct_citations}/{total_citations}")
    else:
        print("引用标题准确率：0/0")

    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
