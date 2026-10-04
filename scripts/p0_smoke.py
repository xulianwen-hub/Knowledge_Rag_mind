"""P0 冒烟验证：真实 Embedding / Reranker / DeepSeek 端到端问答。

为了不污染正式 SQLite 数据库，这里使用临时目录承载文档存储和对象存储；
内存向量索引和 BM25 索引只在本次进程内有效，足够验证 P0 闭环。
"""

from __future__ import annotations

import argparse
import shutil
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
from knowresearch.ingestion.pipeline import IngestSummary


SMOKE_QUESTIONS = [
    "襟翼舵对船舶操纵性有什么影响？",
    "物理信息神经网络在流体力学中主要用来解决什么问题？",
    "无人水下航行器的阻力优化通常考虑哪些方面？",
]


def _ingest_documents(processed_dir: Path, temp_dir: Path, limit: int | None = None):
    db_path = temp_dir / "p0.db"
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

    supported = sorted(
        p
        for p in Path(processed_dir).iterdir()
        if p.suffix.lower() in pipeline.SUPPORTED_EXTENSIONS
    )
    if limit is not None:
        supported = supported[:limit]

    summary = IngestSummary()
    for path in supported:
        result = pipeline.ingest_file(str(path))
        summary.total += 1
        if result.skipped:
            summary.skipped += 1
        elif result.error:
            summary.failed += 1
        else:
            summary.success += 1

    print(
        f"[ingest] total={summary.total} success={summary.success} "
        f"skipped={summary.skipped} failed={summary.failed}"
    )

    return embedding, vector_store, fulltext_store, document_store


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 端到端冒烟测试")
    parser.add_argument("--limit", type=int, default=1, help="只摄取前 N 个文件")
    args = parser.parse_args()

    processed_dir = resolve_project_path(settings.data.processed_dir)
    temp_dir = Path(tempfile.mkdtemp(prefix="knowresearch_p0_"))

    try:
        embedding, vector_store, fulltext_store, document_store = _ingest_documents(
            processed_dir, temp_dir, limit=args.limit
        )
        reranker = build_reranker()
        engine = build_qa_engine(
            vector_store=vector_store,
            fulltext_store=fulltext_store,
            document_store=document_store,
            embedding_provider=embedding,
            reranker_provider=reranker,
        )

        for question in SMOKE_QUESTIONS:
            print("\n" + "=" * 70)
            print(f"问题：{question}")
            answer = engine.answer(question)
            print(f"trace_id：{answer.trace_id}")
            print(f"答案：{answer.text}")
            print("引用：")
            for citation in answer.citations:
                location = citation.document_title
                if citation.page:
                    location += f" 第{citation.page}页"
                print(f"  [{citation.index}] {location}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
