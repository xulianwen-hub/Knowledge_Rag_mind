"""VectorStorePort 的 PostgreSQL + pgvector 实现。"""

from __future__ import annotations

import json
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, MetaData, String, Table, text

from knowresearch.adapters.postgres_base import (
    JSONType,
    create_postgres_engine,
)
from knowresearch.core.ports import VectorStorePort
from knowresearch.core.schemas import Record


class PostgresVectorStore(VectorStorePort):
    """基于 pgvector 的持久化向量存储。"""

    def __init__(
        self,
        database_url: str,
        dimension: int,
        table_name: str = "chunk_vectors",
    ):
        if dimension <= 0:
            raise ValueError("dimension 必须大于 0")
        self._dimension = dimension
        self._table_name = table_name
        self.engine = create_postgres_engine(database_url)
        self.metadata = MetaData()
        self.table = Table(
            table_name,
            self.metadata,
            Column("id", String, primary_key=True),
            Column("embedding", Vector(dimension), nullable=False),
            Column("data", JSONType, nullable=False),
        )
        self._ensure_schema()

    @property
    def dimension(self) -> int:
        return self._dimension

    def _ensure_schema(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        self.metadata.create_all(self.engine)

    def add(self, records: list[Record], embeddings: list[list[float]]) -> None:
        if len(records) != len(embeddings):
            raise ValueError("records 和 embeddings 数量必须一致")

        params = [
            {
                "id": record.id,
                "embedding": embedding,
                "data": json.dumps(record.model_dump(mode="json"), ensure_ascii=False),
            }
            for record, embedding in zip(records, embeddings)
        ]
        if not params:
            return

        stmt = text(
            f"""
            INSERT INTO {self._table_name} (id, embedding, data)
            VALUES (:id, :embedding, :data)
            ON CONFLICT (id) DO UPDATE
            SET embedding = EXCLUDED.embedding,
                data = EXCLUDED.data
            """
        )
        with self.engine.begin() as conn:
            conn.execute(stmt, params)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[Record, float]]:
        if top_k <= 0:
            return []

        stmt = text(
            f"""
            SELECT data,
                   1 - (embedding <=> CAST(:query AS vector)) AS score
            FROM {self._table_name}
            ORDER BY embedding <=> CAST(:query AS vector)
            LIMIT :limit
            """
        )
        with self.engine.connect() as conn:
            rows = conn.execute(
                stmt,
                {"query": query_embedding, "limit": top_k},
            ).mappings().all()

        return [
            (Record(**self._load_data(row["data"])), float(row["score"]))
            for row in rows
        ]

    def delete(self, record_ids: list[str]) -> None:
        if not record_ids:
            return
        with self.engine.begin() as conn:
            conn.execute(self.table.delete().where(self.table.c.id.in_(record_ids)))

    def clear(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(self.table.delete())

    def list_all(self) -> list[Record]:
        """列出全部向量记录，用于启动时重建 BM25 全文索引。"""
        stmt = text(
            f"SELECT data FROM {self._table_name}"
        )
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [Record(**self._load_data(row["data"])) for row in rows]

    @staticmethod
    def _load_data(data: Any) -> dict:
        if isinstance(data, str):
            return json.loads(data)
        return data
