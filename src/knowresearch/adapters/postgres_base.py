"""PostgreSQL 通用基础层。

文档、轨迹、技能等“Record 平表”共用这一套表结构和 CRUD。
使用 SQLAlchemy Core，不引入 ORM，保持简单。

测试兼容性：
- data 列使用 JSON().with_variant(JSONB, "postgresql")，
  在 PostgreSQL 下映射为 JSONB，在 SQLite 测试环境下降级为 JSON。
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    Index,
    MetaData,
    String,
    Table,
    create_engine,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB, insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from knowresearch.core.schemas import Record


JSONType = JSON().with_variant(JSONB, "postgresql")

RECORD_TABLE_NAMES = {"documents", "traces", "skills", "long_term_memory"}


def create_postgres_engine(database_url: str, echo: bool = False):
    """创建 SQLAlchemy engine。pool_pre_ping 用于应对连接失效。"""
    return create_engine(database_url, future=True, pool_pre_ping=True, echo=echo)


def build_record_table(name: str, metadata: MetaData) -> Table:
    """构建通用 Record 表对象。"""
    if name not in RECORD_TABLE_NAMES:
        raise ValueError(f"非法 Record 表名: {name}")

    table = Table(
        name,
        metadata,
        Column("id", String, primary_key=True),
        Column("kind", String, nullable=False),
        Column("source", String, nullable=False, default=""),
        Column("created_at", String, nullable=False),
        Column("updated_at", String, nullable=False),
        Column("data", JSONType, nullable=False),
    )
    Index(f"idx_{name}_kind", table.c.kind)
    Index(f"idx_{name}_source", table.c.source)
    return table


def record_to_row(record: Record) -> dict[str, Any]:
    """Record → 行字段。"""
    return {
        "id": record.id,
        "kind": record.kind,
        "source": record.source,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "data": json.dumps(record.model_dump(mode="json"), ensure_ascii=False),
    }


def row_to_record(row: Any) -> Record:
    """数据库行 → Record。"""
    data = row["data"]
    if isinstance(data, str):
        data = json.loads(data)
    return Record(**data)


def insert_record(engine, table: Table, record: Record) -> None:
    with engine.begin() as conn:
        conn.execute(table.insert().values(**record_to_row(record)))


def upsert_record(engine, table: Table, record: Record) -> None:
    values = record_to_row(record)
    if engine.dialect.name == "postgresql":
        stmt = postgresql_insert(table).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[table.c.id],
            set_=values,
        )
    else:
        stmt = sqlite_insert(table).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[table.c.id],
            set_=values,
        )
    with engine.begin() as conn:
        conn.execute(stmt)


def get_record(engine, table: Table, record_id: str) -> Record | None:
    with engine.connect() as conn:
        row = conn.execute(
            select(table).where(table.c.id == record_id)
        ).mappings().first()
    return row_to_record(row) if row else None


def delete_record(engine, table: Table, record_id: str) -> None:
    with engine.begin() as conn:
        conn.execute(table.delete().where(table.c.id == record_id))


def list_records(
    engine,
    table: Table,
    kind: str | None = None,
    source: str | None = None,
) -> list[Record]:
    stmt = select(table)
    if kind is not None:
        stmt = stmt.where(table.c.kind == kind)
    if source is not None:
        stmt = stmt.where(table.c.source == source)
    stmt = stmt.order_by(table.c.created_at.desc())

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [row_to_record(row) for row in rows]


def delete_records_by_ids(engine, table: Table, record_ids: list[str]) -> None:
    if not record_ids:
        return
    with engine.begin() as conn:
        conn.execute(table.delete().where(table.c.id.in_(record_ids)))
