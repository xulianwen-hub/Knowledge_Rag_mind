"""FullTextStorePort 适配器：jieba 分词 + BM25 检索。

首版用 rank-bm25 库实现 BM25Okapi，内存索引。
后续可换 PostgreSQL tsvector / ParadeDB。
"""

import jieba
from rank_bm25 import BM25Okapi

from knowresearch.core.ports import FullTextStorePort
from knowresearch.core.schemas import Record


def _tokenize(text: str) -> list[str]:
    """jieba 精确模式分词，过滤空白和单字噪音。"""
    tokens = [t.strip() for t in jieba.cut(text) if t.strip()]
    return tokens


class BM25FullTextStore(FullTextStorePort):
    """基于 jieba + BM25 的全文检索。"""

    def __init__(self):
        self._records: dict[str, Record] = {}
        self._corpus: list[list[str]] = []
        self._id_to_index: dict[str, int] = {}
        self._bm25: BM25Okapi | None = None
        self._dirty = False

    def _rebuild_index(self) -> None:
        """重建 BM25 索引（add/delete 后调用）。"""
        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)
        else:
            self._bm25 = None
        self._dirty = False

    def add(self, records: list[Record]) -> None:
        for record in records:
            tokens = _tokenize(record.content)
            if record.id in self._id_to_index:
                idx = self._id_to_index[record.id]
                self._corpus[idx] = tokens
                self._records[record.id] = record
            else:
                self._id_to_index[record.id] = len(self._corpus)
                self._corpus.append(tokens)
                self._records[record.id] = record
        self._dirty = True

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[Record, float]]:
        if self._dirty:
            self._rebuild_index()
        if self._bm25 is None or not self._corpus:
            return []

        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: -x[1])

        results: list[tuple[Record, float]] = []
        for idx, score in indexed_scores[:top_k]:
            if score < 0:
                continue
            record_id = list(self._id_to_index.keys())[
                list(self._id_to_index.values()).index(idx)
            ]
            results.append((self._records[record_id], float(score)))
        return results

    def delete(self, record_ids: list[str]) -> None:
        for record_id in record_ids:
            if record_id in self._id_to_index:
                idx = self._id_to_index.pop(record_id)
                self._corpus.pop(idx)
                self._records.pop(record_id, None)
                for rid, i in self._id_to_index.items():
                    if i > idx:
                        self._id_to_index[rid] = i - 1
        self._dirty = True

    def clear(self) -> None:
        self._records.clear()
        self._corpus.clear()
        self._id_to_index.clear()
        self._bm25 = None
        self._dirty = False