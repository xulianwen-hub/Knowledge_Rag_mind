"""M7-1 评估集加载和校验测试。"""

import json
from pathlib import Path

import pytest

from knowresearch.core.schemas import EvalCase
from knowresearch.evaluation import (
    dataset_stats,
    load_eval_cases,
    validate_eval_cases,
)


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "data" / "eval" / "qa_set.jsonl"


def test_load_eval_set():
    cases = load_eval_cases(EVAL_PATH)
    assert len(cases) == 30
    assert len({case.id for case in cases}) == 30
    validate_eval_cases(cases)


def test_eval_set_stats():
    stats = dataset_stats(load_eval_cases(EVAL_PATH))
    assert stats["total"] == 30
    assert stats["answerable"] == 25
    assert stats["unanswerable"] == 5
    assert stats["by_category"]["comparison"] == 3
    assert stats["by_category"]["unanswerable"] == 5


def test_eval_expected_documents_exist():
    processed = {
        path.name
        for path in (ROOT / "data" / "processed").iterdir()
        if path.suffix.lower() in {".pdf", ".docx"}
    }
    for case in load_eval_cases(EVAL_PATH):
        for title in case.expected_document_titles:
            assert title in processed, f"{case.id} 引用了不存在的文档: {title}"


def test_invalid_case_raises():
    cases = [
        EvalCase(
            id="bad-1",
            question="问题",
            category="factual",
            should_answer=True,
            expected_document_titles=[],
        )
    ]
    with pytest.raises(ValueError):
        validate_eval_cases(cases)


def test_loader_reports_bad_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{bad json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_cases(path)
