# 评估集

`qa_set.jsonl` 是 M7 的第一版基线评估集。

当前版本包含 30 条样本：

- 单文档事实与方法问题
- 综述类问题
- 多文档对比问题
- 无答案问题

每条样本包含：

- `id`
- `question`
- `category`
- `difficulty`
- `should_answer`
- `expected_document_titles`
- `expected_keywords`
- `history`
- `notes`

第一版以文档识别、检索命中和引用溯源为主，后续 M7-2 和 M7-3 会继续扩充更细的全文事实标签和答案质量标签。
