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

## Agent 能力评估集

`routing_set.jsonl` 与 `multihop_set.jsonl` 专门验证 Agent 的决策能力，与上面的检索基线集互补。

### 路由集（25 条）

覆盖六类意图：chitchat / single_doc_qa / comparison / survey / memory / assist，
包含短句、长句和易混淆场景，主要验证「意图识别与路由」是否正确。

每条字段：`id`、`task=routing`、`question`、`expected_intent`、`should_retrieve`、`difficulty`、`notes`。

### 多跳集（25 条）

包含 22 条应回答的多跳问题（跨文档对比或链式推理）和 3 条无答案反例，
主要验证「查询分解 + 多步检索 + 综合 + 诚实拒答」。

每条字段：`id`、`task=multihop`、`question`、`expected_document_titles`、`sub_questions`、
`answer_points`、`should_answer`、`difficulty`、`notes`。

运行方式：

```powershell
python scripts/run_agent_eval.py
```

只跑路由准确率（不启动数据库）：

```powershell
python scripts/run_agent_eval.py --skip-multihop
```
