# 知研 KnowResearch 演示脚本

## 演示目标

在 3～5 分钟内展示一条完整 Agent 链路：

```text
文档上传
  → 混合检索
  → 重排
  → 引用溯源
  → 多轮问答
  → 记忆审核
  → 技能管理
  → 运行状态
```

演示模式不追求检索准确率，优先保证完整流程可以稳定运行。

## 一、启动演示环境

```powershell
python scripts/demo_up.py
```

检查状态：

```powershell
python scripts/demo_status.py
```

停止：

```powershell
python scripts/demo_down.py
```

也可以手动分步启动：

```powershell
docker compose up -d
$env:DEMO_MODE="1"
python scripts/demo_seed.py
python -m uvicorn knowresearch.api.app:app --reload
python -m streamlit run frontend/app.py
```

打开：

```text
http://localhost:8501
```

## 二、演示流程

### 1. 介绍项目定位

一句话：

> KnowResearch 是一个面向研究生的科研知识库 Agent，支持多源文档解析、混合检索、引用溯源、三层记忆、技能沉淀和 MCP 工具扩展。

### 2. 展示知识库上传

进入“知识库”页面，说明：

- 支持 PDF / Word
- 文件 hash 去重
- 解析后进入文档存储、全文索引和向量索引

演示模式已自动准备三篇演示文档。

### 3. 展示问答

进入“问答”页面，提问：

```text
襟翼舵优化时为什么要同时考虑升力和阻力？
```

展示：

- 答案
- 引用编号
- 文档标题和页码
- 多轮会话

### 4. 展示记忆管理

进入“记忆管理”页面，展示：

- 候选画像
- 画像维度
- 置信度
- 证据
- 通过 / 拒绝操作

说明：LLM 只负责提案，正式记忆必须经过用户审核。

### 5. 展示技能管理

进入“技能管理”页面，展示：

- 候选技能
- 回放结果
- 审核通过
- 已启用技能
- 使用次数
- 成功率
- 点赞 / 点踩
- 禁用 / 回滚

### 6. 展示运行状态

打开：

```text
http://localhost:8000/health
http://localhost:8000/health/ready
http://localhost:8000/metrics
http://localhost:8000/tools
```

说明：

- `health`：服务是否启动
- `health/ready`：数据库、Redis、工具是否就绪
- `metrics`：请求数、延迟、token、失败状态
- `tools`：本地工具和 MCP 工具

## 三、录屏建议

建议录成 3～5 分钟视频：

1. 打开首页，介绍项目定位
2. 展示知识库页面
3. 提一个问题，展示答案和引用
4. 展示记忆审核
5. 展示技能管理
6. 展示 health / metrics

## 四、第一版演示边界

第一版演示不承诺：

- 高检索准确率
- 全量 62 篇文档的最佳排序
- 生产级高并发
- 完整 MCP 工具生态

第一版重点展示：

- Agent 主链路完整
- 记忆和技能旁路完整
- 工具层和 MCP 可插拔
- 可观测、可审核、可回滚
