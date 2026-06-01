# 实验设计

## 实验目标

通过对比不同检索策略，验证 GraphRAG 和 GNN 节点表示增强在复杂关联问答中的效果。

## 数据集设计

计划构建一个小规模科研知识问答数据集。

数据来源：

- AI / NLP / GNN / RAG 相关论文摘要
- 技术博客
- 开源项目文档

问题类型：

- 单跳事实型问题
- 跨文档关联问题
- 多跳逻辑推理问题
- 方法对比总结问题

## 对比方法

### Vector-only RAG

仅使用 Milvus 向量检索文本块，并将召回上下文提供给 LLM。

### GraphRAG

融合 Milvus 向量检索和 Neo4j 图谱遍历结果。

### GraphRAG + GNN

在 GraphRAG 基础上，使用 GAT 增强后的节点向量辅助长尾实体召回。

### GraphRAG + GNN + Rerank

在候选上下文合并后，使用 Reranker 对候选证据进行重排序。

当前工程已接入轻量关键词 overlap reranker，用于先打通 `Hybrid Evidence -> Rerank -> QA Prompt -> Citations` 链路。BGE Reranker 或 cross-encoder reranker 将在后续替换或扩展该模块。

## 评估指标

### Recall@K

衡量正确证据是否出现在 TopK 检索结果中。

### MRR

衡量正确证据在检索结果中的排名质量。

### Answer Accuracy

衡量最终回答是否正确。

### Faithfulness

衡量回答是否能被检索到的证据支持。

### Latency

衡量端到端响应延迟。

## 评估记录脚本

当前提供 `scripts/evaluate_retrieval.py`，用于批量调用 `/retrieval/debug` 和 `/qa/ask`，把检索结果、答案、citations 和自动评估指标写入 JSONL。

样例问题集用于快速 smoke test：

```text
data/eval/questions.sample.jsonl
```

20 条小规模 dev set 用于当前阶段的指标验证和配置对比：

```text
data/eval/questions.dev.jsonl
```

运行方式：

```powershell
python scripts/evaluate_retrieval.py --vector-top-k 3 --graph-top-k 5 --graph-max-depth 2 --qa-top-k 3
```

使用 dev set：

```powershell
python scripts/evaluate_retrieval.py --input-file data/eval/questions.dev.jsonl --vector-top-k 3 --graph-top-k 5 --graph-max-depth 2 --qa-top-k 3
```

默认输出：

```text
data/eval/retrieval_eval_results.jsonl
data/eval/retrieval_eval_summary.json
```

`retrieval_eval_results.jsonl` 每条记录包含：

- `run_config`：本次评估使用的 API 地址、TopK 参数和实际 fusion 权重
- `retrieval_debug`：向量、图谱和混合检索结果
- `qa`：问答结果和 `citations`
- `metrics`：基于期望关键词的检索、引用、答案和延迟指标
- `summary`：召回数量、引用数量和 Top hybrid evidence 摘要

`retrieval_eval_summary.json` 汇总本次运行的整体指标，包含题目数、运行配置、平均 evidence keyword recall、Recall@K、MRR、Top1 evidence hit rate、citation hit rate、answer hit rate 和平均延迟，用于快速比较不同检索配置。

问题集中的每条记录可包含：

- `expected_evidence_keywords`：期望在检索证据中命中的关键词
- `expected_answer_keywords`：期望在答案中命中的关键词；如果省略，则复用 `expected_evidence_keywords`

当前自动指标包括：

- `metrics.retrieval.evidence_keyword_recall`：混合检索结果覆盖期望证据关键词的比例
- `metrics.retrieval.retrieval_hit`：TopK 混合检索结果中是否至少命中一个期望证据关键词
- `metrics.retrieval.first_relevant_rank`：第一个关键词命中的混合证据 rank
- `metrics.retrieval.mrr`：基于 `first_relevant_rank` 计算的 reciprocal rank
- `metrics.retrieval.top_hybrid_keyword_hit`：Top1 混合证据是否命中期望证据关键词
- `metrics.citations.citation_keyword_hit`：答案 citations 指向的证据是否命中期望证据关键词
- `metrics.answer.answer_keyword_recall`：答案覆盖期望答案关键词的比例
- `metrics.answer.answer_keyword_hit`：答案是否至少命中一个期望答案关键词
- `metrics.latency.total_ms`：`/retrieval/debug` 和 `/qa/ask` 两次调用的总耗时

聚合摘要中的指标由逐题 `metrics` 平均得到；缺失或不适用的指标会被跳过，整组都不可用时返回 `null`。

这些指标是轻量关键词指标，适合当前小规模 dev set 的链路验证和配置对比；正式实验仍需要更大问题集和人工或模型辅助评测。当前 `questions.dev.jsonl` 包含 20 条问题，基于 `data/raw/sample.txt`、`chunks.jsonl` 和 `graph_triples.jsonl` 构造，重点覆盖信息碎片、vector-only retrieval、knowledge graph、graph traversal、GNN、GraphRAG 关系、证据类型、图节点类型和长尾实体等概念。

评估不同融合策略时，可以在 `.env` 中调整：

```env
FUSION_SCORE_WEIGHT=0.7
FUSION_RANK_WEIGHT=0.3
```

`FUSION_SCORE_WEIGHT` 越高，排序越依赖原始相关性分数；`FUSION_RANK_WEIGHT` 越高，排序越依赖检索 rank。

评估输出会从 `/retrieval/debug` 的 `fusion_weights` 读取实际生效权重并写入 `run_config`，方便对比不同 `.env` 配置下的结果。

## 当前实测记录

### 2026-06-01 GraphRAG MVP dev set

本次实测用于验证当前 GraphRAG MVP 链路是否可以完成从样例文档入库、向量检索、图谱检索、混合证据融合到问答评估的可复现闭环。

数据与配置：

- 原始文档：`data/raw/sample.txt`
- 问题集：`data/eval/questions.dev.jsonl`
- 题目数量：20
- 文本切分：`chunk_size=700`，`chunk_overlap=100`
- 检索参数：`vector_top_k=3`，`graph_top_k=5`，`graph_max_depth=2`，`qa_top_k=3`
- 融合权重：`score_weight=0.7`，`rank_weight=0.3`
- 输出文件：`data/eval/retrieval_eval_results.jsonl`、`data/eval/retrieval_eval_summary.json`

运行命令：

```powershell
python scripts/evaluate_retrieval.py --input-file data/eval/questions.dev.jsonl --vector-top-k 3 --graph-top-k 5 --graph-max-depth 2 --qa-top-k 3 --timeout 180
```

实测结果：

| 方法 | 问题数 | Evidence keyword recall | Recall@K | MRR | Top1 evidence hit rate | Citation hit rate | Answer keyword recall | Answer hit rate | 平均总延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GraphRAG MVP | 20 | 0.9708 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8917 | 1.0000 | 31190.8465 ms |

延迟拆分：

| 指标 | 平均耗时 |
|---|---:|
| `/retrieval/debug` | 14507.5994 ms |
| `/qa/ask` | 16683.2471 ms |

逐题观察：

- `q0012` 和 `q0019` 的检索证据关键词覆盖未达到 1.0，但 `first_relevant_rank` 仍为 1，说明 Top1 证据已经命中核心信息。
- `q0012`、`q0015`、`q0019` 和 `q0020` 的答案关键词覆盖未达到 1.0，但均至少命中一个期望答案关键词。
- citation keyword hit rate 为 1.0，说明当前 citations 能稳定指向包含期望关键词的混合证据。

结论与限制：

- 当前 dev set 能稳定验证 GraphRAG MVP 的端到端可用性，适合作为本地 smoke test 和项目展示基线。
- 本次数据集规模较小，且 `questions.dev.jsonl` 与 `sample.txt` 明确对齐，关键词指标会偏乐观，不能作为最终实验结论。
- 20 题扩展后，答案关键词覆盖不再全满分，更接近真实评估现象；后续可以继续优化 prompt、证据压缩和关键词设计。
- 当前平均延迟偏高，主要用于记录现状；性能优化应在评估闭环和对比实验稳定后单独推进。
- 后续需要在更稳定的数据集上补充 Vector-only、GraphRAG + Rerank、GraphRAG + GNN 等配置对比。

## 结果记录模板

| 方法 | 多跳准确率 | Recall@5 | MRR | 平均延迟 |
|---|---:|---:|---:|---:|
| Vector-only RAG | 待实验 | 待实验 | 待实验 | 待实验 |
| GraphRAG | 待实验 | 待实验 | 待实验 | 待实验 |
| GraphRAG + GNN | 待实验 | 待实验 | 待实验 | 待实验 |
| GraphRAG + GNN + Rerank | 待实验 | 待实验 | 待实验 | 待实验 |

## 消融实验关注点

- 图谱遍历是否提升多跳问题召回。
- GNN 节点表示是否提升长尾实体召回。
- Rerank 是否提升最终上下文质量。
- 混合检索是否引入明显延迟。
