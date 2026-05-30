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

小规模 dev set 用于当前阶段的指标验证和配置对比：

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

这些指标是轻量关键词指标，适合当前小规模 dev set 的链路验证和配置对比；正式实验仍需要更大问题集和人工或模型辅助评测。当前 `questions.dev.jsonl` 基于 `data/raw/sample.txt`、`chunks.jsonl` 和 `graph_triples.jsonl` 构造，重点覆盖信息碎片、vector-only retrieval、knowledge graph、graph traversal、GNN、GraphRAG 关系和证据类型等概念。

评估不同融合策略时，可以在 `.env` 中调整：

```env
FUSION_SCORE_WEIGHT=0.7
FUSION_RANK_WEIGHT=0.3
```

`FUSION_SCORE_WEIGHT` 越高，排序越依赖原始相关性分数；`FUSION_RANK_WEIGHT` 越高，排序越依赖检索 rank。

评估输出会从 `/retrieval/debug` 的 `fusion_weights` 读取实际生效权重并写入 `run_config`，方便对比不同 `.env` 配置下的结果。

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
