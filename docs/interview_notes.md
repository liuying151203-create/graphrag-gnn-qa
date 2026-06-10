# 面试与简历说明

本文档用于整理当前项目的简历表述、面试讲解重点和高频追问回答。

## 项目概述

本项目是一个面向异构图神经网络论文的 GraphRAG 问答系统，支持 PDF/TXT/Markdown 文档解析、文本分块、BGE-m3 向量化、Milvus 向量检索、LLM 实体关系抽取、Neo4j 知识图谱构建、图谱邻域扩展、Hybrid Evidence 融合、轻量 Rerank、LLM 回答生成、citations 返回和自动评估。当前已在 6 篇领域论文和 30 个自建问题上完成端到端评测，并记录 Vector-only 与 GraphRAG hybrid 的检索对比。

## 简历 Bullet

- 构建面向科研论文的 GraphRAG 问答系统，打通 PDF 解析、文本分块、Milvus 向量检索、Neo4j 知识图谱构建、混合证据融合、LLM 生成回答与 citations 返回的端到端链路。
- 设计轻量知识图谱 schema，将论文中的 Method、Task、Dataset、Metric、Concept 等实体及其关系写入 Neo4j，并支持实体邻域检索和可解释证据路径返回。
- 实现 Hybrid Evidence 模型，对向量召回文本块和图谱关系证据进行统一建模、去重、融合排序，并在 QA prompt 和 citations 前接入轻量 Rerank。
- 构建 6 篇异构图神经网络领域论文和 30 条自建评测问题的 domain mini set，自动记录 Recall@K、MRR、citation hit、answer keyword hit 和延迟等指标。
- 完成 Vector-only 与 GraphRAG hybrid 检索对比：在当前 domain mini set 上 GraphRAG hybrid 的 evidence keyword recall 从 0.7161 提升到 0.7492，citations keyword hit rate 为 1.0000。

## 面试讲解主线

建议按下面顺序讲：

1. 传统 RAG 在论文问答中容易只召回相似文本，弱于复杂实体关系和跨段落证据组织。
2. 本项目把论文切成 chunk，用 Milvus 做语义召回，同时用 LLM 抽取实体关系写入 Neo4j。
3. 用户提问后，系统同时走向量检索和图谱邻域检索，并统一为 Hybrid Evidence。
4. Hybrid Evidence 经过去重、fusion score 排序和轻量 Rerank 后进入 QA prompt。
5. 最终答案返回 citations，能追踪到使用了哪些 chunk 或图谱证据。
6. 用 30 条领域问题做评估，并对比 Vector-only 与 GraphRAG hybrid 的检索覆盖。

## 高频追问

### GraphRAG 检索流程是什么？

用户问题进入系统后，会先生成向量查询，召回 Milvus 中的相关文本块；同时从问题中抽取候选实体查询词，在 Neo4j 中做邻域检索。随后系统把向量证据和图谱证据转换成统一的 Hybrid Evidence，按 `document_id + chunk_id` 去重，并结合原始相关性分数和 rank 分数计算 `fusion_score`。问答阶段会对混合证据做轻量 Rerank，再组织 prompt 调用 LLM，并返回 citations。

### 知识图谱 schema 怎么设计？

当前 schema 面向科研论文问答，实体类型包括 `Paper`、`Author`、`Institution`、`Method`、`Task`、`Dataset`、`Metric` 和 `Concept`。关系类型包括 `AUTHORED_BY`、`AFFILIATED_WITH`、`PROPOSED_METHOD`、`USES_METHOD`、`SOLVES_TASK`、`USES_DATASET`、`EVALUATED_BY` 和 `RELATED_TO`。这个 schema 不追求覆盖所有通用百科实体，而是优先服务论文方法、任务、数据集和指标之间的检索与解释。

### Vector-only 和 GraphRAG 对比结果怎么看？

在 6 篇领域论文和 30 条问题上，Vector-only 的 evidence keyword recall 为 0.7161，GraphRAG hybrid 为 0.7492。两者 Recall@K 和 MRR 当前都是 1.0000，说明这个 mini set 的核心证据大多能被 TopK 找到；GraphRAG 的收益主要体现在多关键词证据覆盖略高。这个结果可以说明图谱增强方向有效但收益还不大，后续需要更强的 graph query、rerank/no-rerank 消融和人工复核。

### Rerank 当前怎么做，为什么说是 lightweight？

当前 Rerank 是轻量关键词 overlap reranker，用于验证 `Hybrid Evidence -> Rerank -> QA Prompt -> Citations` 这条工程链路。它不是 BGE Reranker 或 cross-encoder，因此不能声称已经实现强语义重排。它的价值是把 rerank 接口、数据结构和 QA/citation 接入点打通，后续可以替换成 BGE Reranker 并做消融对比。

### GNN/GAT 为什么不作为当前核心结论？

当前项目已经实现 Neo4j 图结构导出，可以生成后续 GNN 训练所需的节点和边数据。但还没有完成节点初始特征构造、GraphSAGE/GAT 训练、节点向量写入和 GNN-assisted retrieval 的评估闭环。因此简历和面试中应把 GNN/GAT 表述为探索性增强方向，而不是当前核心成果或效果提升来源。

### 当前系统瓶颈和下一步优化是什么？

当前瓶颈主要有三个：第一，评估平均延迟较高，`/retrieval/debug` 和 `/qa/ask` 合计约 31 秒；第二，答案关键词评估偏严格，无法完全覆盖同义表达和缩写；第三，GraphRAG hybrid 相比 Vector-only 的提升还不大。下一步应优先做 rerank/no-rerank 消融、优化 prompt 和上下文排序，并在性能层面缓存模型与服务依赖，减少重复初始化和 LLM 调用耗时。

## 安全措辞

推荐写法：

- 已完成 GraphRAG 领域 PDF 问答闭环。
- 已完成 Neo4j 图结构导出，为后续 GNN 节点表示学习提供数据基础。
- 探索引入图结构特征辅助实体召回和候选证据排序。
- 后续可接入 BGE Reranker 或 GraphSAGE/GAT 做进一步消融。

避免写法：

- 已使用 GAT 显著提升长尾实体召回。
- 已将 GNN 增强节点向量写入 Milvus。
- GraphRAG 准确率提升 30%。
- 端到端 2 秒响应。

这些说法当前没有完整训练、消融或性能证据支持，面试中容易被追问。

## 可展示结果

当前最稳的量化结果：

```text
数据：6 篇异构图神经网络与鲁棒性论文，30 条自建问题
文本块：499
图谱抽取记录：499
Vector-only evidence keyword recall：0.7161
GraphRAG hybrid evidence keyword recall：0.7492
Recall@K：1.0000
MRR：1.0000
Citation keyword hit rate：1.0000
Answer hit rate：0.8000
```

讲解时要补充：这些是轻量关键词指标，不等价于人工答案正确率；当前 baseline 用于证明工程闭环和初步检索收益，后续仍需要更细粒度的人工复核和消融实验。
