# 实验设计

## 实验目标

通过对比不同检索策略，验证 GraphRAG 和 GNN 节点表示增强在复杂关联问答中的效果。

## 数据集设计

当前使用三层数据集推进实验：

- `questions.sample.jsonl`：最小 smoke test，用于快速验证链路是否可运行。
- `questions.dev.jsonl`：20 条项目自定义 dev set，用于验证 GraphRAG MVP、Rerank 和评估指标是否稳定。
- `questions.hotpotqa_mini.jsonl`：由 HotpotQA dev distractor 子集生成的 50 条多跳问答集，用于后续做更有说服力的检索策略对比。

HotpotQA mini 的 raw 文档来自 HotpotQA 每条样本自带的 `context` 段落；问题集保留 `question`、`answer`、`supporting_facts`、`type` 和 `level`，并转换为当前评估脚本可用的关键词指标字段。当前阶段先不修改知识图谱 schema，Wikipedia 通用实体会主要映射到 `Concept`、`Author`、`Institution` 或 `RELATED_TO` 等已有类型；如果后续发现图谱抽取收益受限，再单独扩展通用实体类型。

数据来源：

- AI / NLP / GNN / RAG 相关论文摘要
- 技术博客
- 开源项目文档
- HotpotQA dev distractor 子集中的 Wikipedia context 和 supporting facts

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

当前工程已完成 GNN 的第一步输入准备：可以从 Neo4j 导出节点和边，生成 `data/processed/graph_dataset.json`。后续还需要构造节点初始语义向量、PyTorch Geometric 数据和 GAT 训练流程。

### GraphRAG + GNN + Rerank

在候选上下文合并后，使用 Reranker 对候选证据进行重排序。

当前工程已接入轻量关键词 overlap reranker，并支持通过 `RERANKER_TYPE=bge` 切换到 BGE Reranker，用于打通 `Hybrid Evidence -> Rerank -> QA Prompt -> Citations` 链路。历史 baseline 仍主要记录 lightweight rerank 结果，BGE Reranker 的正式收益需要后续补充消融实验。

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

HotpotQA mini 用于后续中等规模多跳检索对比：

```text
data/eval/questions.hotpotqa_mini.jsonl
```

生成 HotpotQA mini：

```powershell
python scripts/build_hotpotqa_mini.py --input-file data/raw/hotpotqa_official/hotpotqa_hf_rows_validation_50.json --limit 50 --output-raw-dir data/raw/hotpotqa_mini --output-questions-file data/eval/questions.hotpotqa_mini.jsonl
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

### 2026-06-10 Domain PDF mini set

本次实测使用 6 篇异构图神经网络与鲁棒性相关论文 PDF，验证项目在真实领域论文语料上的 GraphRAG 端到端效果。

数据与配置：

- 原始文档：`data/raw/domain_papers/*.pdf`
- 问题集：`data/eval/questions.domain_mini.jsonl`
- 题目数量：30
- 文本切分：`chunk_size=700`，`chunk_overlap=100`
- 文本块数量：499
- 图谱抽取记录：499
- 检索参数：`vector_top_k=3`，`graph_top_k=5`，`graph_max_depth=2`，`qa_top_k=3`
- 融合权重：`score_weight=0.7`，`rank_weight=0.3`
- 输出文件：`data/eval/retrieval_eval_results.jsonl`、`data/eval/retrieval_eval_summary.json`

领域语料包括：

- HAN: Heterogeneous Graph Attention Network
- HeCo: Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning
- RoHe: Robust Heterogeneous Graph Neural Networks against Adversarial Attacks
- HeteroGuard: Defending Heterogeneous Graph Neural Networks against Adversarial Attacks
- FastRo-HGCN: A Fast and Robust Attention-Free Heterogeneous Graph Convolutional Network
- HSeCo: Robust Heterogeneous GNNs via Semantic Attention and Contrastive Learning

运行命令：

```powershell
python scripts/evaluate_retrieval.py --input-file data/eval/questions.domain_mini.jsonl --vector-top-k 3 --graph-top-k 5 --graph-max-depth 2 --qa-top-k 3 --timeout 180 --progress-every 1
```

实测结果：

| 方法 | 问题数 | Evidence keyword recall | Recall@K | MRR | Top1 evidence hit rate | Citation hit rate | Answer keyword recall | Answer hit rate | 平均总延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GraphRAG + lightweight Rerank | 30 | 0.7492 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6222 | 0.8000 | 31341.8940 ms |

检索对比：

| 检索方式 | 问题数 | Evidence keyword recall | Recall@K | MRR | Top1 keyword hit rate |
|---|---:|---:|---:|---:|---:|
| Vector-only | 30 | 0.7161 | 1.0000 | 1.0000 | 1.0000 |
| GraphRAG hybrid | 30 | 0.7492 | 1.0000 | 1.0000 | 1.0000 |

延迟拆分：

| 指标 | 平均耗时 |
|---|---:|
| `/retrieval/debug` | 14598.4949 ms |
| `/qa/ask` | 16743.3991 ms |

逐题观察：

- 检索侧 `Recall@K`、`MRR`、Top1 evidence hit rate 和 citation keyword hit rate 均为 1.0，说明 30 个领域问题都能在 TopK 混合证据中命中至少一个期望证据关键词，且 citations 能指向相关证据。
- 与 Vector-only 相比，GraphRAG hybrid 的 `avg_evidence_keyword_recall` 从 0.7161 提升到 0.7492；逐题看，`domain_q005`、`domain_q014` 和 `domain_q027` 的混合证据关键词覆盖更高，没有题目出现 Vector-only 覆盖更高。
- `avg_evidence_keyword_recall=0.7492`，说明检索能够命中核心证据，但仍存在多关键词覆盖不足，尤其是跨论文数据集和指标汇总类问题。
- `avg_answer_keyword_recall=0.6222`、`answer_keyword_hit_rate=0.8`，答案关键词指标偏低。部分问题是答案表达与关键词严格匹配不一致，例如 `node-level attentions` 与 `node-level attention`、`topology-level similarity` 与 `topology similarity`。
- `domain_q003`、`domain_q012` 等问题出现检索证据已命中关键词，但 QA 阶段仍回答证据不足的情况，说明后续需要优化上下文排序、证据压缩或 prompt。
- 平均总延迟约 31.3 秒，当前作为 baseline 记录；性能优化后续单独推进。

结论与限制：

- 该结果比样例 dev set 更接近真实论文问答场景，可作为当前项目展示的领域 baseline。
- 当前指标是轻量关键词指标，不等价于人工答案正确率；后续需要补充人工复核或模型辅助评测。
- 当前 GraphRAG hybrid 相比 Vector-only 有小幅证据覆盖提升，但收益还不大；后续应继续做更明确的 GraphRAG/no-graph、rerank/no-rerank 消融，并结合人工复核判断真实答案质量。
- GNN/GAT 仍应定位为探索性增强，当前不作为实验结论主线。

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

## 消融实验结果汇总

当前已经形成两个层面的 baseline：一是检索侧的 Vector-only 与 GraphRAG hybrid 对比，二是端到端 QA 侧的 GraphRAG + lightweight Rerank baseline。工程上已支持 BGE Reranker，以下表格用于后续继续补充 BGE Reranker、GNN 和 fusion 权重消融。

### Domain PDF mini set 已记录结果

数据集：`data/eval/questions.domain_mini.jsonl`，共 30 题。

| 方法 | 阶段 | 问题数 | Evidence keyword recall | Recall@K | MRR | Top1 evidence hit rate | Citation hit rate | Answer keyword recall | Answer hit rate | 平均总延迟 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vector-only | 检索 | 30 | 0.7161 | 1.0000 | 1.0000 | 1.0000 | 不适用 | 不适用 | 不适用 | 未单独记录 |
| GraphRAG hybrid | 检索 | 30 | 0.7492 | 1.0000 | 1.0000 | 1.0000 | 不适用 | 不适用 | 不适用 | 未单独记录 |
| GraphRAG + lightweight Rerank | QA | 30 | 0.7492 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6222 | 0.8000 | 31341.8940 ms |

观察：

- GraphRAG hybrid 相比 Vector-only 的 evidence keyword recall 从 0.7161 提升到 0.7492，说明图谱证据融合带来小幅证据覆盖收益。
- 当前 Recall@K、MRR 和 Top1 evidence hit rate 均为 1.0000，说明该 mini set 的核心证据大多能被 TopK 找到；后续评估需要更难问题集或更细粒度人工复核。
- GraphRAG + lightweight Rerank 已打通 `Hybrid Evidence -> Rerank -> QA Prompt -> Citations` 链路，但 lightweight reranker 不等价于 BGE Reranker，不能作为强语义重排结论。
- 平均总延迟约 31.3 秒，当前只作为 baseline 记录，性能优化后续单独推进。

### 待补消融表

| 方法 | 状态 | 目标问题 |
|---|---|---|
| GraphRAG without rerank | 待补 | 衡量 lightweight rerank 对 citations 和 answer keyword hit 的影响 |
| GraphRAG + BGE Reranker | 待补实验 | 验证语义重排是否优于关键词 overlap rerank |
| GraphRAG + GNN | 待实现 | 验证 GNN 节点表示是否改善长尾实体或多跳实体召回 |
| GraphRAG + GNN + Rerank | 待实现 | 验证 GNN 召回与二阶段重排叠加后的端到端收益 |
| 不同 fusion 权重配置 | 待补 | 比较 `score_weight` 与 `rank_weight` 对证据覆盖和答案质量的影响 |

## 消融实验关注点

- 图谱遍历是否提升多跳问题召回。
- GNN 节点表示是否提升长尾实体召回。
- Rerank 是否提升最终上下文质量。
- 混合检索是否引入明显延迟。
