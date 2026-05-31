# 系统架构设计

## 目标

本系统旨在构建一个面向科研论文和技术文档的复杂关联知识问答系统，通过 GraphRAG 将向量检索与知识图谱检索结合起来，缓解传统 RAG 在复杂实体关系、多跳关联和证据可解释性方面的不足。

当前阶段定位为 GraphRAG MVP：已经完成文档处理、向量检索、图谱构建、混合证据融合、问答生成、引用返回、轻量自动指标和评估记录链路。GNN 节点表示增强、Rerank 和更完整的对比实验作为后续增强模块逐步接入。

## 总体架构

```text
离线数据构建：

原始文档
  -> 文档读取与文本切分
  -> 文本块 JSONL
  -> Embedding 生成
  -> Milvus 文本块向量库

文本块 JSONL
  -> LLM 实体关系抽取
  -> 三元组 JSONL
  -> Neo4j 知识图谱

在线问答：

用户问题
  -> 查询理解与候选实体抽取
  -> Milvus 向量检索
  -> Neo4j 图谱邻域检索
  -> Hybrid Evidence 转换、去重和融合排序
  -> GraphRAG Context Builder
  -> LLM 生成答案
  -> 返回 answer、sources、graph_sources 和 citations

后续增强：

Neo4j 知识图谱
  -> GNN 节点表示学习
  -> GNN 节点召回
  -> 接入 Hybrid Evidence Fusion
  -> Rerank
  -> 更高质量的上下文和引用排序
```

## 核心模块

### API 层

使用 FastAPI 提供 HTTP 接口。

当前已实现：

- `GET /health`：健康检查接口
- `POST /retrieve`：Milvus 文本块向量检索接口
- `POST /graph/retrieve`：Neo4j 图谱邻域检索接口
- `POST /retrieval/debug`：向量、图谱和混合检索调试接口
- `POST /qa/ask`：GraphRAG-aware 问答接口

待实现：

- `POST /documents/upload`：文档上传并触发解析、切分、向量写入和图谱写入的端到端导入接口

### 配置层

通过 `.env` 管理运行配置，包括 LLM、Neo4j、Milvus、Embedding、TopK 和混合融合参数。

当前已实现：

- LLM、Neo4j、Milvus 和检索参数配置
- `FUSION_SCORE_WEIGHT` 和 `FUSION_RANK_WEIGHT` 可配置
- 融合权重非负且不能同时为零的校验

### 文档处理层

负责读取 TXT、Markdown、PDF 等文档，完成文本切分和元数据维护。

当前已实现：

- `DocumentLoader`：文档读取
- `TextSplitter`：固定长度重叠文本切分
- `scripts/ingest_documents.py`：生成 `data/processed/chunks.jsonl`

### 向量检索层

负责文本块向量生成、Milvus 写入和 TopK 语义检索。

当前已实现：

- Embedding 模型抽象和 hash embedding 测试实现
- `scripts/embed_chunks.py`：生成 chunk embeddings
- `scripts/load_embeddings_to_milvus.py`：写入 Milvus
- `VectorRetriever`：基于向量库的文本块检索
- `scripts/search_chunks.py`：命令行检索脚本
- `POST /retrieve`：向量检索 API

### 图谱构建与检索层

负责从文本块抽取实体关系，并基于 Neo4j 进行图谱存储和邻域检索。

当前已实现：

- `GraphExtractor`：基于 LLM 的实体关系抽取
- `Neo4jGraphStore`：Neo4j 节点和关系写入
- `scripts/extract_graph.py`：生成 graph triples
- `scripts/load_graph_to_neo4j.py`：写入 Neo4j
- `GraphRetriever`：按实体关键词检索图谱邻域关系
- `scripts/search_graph.py`：命令行图谱检索脚本
- `POST /graph/retrieve`：图谱检索 API

### GraphRAG 混合检索层

负责统一向量证据和图谱证据，形成可排序、可去重、可解释的 Hybrid Evidence。

当前已实现：

- `HybridEvidence`：统一证据结构
- `fusion_score`：融合原始相关性分数和 rank 分数
- `deduplicate_hybrid_evidences`：按 `document_id + chunk_id` 合并重复证据
- `build_hybrid_retrieval_result`：构造完整混合检索结果
- `/retrieval/debug`：返回 `vector_results`、`graph_results`、`hybrid_results` 和实际 `fusion_weights`

### RAG 问答层

负责编排向量检索、图谱检索、混合证据构建、prompt 组织和 LLM 调用。

当前已实现：

- `RAGQAService`：GraphRAG-aware 问答服务
- `build_hybrid_rag_prompt`：基于去重混合证据构造 prompt
- `/qa/ask`：返回答案、向量来源、图谱来源和混合证据引用
- `citations`：返回答案使用的 `evidence_id`、`evidence_type`、`document_id`、`chunk_id`、`source` 和 `fusion_score`

### 评估层

负责记录不同检索和问答配置下的输出结果，为后续消融实验和指标对比提供基础数据。

当前已实现：

- `scripts/evaluate_retrieval.py`：批量调用 `/retrieval/debug` 和 `/qa/ask`
- `run_config`：记录 API 地址、TopK 参数和实际生效的 fusion 权重
- `metrics`：记录基于期望关键词的 Recall@K、MRR、citation hit rate、answer keyword match 和 latency
- `summary`：记录召回数量、引用数量、Top hybrid evidence 摘要和聚合指标
- `data/eval/questions.sample.jsonl`：样例问题集，用于 smoke test 和链路验证
- `data/eval/questions.dev.jsonl`：小规模 dev set，用于当前阶段的指标验证和配置对比

待增强：

- 扩充 20 到 50 条小规模评估集，用于更稳定的对比实验
- 补充 Vector-only、GraphRAG、GraphRAG + Rerank、GraphRAG + GNN 等对比结果
- 引入人工或模型辅助评测，补足关键词指标的局限性

### GNN 表示学习层

从 Neo4j 导出节点和边，将节点语义向量作为初始特征，使用 GAT 学习结构感知的节点表示，并辅助长尾实体召回。

当前状态：

- 设计目标已明确
- 尚未实现 `gnn/` 模块、图导出、GAT 模型训练和 GNN 召回接入

### Rerank 层

对混合候选证据进行二阶段重排序，提升最终上下文质量。

当前状态：

- 配置中已预留 `reranker_model` 和 `rerank_top_k`
- 尚未实现独立 reranker 模块，也尚未接入 QA prompt 构建流程

## 当前实现状态

### 已完成

- FastAPI 服务和健康检查
- `.env` 配置加载和环境变量模板
- Docker Compose 基础服务配置
- 文档读取、文本切分和 chunks JSONL 生成
- Embedding 生成脚本
- Milvus 文本块向量写入和向量检索
- Neo4j 实体关系写入和图谱邻域检索
- LLM 实体关系抽取
- GraphRAG 混合检索证据建模、融合排序和去重
- `/retrieve`、`/graph/retrieve`、`/retrieval/debug`、`/qa/ask` API
- GraphRAG Context Builder 和 QA citations
- 可配置 fusion 权重
- 评估记录脚本、轻量自动指标、聚合摘要和样例评估数据
- Pytest 自动化测试覆盖核心模块

### 部分完成

- 评估体系：已能记录逐题结果、轻量关键词指标和聚合摘要，仍需更大问题集、对比实验和人工或模型辅助评测
- 实验数据集：已有样例问题集和小规模 dev set，尚未扩展为更稳定的 20 到 50 条评估集
- 文档导入：已有命令行数据构建流程，尚未提供上传 API

### 待实现

- `POST /documents/upload` 文档上传与端到端索引构建 API
- GNN 图导出、GAT 训练、节点向量写入和 GNN 辅助召回
- Rerank 模块和二阶段证据排序
- 更完整的实验对比报告和消融实验

## 路线图

### Stage 1：GraphRAG MVP

状态：已完成。

目标是打通最小可用 GraphRAG 问答链路：

- 文档读取和切分
- 文本块向量化和 Milvus 检索
- 实体关系抽取和 Neo4j 检索
- 向量证据与图谱证据融合
- GraphRAG prompt 构造
- QA answer、sources、graph_sources 和 citations 返回
- 检索调试和评估记录

### Stage 2：评估指标与项目展示完善

状态：建议优先推进。

目标是让项目更适合简历展示和面试讲解：

- 更新 README、架构文档和实验文档中的真实实现状态
- 记录当前 dev set 的可复现实测结果
- 扩充 20 到 50 条小规模评估问题
- 输出 Vector-only、GraphRAG 等可对比的实验结果表格

### Stage 3：Rerank 增强

状态：待实现。

目标是在 Hybrid Evidence 之后增加二阶段排序：

- 新增 reranker 模块
- 支持 cross-encoder 或轻量规则 rerank
- 将 rerank 后的证据用于 QA prompt 和 citations
- 对比 rerank 前后的 citation hit rate 和 answer quality

### Stage 4：GNN 节点表示增强

状态：待实现。

目标是体现项目名中的 GNN 能力：

- 从 Neo4j 导出节点、边和节点文本特征
- 构造 PyTorch Geometric 数据
- 使用 GAT 学习结构感知节点表示
- 将 GNN 节点向量写入 Milvus 或单独索引
- 增加 GNN-assisted node retrieval
- 将 GNN 召回结果接入 Hybrid Evidence Fusion

### Stage 5：更完整的消融实验

状态：待实现。

目标是验证各模块的实际收益：

- Vector-only RAG
- GraphRAG
- GraphRAG + Rerank
- GraphRAG + GNN
- GraphRAG + GNN + Rerank
- 不同 fusion 权重配置对比

## 面试讲解定位

当前项目可以定位为：

```text
一个已经打通 GraphRAG MVP 的工程项目：
支持文档入库、向量检索、图谱检索、混合证据融合、问答生成、引用返回和评估记录。
后续通过 Rerank、GNN 节点表示增强和更完整的对比实验继续提升检索质量。
```

面试中可以重点强调：

- 工程链路完整，不只是单点 demo
- 支持向量检索和图谱检索的统一证据建模
- 支持可解释 citations
- 支持可配置 fusion 权重和可复现实验记录
- GNN 和 Rerank 有清晰的后续接入路线
