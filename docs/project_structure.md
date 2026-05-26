# 项目结构说明

本文档用于说明当前项目目录结构、各模块职责，以及当前已经实现的数据处理流程，方便后续复盘和扩展。

## 当前目录结构

```text
graphrag-gnn-qa/
├── data/
│   ├── eval/
│   │   └── questions.sample.jsonl
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── tmp/
│       └── .gitkeep
├── docs/
│   ├── api.md
│   ├── architecture.md
│   ├── experiments.md
│   ├── graph_schema.md
│   └── project_structure.md
├── scripts/
│   ├── embed_chunks.py
│   ├── evaluate_retrieval.py
│   ├── extract_graph.py
│   ├── ingest_documents.py
│   ├── load_graph_to_neo4j.py
│   ├── load_embeddings_to_milvus.py
│   ├── search_graph.py
│   └── search_chunks.py
├── src/
│   └── graphrag_gnn_qa/
│       ├── __init__.py
│       ├── config.py
│       ├── main.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes_debug.py
│       │   ├── routes_graph.py
│       │   ├── routes_health.py
│       │   ├── routes_qa.py
│       │   └── routes_retrieve.py
│       ├── graph/
│       │   ├── __init__.py
│       │   ├── extractor.py
│       │   └── neo4j_store.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── document_loader.py
│       │   └── text_splitter.py
│       ├── llm/
│       │   ├── __init__.py
│       │   └── client.py
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── context_builder.py
│       │   └── qa_service.py
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── graph_retriever.py
│       │   ├── hybrid_result.py
│       │   ├── query_entities.py
│       │   └── vector_retriever.py
│       └── vectorstore/
│           ├── __init__.py
│           ├── embedding.py
│           └── milvus_client.py
├── tests/
│   ├── test_document_loader.py
│   ├── test_context_builder.py
│   ├── test_debug_api.py
│   ├── test_embed_chunks.py
│   ├── test_embedding.py
│   ├── test_evaluate_retrieval.py
│   ├── test_extract_graph.py
│   ├── test_graph_extractor.py
│   ├── test_graph_api.py
│   ├── test_graph_retriever.py
│   ├── test_hybrid_result.py
│   ├── test_health.py
│   ├── test_ingest_documents.py
│   ├── test_load_graph_to_neo4j.py
│   ├── test_milvus_client.py
│   ├── test_neo4j_store.py
│   ├── test_qa_api.py
│   ├── test_qa_service.py
│   ├── test_query_entities.py
│   ├── test_retrieve_api.py
│   ├── test_search_graph.py
│   ├── test_text_splitter.py
│   └── test_vector_retriever.py
├── .env.example
├── .gitignore
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── requirements.txt
```

## 根目录文件

### `README.md`

项目首页文档，面向 GitHub 访问者，说明项目目标、技术栈、运行方式、文档入口和当前实现能力。

### `requirements.txt`

Python 依赖列表，用于安装 FastAPI、LangChain、Neo4j、Milvus、Embedding、测试等相关依赖。

### `pyproject.toml`

Python 项目配置文件，目前包含三类配置：

- 构建系统配置
- 项目包元数据
- Pytest 测试配置

当前项目采用 `src` 布局，因此需要通过 `pip install -e .` 将项目以可编辑模式安装到虚拟环境中。

### `.env.example`

环境变量模板，记录运行项目需要的配置项，例如 LLM、Neo4j、Milvus 和检索参数。

### `.gitignore`

Git 忽略规则，用于避免提交虚拟环境、缓存、日志、模型文件和真实数据文件。

### `docker-compose.yml`

用于启动 Neo4j、Milvus、etcd 和 MinIO 等外部基础服务。

## `src/graphrag_gnn_qa/`

项目核心源码目录。

### `main.py`

FastAPI 应用入口，负责创建应用实例并注册路由。

当前接口：

```text
GET /health
POST /retrieve
POST /qa/ask
```

### `config.py`

配置管理模块，基于 `pydantic-settings` 从 `.env` 读取配置。

Neo4j、Milvus、LLM、Embedding、TopK 和 hybrid fusion 权重等参数都会统一从这里读取。

### `api/`

API 路由目录。

当前包含：

- `routes_health.py`：健康检查接口
- `routes_debug.py`：检索调试接口
- `routes_graph.py`：图谱检索接口
- `routes_qa.py`：问答接口
- `routes_retrieve.py`：向量检索接口

#### `routes_health.py`

提供健康检查 API。

当前接口：

```text
GET /health
```

#### `routes_retrieve.py`

提供向量检索 API。

当前接口：

```text
POST /retrieve
```

该接口接收用户问题，生成 query embedding，调用 Milvus 返回 TopK 相关文本块。

#### `routes_graph.py`

提供图谱检索 API。

当前包含：

```text
POST /graph/retrieve
```

该接口接收实体关键词或短查询，调用 Neo4j 返回匹配中心节点的邻域关系。

#### `routes_debug.py`

提供检索调试 API。

当前接口：

```text
POST /retrieval/debug
```

该接口接收用户问题，同时返回 Milvus 向量召回结果、图谱查询词和 Neo4j 图谱召回结果。

#### `routes_qa.py`

提供 GraphRAG-aware 问答 API。

当前接口：

```text
POST /qa/ask
```

该接口接收用户问题，检索相关文本块和图谱关系，再调用 LLM 生成答案、来源证据和混合证据引用。

### `ingestion/`

文档导入与预处理模块。

当前包含：

- `document_loader.py`
- `text_splitter.py`

#### `document_loader.py`

负责从文件中读取文本内容，并统一封装为 `LoadedDocument`。

当前支持：

- TXT
- Markdown
- PDF

核心输出字段：

- `content`：文档正文
- `source`：文档路径
- `file_name`：文件名
- `file_type`：文件类型

#### `text_splitter.py`

负责将长文本切分成多个带重叠区域的文本块。

核心输出字段：

- `chunk_id`：文本块 ID
- `content`：文本块内容
- `start_index`：在原文中的起始位置
- `end_index`：在原文中的结束位置

### `graph/`

知识图谱构建相关模块目录。

当前包含：

- `extractor.py`：基于 LLM 的实体关系抽取模块
- `neo4j_store.py`：Neo4j 图谱写入模块

#### `extractor.py`

负责从文本块中抽取符合 `docs/graph_schema.md` 的实体和关系。

当前包含：

- `GraphEntity`：图谱实体结构
- `GraphRelation`：图谱关系结构
- `GraphExtractionResult`：单个文本块的抽取结果
- `GraphExtractor`：调用 LLM 完成实体关系抽取
- `build_extraction_prompt`：构造抽取提示词
- `parse_extraction_response`：解析 LLM 返回的 JSON 结果

#### `neo4j_store.py`

负责将实体和关系写入 Neo4j。

当前包含：

- `Neo4jGraphStore`：封装 Neo4j driver、约束创建、节点和关系写入
- `build_entity_id`：生成稳定实体 ID
- `build_entity_merge_query`：构造节点 `MERGE` Cypher
- `build_relation_merge_query`：构造关系 `MERGE` Cypher
- `validate_entity_type`：校验实体类型白名单
- `validate_relation_type`：校验关系类型白名单

### `llm/`

大语言模型调用模块目录。

当前包含：

- `client.py`：OpenAI-compatible LLM 客户端封装

#### `client.py`

负责调用兼容 OpenAI Chat Completions 格式的 LLM 服务。

当前包含：

- `LLMClient`：LLM 调用接口
- `OpenAICompatibleLLMClient`：基于 `httpx` 的 DeepSeek/OpenAI-compatible 客户端

### `rag/`

RAG 问答编排模块目录。

当前包含：

- `context_builder.py`：GraphRAG 上下文和 prompt 构造模块
- `qa_service.py`：GraphRAG-aware 问答服务

#### `qa_service.py`

负责编排向量检索、图谱检索和 LLM 调用，返回答案和来源证据。

当前包含：

- `SourceEvidence`：答案来源证据结构
- `GraphEvidence`：图谱来源证据结构
- `CitationEvidence`：答案引用的混合证据结构
- `QAResult`：问答结果结构
- `RAGQAService`：检索、调用 Context Builder、调用 LLM 的完整问答流程

#### `context_builder.py`

负责把向量检索结果、图谱检索结果和混合证据组织成 GraphRAG prompt。

当前包含：

- `GraphRAGContext`：结构化上下文结果
- `HybridRAGContext`：混合证据上下文结果
- `build_vector_context`：构造向量上下文
- `build_graph_context`：构造图谱上下文
- `build_hybrid_context`：构造去重后的混合证据上下文
- `build_graphrag_context`：组合向量和图谱上下文
- `build_rag_prompt`：构造最终 LLM prompt
- `build_hybrid_rag_prompt`：基于混合证据构造最终 LLM prompt

### `vectorstore/`

向量检索相关模块目录。

当前包含：

- `embedding.py`：文本向量模型封装
- `milvus_client.py`：Milvus 向量库客户端封装

#### `embedding.py`

负责将文本转换为向量表示。

当前包含两类模型：

- `SentenceTransformerEmbeddingModel`：基于 `sentence-transformers` 加载 BGE-m3 等真实 Embedding 模型
- `HashEmbeddingModel`：轻量确定性假向量模型，用于单元测试，避免测试阶段下载大模型

#### `milvus_client.py`

负责读取文本块向量文件，并将向量写入 Milvus。

当前包含：

- `EmbeddingRecord`：文本块向量记录结构
- `read_embedding_records`：读取 `chunk_embeddings.jsonl`
- `infer_embedding_dimension`：推断向量维度
- `prepare_insert_columns`：将记录转换为 Milvus 插入格式
- `MilvusVectorStore`：封装 Milvus 连接、collection 创建、插入和搜索

### `retrieval/`

检索逻辑模块目录。

当前包含：

- `vector_retriever.py`：Vector-only 检索服务
- `graph_retriever.py`：Graph-only 检索服务
- `hybrid_result.py`：统一混合检索证据模型
- `query_entities.py`：查询问题中的候选实体抽取

#### `vector_retriever.py`

负责将用户问题转换成 query embedding，并调用向量库返回 TopK 文本块。

当前包含：

- `VectorSearchStore`：向量搜索存储接口
- `RetrievedChunk`：检索结果结构
- `VectorRetriever`：封装 query embedding 和 vector search 的检索流程

#### `graph_retriever.py`

负责调用 Neo4j 图谱存储查询中心节点及其邻域关系。

当前包含：

- `GraphSearchStore`：图谱搜索存储接口
- `RetrievedGraphRelation`：图谱关系检索结果结构
- `GraphRetriever`：封装 query 规范化、参数校验和图谱邻域检索流程

#### `hybrid_result.py`

负责把向量检索结果和图谱检索结果转换成统一的混合证据结构。

当前包含：

- `EvidenceType`：证据类型枚举，目前包含 `vector_chunk`、`graph_relation` 和 `hybrid`
- `HybridEvidence`：统一证据结构，用于表示文本块证据或图谱关系证据，并携带 `fusion_score`
- `HybridRetrievalResult`：混合检索结果结构
- `normalize_scores`：按证据类型对原始分数进行归一化
- `apply_fusion_scores`：融合原始相关性分数和检索 rank 分数
- `validate_fusion_weights`：校验融合权重
- `rank_hybrid_evidences`：按 `fusion_score` 对混合证据降序排序
- `deduplicate_hybrid_evidences`：按 `document_id + chunk_id` 合并重复证据，并保留来源证据信息
- `build_hybrid_evidences`：合并向量证据和图谱证据
- `build_hybrid_retrieval_result`：构造完整混合检索结果

当前用于 `/retrieval/debug` 返回去重后并按 `fusion_score` 排序的 `hybrid_results`，为后续 rerank、GNN 和引用排序预留统一输入。融合权重由 `.env` 中的 `FUSION_SCORE_WEIGHT` 和 `FUSION_RANK_WEIGHT` 配置。

#### `query_entities.py`

负责从自然语言问题中抽取用于图谱召回的候选实体查询词。

示例：

```text
What is GraphRAG?
  -> GraphRAG
```

当前用于 GraphRAG-aware QA 中的 Neo4j 图谱召回增强。

## `scripts/`

命令行脚本目录，用于执行离线任务。

### `ingest_documents.py`

当前已经实现的文档处理脚本。

默认流程：

```text
data/raw/ 中的原始文档
  -> DocumentLoader 读取内容
  -> TextSplitter 切分文本
  -> data/processed/chunks.jsonl
```

运行方式：

```powershell
python scripts/ingest_documents.py
```

可选参数：

```powershell
python scripts/ingest_documents.py --chunk-size 800 --chunk-overlap 120
```

### `embed_chunks.py`

当前已经实现的文本向量生成脚本。

默认流程：

```text
data/processed/chunks.jsonl
  -> 读取 chunk content
  -> 使用 Embedding 模型生成向量
  -> data/processed/chunk_embeddings.jsonl
```

运行方式：

```powershell
python scripts/embed_chunks.py
```

可选参数：

```powershell
python scripts/embed_chunks.py --model-name BAAI/bge-m3 --batch-size 16
```

### `extract_graph.py`

当前已经实现的实体关系抽取脚本。

默认流程：

```text
data/processed/chunks.jsonl
  -> 调用 LLM 抽取 entities 和 relations
  -> data/processed/graph_triples.jsonl
```

运行方式：

```powershell
python scripts/extract_graph.py
```

调试时可以限制处理数量：

```powershell
python scripts/extract_graph.py --limit 3
```

### `load_embeddings_to_milvus.py`

当前已经实现的 Milvus 向量导入脚本。

默认流程：

```text
data/processed/chunk_embeddings.jsonl
  -> 读取 EmbeddingRecord
  -> 推断向量维度
  -> 创建 Milvus collection
  -> 插入文本块向量
```

运行方式：

```powershell
python scripts/load_embeddings_to_milvus.py
```

如果需要删除并重建 collection：

```powershell
python scripts/load_embeddings_to_milvus.py --drop-existing
```

### `load_graph_to_neo4j.py`

当前已经实现的 Neo4j 图谱写入脚本。

默认流程：

```text
data/processed/graph_triples.jsonl
  -> 读取 entities 和 relations
  -> 创建 Neo4j 唯一约束
  -> MERGE 实体节点
  -> MERGE 图谱关系并保留 evidence 属性
```

运行方式：

```powershell
python scripts/load_graph_to_neo4j.py
```

如果需要跳过约束创建：

```powershell
python scripts/load_graph_to_neo4j.py --skip-constraints
```

### `search_chunks.py`

当前已经实现的 Vector Search Baseline 查询脚本。

默认流程：

```text
用户问题
  -> BGE-m3 query embedding
  -> Milvus rag_chunks collection
  -> TopK 相关文本块
```

运行方式：

```powershell
python scripts/search_chunks.py "What is GraphRAG?" --top-k 3
```

### `search_graph.py`

当前已经实现的 Graph Search Baseline 查询脚本。

默认流程：

```text
用户问题或实体关键词
  -> Neo4j 节点名称匹配
  -> 查询 1 到 N 跳邻域关系
  -> 返回 graph context
```

运行方式：

```powershell
python scripts/search_graph.py "GraphRAG" --top-k 5 --max-depth 2
```

### `evaluate_retrieval.py`

当前已经实现的检索与问答评估记录脚本。

默认流程：

```text
data/eval/questions.sample.jsonl
  -> 调用 /retrieval/debug
  -> 调用 /qa/ask
  -> 写入 run_config、retrieval_debug、qa 和 summary
  -> data/eval/retrieval_eval_results.jsonl
```

其中 `run_config` 会记录本次评估的 API 地址、TopK 参数，以及 `/retrieval/debug` 返回的实际 `fusion_weights`，便于复现实验。

运行方式：

```powershell
python scripts/evaluate_retrieval.py --vector-top-k 3 --graph-top-k 5 --graph-max-depth 2 --qa-top-k 3
```

## `data/`

数据目录。

### `data/eval/`

存放检索与问答评估问题集和评估结果。

当前包含：

```text
questions.sample.jsonl
```

`retrieval_eval_results.jsonl` 是脚本运行生成的评估结果文件。

### `data/raw/`

存放原始文档，例如论文 PDF、Markdown 文档或 TXT 文本。

真实数据文件不会提交到 GitHub。

### `data/processed/`

存放处理后的中间结果，例如：

```text
chunks.jsonl
chunk_embeddings.jsonl
graph_triples.jsonl
```

生成数据不会提交到 GitHub。

### `data/tmp/`

存放临时文件。

## `docs/`

项目文档目录。

当前包含：

- `architecture.md`：系统架构设计
- `api.md`：API 设计
- `graph_schema.md`：知识图谱 Schema 设计
- `experiments.md`：实验设计
- `project_structure.md`：项目结构说明

## `tests/`

自动化测试目录。

当前测试覆盖：

- FastAPI 健康检查接口
- 文档读取模块
- 文本切分模块
- 文档处理脚本
- Embedding 生成模块
- Milvus 导入辅助逻辑
- Vector-only 检索模块
- Graph-only 检索模块
- Hybrid Retrieval Result Model
- 检索调试 API
- 图谱检索 API
- GraphRAG Context Builder
- Query entity extraction 模块
- GraphRAG-aware 问答模块
- 检索与问答评估记录脚本
- 实体关系抽取模块
- Neo4j 图谱写入模块

运行测试：

```powershell
pytest
```

或：

```powershell
python -m pytest
```

## 当前数据流

当前已经实现的数据流：

```text
原始文档
  -> DocumentLoader
  -> LoadedDocument
  -> TextSplitter
  -> TextChunk 列表
  -> ingest_documents.py
  -> chunks.jsonl
  -> embed_chunks.py
  -> chunk_embeddings.jsonl
  -> load_embeddings_to_milvus.py
  -> Milvus collection
  -> search_chunks.py
  -> TopK RetrievedChunk 列表
  -> retrieval/debug
  -> hybrid_results
  -> RAGQAService
  -> answer + sources + graph_sources + citations
  -> extract_graph.py
  -> graph_triples.jsonl
  -> load_graph_to_neo4j.py
  -> Neo4j 知识图谱
  -> search_graph.py
  -> Graph context
```

## 后续扩展方向

下一阶段会在当前 Neo4j 知识图谱基础上继续实现：

```text
Neo4j 知识图谱 + Milvus 向量库
  -> GraphRAG 混合检索
  -> Graph-aware Answer Generation
```

之后继续扩展：

```text
chunks.jsonl
  -> LLM 实体关系抽取
  -> Neo4j 知识图谱
  -> GraphRAG 混合检索
  -> GNN 节点表示增强
```
