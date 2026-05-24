# GraphRAG-GNN-QA

基于 GraphRAG 与 GNN 的复杂关联知识智能问答系统。

本项目面向科研论文、技术文档等长文本知识场景，针对传统 RAG 在跨文档复杂实体关系推理中容易出现的信息断裂与大模型幻觉问题，设计并实现一个结合向量检索、知识图谱检索和图神经网络节点表示增强的智能问答系统。

## 项目目标

- 构建面向科研知识的文档问答系统。
- 使用 BGE-m3 对文档切片和实体节点进行语义向量化。
- 使用 Neo4j 存储实体、关系和证据路径。
- 使用 Milvus 存储文本块向量和 GNN 增强后的节点向量。
- 使用 GraphRAG 实现向量召回与图谱多跳检索的融合。
- 使用 GAT 对知识图谱节点进行结构感知表示学习。
- 使用 FastAPI 对外提供文档导入、问答和检索调试接口。

## 技术栈

- Python
- FastAPI
- LangChain
- Neo4j
- Milvus
- BGE-m3
- PyTorch
- PyTorch Geometric
- GAT

## 项目文档

- [系统架构设计](docs/architecture.md)
- [API 设计](docs/api.md)
- [知识图谱 Schema 设计](docs/graph_schema.md)
- [实验设计](docs/experiments.md)
- [项目结构说明](docs/project_structure.md)

## 本地运行

### 创建虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 安装依赖

```powershell
pip install -r requirements.txt
pip install -e .
```

### 创建本地环境变量文件

```powershell
copy .env.example .env
```

### 启动 API 服务

```powershell
uvicorn graphrag_gnn_qa.main:app --app-dir src --reload
```

### 健康检查

访问：

```text
http://localhost:8000/health
```

预期返回：

```json
{
  "status": "ok",
  "app_name": "graphrag-gnn-qa",
  "environment": "development"
}
```

## 当前已实现能力

- FastAPI 服务启动
- `/health` 健康检查接口
- `.env` 环境变量加载
- TXT、Markdown、PDF 文档读取
- 固定长度重叠文本切分
- 基于 BGE-m3 的文本向量生成脚本
- Milvus 向量集合创建与文本块向量导入脚本
- 基于 Milvus 的文本块 TopK 向量检索脚本
- `POST /retrieve` 向量检索 API
- `POST /graph/retrieve` 图谱检索 API
- `POST /retrieval/debug` 检索调试 API
- `POST /qa/ask` GraphRAG-aware 问答 API
- GraphRAG Context Builder，用于统一组织向量上下文、图谱上下文和 LLM prompt
- Hybrid Retrieval Result Model，用于统一表示、融合排序并去重向量证据和图谱证据
- 基于 LLM 的实体关系抽取脚本
- Neo4j 图谱节点与关系写入脚本
- 基于 Neo4j 的图谱邻域检索脚本
- Pytest 自动化测试

## 文档处理流程

当前阶段的文档处理流程：

```text
输入 TXT / Markdown / PDF
  -> DocumentLoader 读取文本内容
  -> TextSplitter 切分为带 chunk_id 的文本块
  -> 生成 chunks.jsonl
  -> Embedding 模块生成 chunk_embeddings.jsonl
  -> 写入 Milvus
  -> Vector Search TopK 检索
  -> GraphRAG-aware 问答生成
  -> 实体关系抽取生成 graph_triples.jsonl
  -> 写入 Neo4j 知识图谱
  -> Graph Search 邻域检索
```

### 生成文本块数据

将原始文档放入：

```text
data/raw/
```

运行：

```powershell
python scripts/ingest_documents.py
```

默认输出：

```text
data/processed/chunks.jsonl
```

也可以自定义切分参数：

```powershell
python scripts/ingest_documents.py --chunk-size 800 --chunk-overlap 120
```

### 生成文本向量数据

在生成 `chunks.jsonl` 后，运行：

```powershell
python scripts/embed_chunks.py
```

默认输出：

```text
data/processed/chunk_embeddings.jsonl
```

可以指定 Embedding 模型：

```powershell
python scripts/embed_chunks.py --model-name BAAI/bge-m3 --batch-size 16
```

### 导入 Milvus

启动 Milvus 相关服务：

```powershell
docker compose up -d etcd minio milvus
```

将文本块向量写入 Milvus：

```powershell
python scripts/load_embeddings_to_milvus.py
```

如果需要重建 collection：

```powershell
python scripts/load_embeddings_to_milvus.py --drop-existing
```

### 检索相关文本块

在完成 Milvus 导入后，可以输入一个问题检索最相关的文本块：

```powershell
python scripts/search_chunks.py "What is GraphRAG?" --top-k 3
```

当前阶段会输出相关文本块的：

- 相似度分数
- `chunk_id`
- 来源文件
- 文本块内容

### 启动 API 服务

```powershell
uvicorn graphrag_gnn_qa.main:app --reload
```

向量检索 API：

```text
POST /retrieve
```

请求示例：

```json
{
  "query": "What is GraphRAG?",
  "top_k": 3
}
```

图谱检索 API：

```text
POST /graph/retrieve
```

请求示例：

```json
{
  "query": "GraphRAG",
  "top_k": 5,
  "max_depth": 2
}
```

检索调试 API：

```text
POST /retrieval/debug
```

请求示例：

```json
{
  "query": "What is GraphRAG?",
  "vector_top_k": 3,
  "graph_top_k": 5,
  "graph_max_depth": 2
}
```

该接口会同时返回：

- `graph_query_terms`：从问题中生成的图谱查询词
- `vector_results`：Milvus 文本块召回结果
- `graph_results`：Neo4j 图谱关系召回结果
- `hybrid_results`：统一后的混合检索证据，包含 `fusion_score`，会按 `document_id + chunk_id` 合并重复证据，并按融合分降序返回，便于后续 rerank、GNN 和引用排序

问答 API：

```text
POST /qa/ask
```

请求示例：

```json
{
  "question": "What is GraphRAG?",
  "top_k": 3
}
```

运行问答 API 前，需要在 `.env` 中配置 `LLM_API_KEY`。

当前问答 API 会同时返回：

- `sources`：Milvus 向量检索召回的文本证据
- `graph_sources`：Neo4j 图谱检索召回的关系证据

图谱召回会先从自然语言问题中抽取候选实体查询词。例如：

```text
What is GraphRAG?
  -> GraphRAG
```

这样可以避免完整问句直接匹配 Neo4j 节点名导致 `graph_sources` 为空。

### 抽取实体关系

在生成 `chunks.jsonl` 后，可以运行：

```powershell
python scripts/extract_graph.py
```

默认输出：

```text
data/processed/graph_triples.jsonl
```

调试时可以只处理前几条 chunk：

```powershell
python scripts/extract_graph.py --limit 3
```

### 导入 Neo4j

启动 Neo4j：

```powershell
docker compose up -d neo4j
```

将 `graph_triples.jsonl` 写入 Neo4j：

```powershell
python scripts/load_graph_to_neo4j.py
```

如果已经创建过唯一约束，也可以跳过约束创建：

```powershell
python scripts/load_graph_to_neo4j.py --skip-constraints
```

默认会为实体节点创建唯一约束，并按实体类型创建节点，例如 `Method`、`Task`、`Concept`。关系会保留以下证据属性：

```text
chunk_id
document_id
source
evidence
confidence
```

### 检索图谱邻域

在完成 Neo4j 导入后，可以按实体关键词检索图谱邻域：

```powershell
python scripts/search_graph.py "GraphRAG" --top-k 5 --max-depth 2
```

当前图谱检索会返回：

- 匹配到的中心节点
- 邻域关系三元组
- `chunk_id`
- `source`
- `evidence`
- `confidence`

## 初步开发路线

### 阶段一：项目初始化

- 创建 GitHub 仓库。
- 编写 README。
- 配置 Python 依赖。
- 配置 Git 忽略规则。

### 阶段二：Vector-only RAG Baseline

- 实现文档读取。
- 实现文本切分。
- 接入 BGE-m3 Embedding。
- 接入 Milvus 向量检索。
- 实现基础问答接口。

### 阶段三：知识图谱构建

- 使用 LLM 抽取实体和关系。
- 设计 Neo4j 图谱 Schema。
- 将三元组、来源和证据写入 Neo4j。

### 阶段四：GraphRAG 混合检索

- 实现问题实体识别。
- 实现 Neo4j 子图遍历。
- 融合向量检索结果和图谱路径结果。
- 构建带证据的问答上下文。

### 阶段五：GNN 节点表示增强

- 从 Neo4j 导出节点和边。
- 使用 BGE-m3 生成节点初始语义向量。
- 使用 GAT 学习结构增强节点表示。
- 将增强后的节点向量写入 Milvus。

### 阶段六：Rerank 与实验评估

- 接入 BGE Reranker。
- 构建多跳 QA 测试集。
- 对比 Vector-only RAG、GraphRAG、GraphRAG + GNN、GraphRAG + GNN + Rerank。

## 预期实验指标

- Recall@K
- MRR
- Answer Accuracy
- Faithfulness
- End-to-end Latency

## 项目状态

当前状态：项目初始化阶段。
