# GraphRAG-GNN-QA

面向科研论文和技术文档的 GraphRAG 知识问答系统。

本项目面向科研论文、技术文档等长文本知识场景，针对传统 RAG 在复杂实体关系、多跳证据组织和引用可解释性方面的不足，设计并实现一个结合 Milvus 向量检索、Neo4j 知识图谱检索、混合证据融合、可配置 Rerank、LLM 问答生成和自动评估的 GraphRAG 系统。

当前主线是一个可运行、可评估、可演示的 GraphRAG 工程闭环；GNN/GAT 节点表示增强保留为探索性优化方向，当前已完成 Neo4j 图结构导出，为后续节点表示学习提供数据基础。

## 项目目标

- 构建面向科研知识的文档问答系统。
- 使用 BGE-m3 对文档切片进行语义向量化。
- 使用 Neo4j 存储实体、关系和证据路径。
- 使用 Milvus 存储文本块向量。
- 使用 GraphRAG 实现向量召回与图谱多跳检索的融合。
- 使用可配置 Rerank 对混合证据进行二阶段排序，默认轻量关键词 overlap，可切换到 BGE Reranker。
- 使用评估脚本对 Vector-only 与 GraphRAG hybrid 检索效果进行对比。
- 预留 GNN/GAT 节点表示增强路线，但不作为当前核心结论。
- 使用 FastAPI 对外提供文档导入、问答和检索调试接口。

## 技术栈

- Python
- FastAPI
- Streamlit / Graphviz
- LangChain
- Neo4j
- Milvus
- BGE-m3
- PyTorch / PyTorch Geometric（后续 GNN 实验预留）

## 项目文档

- [系统架构设计](docs/architecture.md)
- [API 设计](docs/api.md)
- [知识图谱 Schema 设计](docs/graph_schema.md)
- [实验设计](docs/experiments.md)
- [演示指南](docs/demo_guide.md)
- [项目后续开发与演示 Demo 计划](docs/development_plan.md)
- [面试与简历说明](docs/interview_notes.md)
- [项目结构说明](docs/project_structure.md)
- [项目开发工作流与个人偏好](docs/development_workflow.md)

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

### 开发者日常启动流程

启动 Neo4j、Milvus、etcd 和 MinIO：

```powershell
docker compose up -d neo4j etcd minio milvus
```

运行测试：

```powershell
python -m pytest
```

启动后端服务：

```powershell
uvicorn graphrag_gnn_qa.main:app --app-dir src --reload
```

如果希望把模型或 pip 缓存放到指定目录，可以在当前终端手动设置 `PIP_CACHE_DIR`、`HF_HOME` 和 `TORCH_HOME`。这属于本机开发偏好，不是运行项目的必要步骤。

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

### 启动演示工作台

```powershell
python -m streamlit run demo/app.py --server.address 127.0.0.1 --server.port 8501
```

访问 `http://127.0.0.1:8501`。FastAPI 未启动时，页面会明确显示服务不可用，并使用仓库内的样例快照展示完整交互结构；实时演示流程见 [演示指南](docs/demo_guide.md)。

### 上传并入库文档

FastAPI、Milvus、Neo4j 和 LLM 均就绪后，可同步上传 TXT、Markdown 或 PDF：

```powershell
curl.exe -X POST http://127.0.0.1:8000/documents/upload -F "file=@data/raw/sample.txt"
```

接口使用内容 SHA-256 生成稳定 `document_id`，并完成切分、Embedding、图谱抽取和双库写入。默认文件上限为 20 MiB，重复内容返回 HTTP 409；需要重建相同内容的索引时显式传入 `overwrite=true`：

```powershell
curl.exe -X POST http://127.0.0.1:8000/documents/upload -F "overwrite=true" -F "file=@data/raw/sample.txt"
```

使用上传响应中的 `document_id` 可以删除对应的向量和图谱证据：

```powershell
curl.exe -X DELETE http://127.0.0.1:8000/documents/<document_id>
```

Streamlit 工作台也提供上传、覆盖重建和确认删除入口。内容变化后会生成新的 ID，需要删除旧 ID 再上传新文件；详细契约见 [API 设计](docs/api.md)。

## 当前已实现能力

- Streamlit GraphRAG 演示工作台，支持服务状态、文档生命周期管理、预设问题、方法对比、引用追溯、阶段耗时和局部图谱视图
- FastAPI 服务启动
- FastAPI lifespan 运行时资源复用，避免每次请求重复加载 Embedding、Reranker 和数据库客户端
- `/health` 健康检查接口
- `/ready` 运行就绪检查，返回 Embedding、Milvus、Neo4j、Reranker 和 LLM 的组件状态
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
- `POST /documents/upload` 同步文档入库 API，支持内容哈希、重复检测、覆盖重建、Milvus upsert、Neo4j MERGE 和阶段耗时
- `DELETE /documents/{document_id}` 文档删除 API，清理向量、文档关系和可安全删除的孤立实体
- 检索和问答分阶段耗时，覆盖 vector、graph、fusion、rerank 和 LLM
- GraphRAG Context Builder，用于统一组织向量上下文、图谱上下文、混合证据上下文和 LLM prompt
- Hybrid Retrieval Result Model，用于统一表示、融合排序并去重向量证据和图谱证据
- 可配置 Rerank 模块，用于在 QA prompt 和 citations 前对 Hybrid Evidence 进行二阶段重排序，支持 keyword overlap 与 BGE Reranker
- 检索与问答评估脚本，用于批量记录 `/retrieval/debug`、`/qa/ask` 和 `citations`
- 领域 PDF mini set，基于 6 篇异构图神经网络与鲁棒性论文构造 30 条评估问题
- Vector-only 与 GraphRAG hybrid 检索指标对比
- 基于 LLM 的实体关系抽取脚本
- Neo4j 图谱节点与关系写入脚本
- 基于 Neo4j 的图谱邻域检索脚本
- Neo4j 图结构导出脚本，用于生成后续 GNN 训练数据输入
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
- `fusion_weights`：本次请求实际使用的融合权重，便于复现实验配置
- `hybrid_results`：统一后的混合检索证据，包含 `fusion_score`，会按 `document_id + chunk_id` 合并重复证据，并按融合分降序返回，便于 QA rerank、GNN 和引用排序

`fusion_score` 默认由相关性分数和 rank 分数融合得到，可在 `.env` 中调整：

```env
FUSION_SCORE_WEIGHT=0.7
FUSION_RANK_WEIGHT=0.3
```

这两个权重会影响 `/retrieval/debug` 的 `hybrid_results` 初始排序；`/qa/ask` 会在混合证据基础上进一步执行 rerank，并将 rerank 后的证据用于 prompt 和 `citations`。默认使用轻量关键词 overlap reranker，也可以通过 `.env` 切换到 BGE Reranker：

```env
RERANKER_TYPE=bge
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

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
- `citations`：用于生成答案的混合证据引用，包含 `evidence_id`、`evidence_type`、`document_id`、`chunk_id`、`source` 和 `fusion_score`

问答生成时会先构造去重后的 `hybrid_results`，再对混合证据执行可配置 rerank，最后使用 rerank 后的证据上下文调用 LLM；响应保留原有来源字段，并额外返回可解释引用。

图谱召回会先从自然语言问题中抽取候选实体查询词。例如：

```text
What is GraphRAG?
  -> GraphRAG
```

这样可以避免完整问句直接匹配 Neo4j 节点名导致 `graph_sources` 为空。

### 检索与问答评估

启动 API 服务后，可以用样例问题集批量记录检索结果、答案和引用：

```powershell
python scripts/evaluate_retrieval.py --vector-top-k 3 --graph-top-k 5 --graph-max-depth 2 --qa-top-k 3
```

默认输入：

```text
data/eval/questions.sample.jsonl
```

默认输出：

```text
data/eval/retrieval_eval_results.jsonl
data/eval/retrieval_eval_summary.json
```

`retrieval_eval_results.jsonl` 每行包含：

- `run_config`：本次评估使用的 API 地址、TopK 参数和 fusion 权重
- `retrieval_debug`：`/retrieval/debug` 的完整响应
- `qa`：`/qa/ask` 的完整响应，包含 `citations`
- `metrics`：基于期望关键词的检索、引用、答案和延迟指标
- `summary`：召回数量、引用数量和 Top hybrid evidence 摘要

`retrieval_eval_summary.json` 会汇总本次运行的题目数、平均 evidence keyword recall、Recall@K、MRR、citation hit rate、answer hit rate 和平均延迟，便于快速对比不同 TopK 或 fusion 权重配置。

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

### 导出 GNN 图数据

在完成 Neo4j 导入后，可以导出节点和边数据，为后续 GNN/GAT 节点表示学习实验准备输入：

```powershell
python scripts/export_graph_dataset.py
```

默认输出：

```text
data/processed/graph_dataset.json
```

当前导出内容包括：

- `nodes`：图谱节点，包含 `node_id`、`name`、`node_type` 和 `description`
- `edges`：图谱边，包含 `source_id`、`target_id`、`relation_type`、证据来源和置信度
- `summary`：节点数量和边数量

## 开发路线与当前状态

### 阶段一：项目初始化（已完成）

- 创建 GitHub 仓库。
- 编写 README。
- 配置 Python 依赖。
- 配置 Git 忽略规则。

### 阶段二：Vector-only RAG Baseline（已完成）

- 实现文档读取。
- 实现文本切分。
- 接入 BGE-m3 Embedding。
- 接入 Milvus 向量检索。
- 实现基础问答接口。

### 阶段三：知识图谱构建（已完成）

- 使用 LLM 抽取实体和关系。
- 设计 Neo4j 图谱 Schema。
- 将三元组、来源和证据写入 Neo4j。

### 阶段四：GraphRAG 混合检索（已完成）

- 实现问题实体识别。
- 实现 Neo4j 子图遍历。
- 融合向量检索结果和图谱路径结果。
- 构建带证据的问答上下文。

### 阶段五：评估指标与项目展示完善（优先推进）

- 自动计算 Recall@K、MRR、citation hit、答案关键词命中和延迟指标。
- 构建领域 PDF mini set（当前包含 6 篇异构图相关论文和 30 条评估问题）。
- 对比 Vector-only 与 GraphRAG hybrid 检索指标。
- 维护 README 和 docs 的项目状态一致性。
- 形成可复现实验记录和面试讲解材料。

### 阶段六：Rerank 增强（部分完成）

- 已接入轻量关键词 overlap reranker。
- 已支持通过 `RERANKER_TYPE=bge` 接入 BGE Reranker。
- 已将 rerank 后的证据用于 QA prompt 和 citations。
- 后续需要补充 BGE/no-BGE、rerank/no-rerank 消融评估。

### 阶段七：GNN 节点表示增强（部分完成）

- 已从 Neo4j 导出节点和边，生成 GNN 数据集 JSON。
- 后续可使用实体文本 embedding 作为节点初始特征。
- 后续可探索 GraphSAGE/GAT 等模型学习结构感知节点表示。
- 后续再评估 GNN-assisted entity retrieval 对 Recall@K 和 MRR 的影响。

### 阶段八：完整消融实验（待实现）

- 对比 Vector-only RAG、GraphRAG hybrid、GraphRAG + lightweight Rerank。
- 在 GNN 节点表示完成后，再补充 GraphRAG + GNN 相关消融。

## 预期实验指标

- Recall@K
- MRR
- Answer Accuracy
- Faithfulness
- End-to-end Latency

## 项目状态

当前状态：GraphRAG 领域 PDF 问答闭环已完成，当前优先推进对比实验、项目展示和面试讲解材料完善。

已完成的核心链路包括 PDF/TXT/Markdown 文档处理、向量检索、图谱构建、混合证据融合、GraphRAG-aware 问答、citations、可配置 Rerank 和评估记录。当前已在 6 篇领域论文和 30 条自建问题上完成 baseline 评估，并记录 Vector-only 与 GraphRAG hybrid 检索对比。GNN/GAT 和更完整消融实验将作为后续增强分阶段接入。

当前已有领域 PDF mini set、演示问题清单和 Vector-only vs GraphRAG hybrid baseline，可用于本地演示、GitHub 展示和面试讲解。
