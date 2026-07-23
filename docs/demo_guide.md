# 演示指南

本文档用于面试、GitHub 展示和本地复盘时快速演示当前 GraphRAG 闭环。第一版 Demo 使用 Streamlit 构建研究工作台，通过 HTTP 调用 FastAPI，不直接连接 Milvus、Neo4j 或 LLM。

## 演示模式

Demo 支持两种清晰区分的展示方式：

- 实时模式：FastAPI 及依赖服务可用时，调用 `/ready`、`/retrieval/debug` 和 `/qa/ask`。
- 快照模式：后端未启动时，默认展示 `data/demo/sample_snapshot.json`，页面会明确提示“已保存演示快照，不是实时请求”。

快照用于保证新环境和面试现场可以先查看完整界面，不代表当前数据库或模型的实时输出。实时请求成功后，页面会切换到本次结果；请求整体失败时会保留上一份成功结果。

## 快速启动

### 1. 安装依赖

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 2. 启动实时后端

需要实时检索和问答时，先启动基础服务：

```powershell
docker compose up -d neo4j etcd minio milvus
```

再启动 FastAPI：

```powershell
python -m uvicorn graphrag_gnn_qa.main:app --app-dir src --host 127.0.0.1 --port 8000
```

检查运行状态：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
```

`/health` 只表示 API 进程存活；`/ready` 会检查 Embedding、Milvus、Neo4j、Reranker 和 LLM。实时完整问答要求这些组件处于可用状态。

### 3. 启动 Demo

另开一个 PowerShell 终端：

```powershell
python -m streamlit run demo/app.py --server.address 127.0.0.1 --server.port 8501
```

访问：

```text
http://127.0.0.1:8501
```

只想检查界面时可以省略后端和 Docker 步骤，Demo 会显示默认样例快照以及后端不可用状态。

## 页面说明

### 服务与数据状态

页面顶部展示 API、Embedding、Milvus、Neo4j、Reranker 和 LLM 状态。状态来自 `/ready`，不会根据前端推测。

数据指标从本地文件统计：

- 文档：`data/raw/` 下支持的 TXT、Markdown 和 PDF 数量。
- Chunks：`data/processed/chunks.jsonl` 记录数。
- 实体与关系：`data/processed/graph_triples.jsonl` 中去重实体数和关系数。

### 文档管理

`文档管理` 折叠区提供端到端入库和删除操作：

- 上传 TXT、Markdown 或 PDF，并展示文档 ID、chunk、实体、关系和总耗时。
- 勾选覆盖选项后，重建相同内容对应的索引；内容变化会生成新的文档 ID。
- 输入文档 ID 并显式确认后，删除 Milvus chunks、Neo4j 文档关系和可安全清理的孤立实体。

上传依赖 Embedding、LLM、Milvus 和 Neo4j 全部就绪，删除不依赖 LLM。页面顶部的数据规模来自本地预处理文件，不会随着 API 数据库操作即时变化；上传或删除结果以操作区返回的统计为准。

### 查询配置

侧边栏提供：

- API 地址。
- 预设问题和自定义问题。
- Vector TopK。
- Graph TopK。
- Graph Depth。

Reranker 类型由后端 `.env` 的 `RERANKER_TYPE` 决定，当前页面只展示后端实际就绪状态，不提供仅影响界面、不影响 API 的伪切换。修改 Reranker 后需要重启 FastAPI。

### 答案与耗时

运行分析后，主区域展示：

- LLM 答案。
- citations 数量和混合证据数量。
- vector、graph、fusion、rerank、LLM 分阶段耗时。
- 可定位到具体 `document_id`、`chunk_id` 和证据内容的 citation。

### 方法对比与证据

`Vector-only 与 GraphRAG` 表格并列展示证据数、Top 证据、Top 分数、引用数和检索耗时。下方五个页签用于展开中间过程：

- `Hybrid Evidence`：融合、去重和排序后的证据。
- `Vector Results`：Milvus 文本块召回。
- `Graph Results`：Neo4j 实体关系召回。
- `Graph View`：当前问题命中的局部有向图。
- `Raw Response`：后端原始响应和客户端耗时。

## 默认数据

仓库内可复现的默认展示数据包括：

- `data/raw/sample.txt`。
- `data/eval/questions.sample.jsonl`。
- `data/demo/sample_snapshot.json`。

`data/eval/questions.domain_mini.jsonl` 提供论文领域预设问题。真实 PDF、生成的 chunks、embeddings、图谱数据和评估输出通常不会提交到 Git；如果本机没有对应领域论文索引，这些问题只能用于查看问题设计，不能保证实时命中。

需要重新构建本地索引时，依次执行：

```powershell
python scripts/ingest_documents.py
python scripts/embed_chunks.py --batch-size 8
python scripts/load_embeddings_to_milvus.py --drop-existing
python scripts/extract_graph.py --resume --max-retries 3 --retry-delay 2
python scripts/load_graph_to_neo4j.py
```

## 推荐演示脚本

1. 用 30 秒说明传统 RAG 在实体关系和多跳证据上的局限。
2. 展示顶部服务状态、数据规模和当前 Reranker。
3. 选择预设问题并运行分析。
4. 用对比表解释 Vector-only 与 GraphRAG hybrid 的证据差异。
5. 打开 citation，追溯到文档 chunk。
6. 在 Graph Results 和 Graph View 中解释实体关系与邻域检索。
7. 用耗时图说明检索、重排和 LLM 的性能瓶颈。
8. 展开文档管理区，展示上传、覆盖重建和确认删除能力；现场可使用小型 TXT 演示，避免全量 PDF 处理。
9. 最后说明快照降级、当前评估边界，以及后续后台任务和进度查询计划。

面试现场不建议执行全量 PDF embedding 或 LLM 图谱抽取，这些步骤耗时较长并依赖外部服务。

## 评估演示

已有领域数据和后端时可以运行：

```powershell
python scripts/evaluate_retrieval.py --input-file data/eval/questions.domain_mini.jsonl --vector-top-k 3 --graph-top-k 5 --graph-max-depth 2 --qa-top-k 3 --timeout 180
```

评估输出位于 `data/eval/retrieval_eval_results.jsonl` 和 `data/eval/retrieval_eval_summary.json`。展示指标时应同时说明数据版本、运行配置以及关键词指标的局限，不把样例结果表述为通用准确率。

## 故障排查

### API 显示 DOWN

确认 FastAPI 已启动，并检查侧边栏 API 地址：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### Milvus 或 Neo4j 不可用

```powershell
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/ready
```

如果容器未运行，先启动 Docker Desktop，再执行 `docker compose up -d neo4j etcd minio milvus`。

### LLM 未配置

检查本地 `.env` 中的 `LLM_API_KEY` 和 `LLM_BASE_URL`。未配置 LLM 时仍可检查检索接口和默认快照，但实时 `/qa/ask` 会返回不可用状态。

### 页面仍显示旧版本

开发阶段可直接重启 Streamlit；正式演示建议关闭自动文件监听以减少额外开销：

```powershell
python -m streamlit run demo/app.py --server.address 127.0.0.1 --server.port 8501 --server.fileWatcherType none
```
