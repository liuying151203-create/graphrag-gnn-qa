# 演示指南

本文档用于面试、GitHub 展示或本地复盘时快速演示当前 GraphRAG 领域 PDF 问答闭环。

## 演示前置条件

确认本地服务和数据已经准备好：

```powershell
docker compose up -d neo4j etcd minio milvus
python -m uvicorn graphrag_gnn_qa.main:app --app-dir src --host 127.0.0.1 --port 8000 --reload
```

另开一个 PowerShell 终端检查 API：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

当前演示基于以下已入库数据：

- 6 篇异构图神经网络与鲁棒性相关 PDF。
- `data/processed/chunks.jsonl`：499 个文本块。
- `data/processed/graph_triples.jsonl`：499 条图谱抽取记录。
- Milvus collection：`rag_chunks`。
- Neo4j 知识图谱：已写入实体和关系。

如果需要重新构建数据，按顺序执行：

```powershell
python scripts/ingest_documents.py --input-dir data\raw\domain_papers --chunk-size 700 --chunk-overlap 100
python scripts/embed_chunks.py --batch-size 8
python scripts/load_embeddings_to_milvus.py --drop-existing
python scripts/extract_graph.py --resume --max-retries 3 --retry-delay 2 --progress-every 10
python scripts/load_graph_to_neo4j.py
```

## 推荐演示问题

以下问题来自 `data/eval/questions.domain_mini.jsonl`，在当前领域 PDF 语料中表现较稳定。

### 1. HAN 的研究问题

推荐 API：`/qa/ask`

```powershell
Invoke-RestMethod http://127.0.0.1:8000/qa/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"HAN 这篇论文主要解决异构图神经网络中的什么问题？","top_k":3}'
```

观察点：

- 答案应提到 heterogeneous graph 和 attention。
- `citations` 应指向 HAN 论文相关文本块。
- 适合开场展示 PDF 问答和可解释引用。

### 2. HeCo 的双视图对比学习

推荐 API：`/qa/ask`

```powershell
Invoke-RestMethod http://127.0.0.1:8000/qa/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"HeCo 的 co-contrastive learning 使用了哪两种视图？","top_k":3}'
```

观察点：

- 答案应提到 network schema view 和 meta-path view。
- 可解释 HeCo 如何通过跨视图对比学习处理标签稀缺问题。

### 3. RoHe 的恶意邻居过滤机制

推荐 API：`/qa/ask`

```powershell
Invoke-RestMethod http://127.0.0.1:8000/qa/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"RoHe 使用什么机制来过滤恶意邻居？","top_k":3}'
```

观察点：

- 答案应提到 attention purifier、topology 和 feature。
- 适合展示图谱与向量证据如何共同支持鲁棒性问题回答。

### 4. HetePR-BCD 的攻击预算和效果

推荐 API：`/qa/ask`

```powershell
Invoke-RestMethod http://127.0.0.1:8000/qa/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"HetePR-BCD 的攻击预算和性能下降结果在摘要中是如何描述的？","top_k":3}'
```

观察点：

- 答案应提到 15% perturbation budget 和 up to 32% F1 score degradation。
- 适合展示系统能从摘要中提取定量结果。

### 5. FastRo-HGCN 的数据集和指标

推荐 API：`/qa/ask`

```powershell
Invoke-RestMethod http://127.0.0.1:8000/qa/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"FastRo-HGCN 在哪些数据集上评估，并使用什么分类指标？","top_k":3}'
```

观察点：

- 答案应提到 IMDB、Yelp、ACM、Micro-F1 和 Macro-F1。
- 适合展示表格/实验段落中的信息检索。

### 6. HSeCo 的攻击设置

推荐 API：`/qa/ask`

```powershell
Invoke-RestMethod http://127.0.0.1:8000/qa/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"HSeCo 使用哪些攻击设置评估防御性能？","top_k":3}'
```

观察点：

- 答案应提到 HetePRBCD 和 HG Baseline。
- 适合展示跨论文鲁棒性实验设置。

### 7. RoHe 与 HSeCo 的对比

推荐 API：`/qa/ask`

```powershell
Invoke-RestMethod http://127.0.0.1:8000/qa/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"RoHe 和 HSeCo 都如何利用 meta-path 或语义信息提升鲁棒性？","top_k":3}'
```

观察点：

- 答案应同时涉及 RoHe 和 HSeCo。
- 适合展示跨文档比较型问题。

## 检索调试演示

如果想展示 GraphRAG 的中间过程，用 `/retrieval/debug`：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/retrieval/debug `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"RoHe 使用什么机制来过滤恶意邻居？","vector_top_k":3,"graph_top_k":5,"graph_max_depth":2}'
```

重点观察：

- `vector_results`：Milvus 召回的文本块。
- `graph_results`：Neo4j 返回的实体关系证据。
- `hybrid_results`：融合后的统一证据，包含 `fusion_score`。
- `fusion_weights`：实际生效的融合权重。

## 评估演示

完整 30 题评估耗时较长，演示时通常只展示已有结果：

```powershell
Get-Content data\eval\retrieval_eval_summary.json
```

当前领域 PDF mini set baseline：

- Vector-only evidence keyword recall：0.7161
- GraphRAG hybrid evidence keyword recall：0.7492
- Recall@K：1.0000
- MRR：1.0000
- Citation keyword hit rate：1.0000
- Answer hit rate：0.8000

如果需要重新跑：

```powershell
python scripts/evaluate_retrieval.py --input-file data\eval\questions.domain_mini.jsonl --vector-top-k 3 --graph-top-k 5 --graph-max-depth 2 --qa-top-k 3 --timeout 180 --progress-every 1
```

## 故障排查

### API 连接失败

现象：

```text
httpx.ConnectError: [WinError 10061]
```

处理：

```powershell
python -m uvicorn graphrag_gnn_qa.main:app --app-dir src --host 127.0.0.1 --port 8000 --reload
```

### Milvus 没有向量数据

现象：

- `/retrieve` 没有结果。
- `/retrieval/debug` 的 `vector_results` 为空。

处理：

```powershell
python scripts/embed_chunks.py --batch-size 8
python scripts/load_embeddings_to_milvus.py --drop-existing
```

### Neo4j 没有图谱节点

检查：

```powershell
docker exec graphrag-neo4j cypher-shell -u neo4j -p graphrag_neo4j_password "MATCH (n) RETURN count(n) AS node_count"
```

如果结果为 0：

```powershell
python scripts/extract_graph.py --resume --max-retries 3 --retry-delay 2 --progress-every 10
python scripts/load_graph_to_neo4j.py
```

### LLM 超时或网络错误

抽图时使用 retry/resume：

```powershell
python scripts/extract_graph.py --resume --max-retries 3 --retry-delay 2 --progress-every 10
```

评估时增加 timeout：

```powershell
python scripts/evaluate_retrieval.py --input-file data\eval\questions.domain_mini.jsonl --timeout 180 --progress-every 1
```

### 评估结果为空

检查 API 是否运行：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

检查问题集是否存在：

```powershell
(Get-Content data\eval\questions.domain_mini.jsonl).Count
```
