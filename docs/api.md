# API 设计

## 健康检查

### `GET /health`

用于检查后端服务是否正常运行。

响应示例：

```json
{
  "status": "ok",
  "app_name": "graphrag-gnn-qa",
  "environment": "development"
}
```

## 文档导入

### `POST /documents/upload`

计划用于上传文档并触发解析、切分、实体关系抽取、向量写入和图谱写入。

当前状态：待实现。

请求形式：

```text
multipart/form-data
```

计划响应：

```json
{
  "document_id": "doc_xxx",
  "filename": "example.pdf",
  "chunk_count": 32,
  "entity_count": 48,
  "relation_count": 76
}
```

## 问答接口

### `POST /qa/ask`

计划用于接收用户问题，并返回答案、来源证据和图谱路径。

当前状态：待实现。

请求示例：

```json
{
  "question": "哪些论文使用图注意力网络解决信息抽取任务？",
  "top_k": 5,
  "use_graph": true,
  "use_gnn": true
}
```

计划响应：

```json
{
  "answer": "...",
  "sources": [
    {
      "document": "example.pdf",
      "chunk_id": "chunk_001",
      "evidence": "..."
    }
  ],
  "graph_paths": [
    "Paper A -> USES_METHOD -> GAT -> SOLVES_TASK -> Information Extraction"
  ],
  "latency_ms": 1530
}
```

## 检索调试接口

### `POST /retrieval/debug`

计划用于展示向量召回、图谱召回、GNN 节点召回和 Rerank 结果。

当前状态：待实现。
