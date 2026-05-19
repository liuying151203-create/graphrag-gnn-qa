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

用于接收用户问题，并返回结合向量检索上下文和图谱关系上下文生成的答案和来源证据。

当前状态：已实现 GraphRAG-aware 版本。

请求示例：

```json
{
  "question": "What is GraphRAG?",
  "top_k": 3
}
```

响应示例：

```json
{
  "question": "What is GraphRAG?",
  "answer": "GraphRAG combines graph-based retrieval with text generation.",
  "sources": [
    {
      "chunk_id": "sample_chunk_0000",
      "document_id": "sample",
      "source": "sample.txt",
      "file_name": "sample.txt",
      "score": 0.91,
      "content": "GraphRAG connects vector search and graph traversal."
    }
  ],
  "graph_sources": [
    {
      "center_name": "GraphRAG",
      "center_type": "Method",
      "source_name": "GraphRAG",
      "source_type": "Method",
      "relation_type": "SOLVES_TASK",
      "target_name": "question answering",
      "target_type": "Task",
      "chunk_id": "sample_chunk_0000",
      "document_id": "sample",
      "source": "sample.txt",
      "evidence": "GraphRAG improves question answering.",
      "confidence": 0.9
    }
  ]
}
```

## 向量检索接口

### `POST /retrieve`

用于接收用户问题，并返回 Milvus 中最相关的 TopK 文本块。

当前状态：已实现。

请求示例：

```json
{
  "query": "What is GraphRAG?",
  "top_k": 3
}
```

响应示例：

```json
{
  "query": "What is GraphRAG?",
  "top_k": 3,
  "results": [
    {
      "score": 0.91,
      "chunk_id": "sample_chunk_0000",
      "document_id": "sample",
      "content": "GraphRAG connects vector search and graph traversal.",
      "source": "sample.txt",
      "file_name": "sample.txt",
      "file_type": "txt"
    }
  ]
}
```

## 检索调试接口

### `POST /retrieval/debug`

计划用于展示向量召回、图谱召回、GNN 节点召回和 Rerank 结果。

当前状态：待实现。
