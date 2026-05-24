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

用于接收用户问题，并返回结合混合检索证据上下文生成的答案和来源证据。

当前状态：已实现 GraphRAG-aware 版本，问答生成内部使用去重后的 Hybrid Evidence Context。

图谱召回会从自然语言问题中抽取候选实体查询词，例如 `What is GraphRAG?` 会额外使用 `GraphRAG` 查询 Neo4j。
向量检索结果和图谱检索结果会先转换为去重、融合排序后的混合证据，最终 LLM prompt 由 GraphRAG Context Builder 统一组织。响应中的 `sources` 和 `graph_sources` 保持兼容，并通过 `citations` 返回答案使用的混合证据引用。

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
  ],
  "citations": [
    {
      "evidence_id": "V1+G1",
      "evidence_type": "hybrid",
      "document_id": "sample",
      "chunk_id": "sample_chunk_0000",
      "source": "sample.txt",
      "fusion_score": 0.944
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

## 图谱检索接口

### `POST /graph/retrieve`

用于接收实体关键词或短查询，并返回 Neo4j 中匹配中心节点的邻域关系。

当前状态：已实现。

请求示例：

```json
{
  "query": "GraphRAG",
  "top_k": 5,
  "max_depth": 2
}
```

响应示例：

```json
{
  "query": "GraphRAG",
  "top_k": 5,
  "max_depth": 2,
  "results": [
    {
      "center_id": "Method:graphrag",
      "center_name": "GraphRAG",
      "center_type": "Method",
      "source_id": "Method:graphrag",
      "source_name": "GraphRAG",
      "source_type": "Method",
      "relation_type": "SOLVES_TASK",
      "target_id": "Task:question answering",
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

## 检索调试接口

### `POST /retrieval/debug`

用于展示当前问题的向量召回、图谱查询词、图谱召回结果和统一混合证据，便于调试 GraphRAG 检索效果。

当前状态：已实现 Vector + Graph + Hybrid Evidence 检索调试版本，`hybrid_results` 会按 `document_id + chunk_id` 去重，包含 `fusion_score` 并按融合分降序返回。

请求示例：

```json
{
  "query": "What is GraphRAG?",
  "vector_top_k": 3,
  "graph_top_k": 5,
  "graph_max_depth": 2
}
```

响应示例：

```json
{
  "query": "What is GraphRAG?",
  "vector_top_k": 3,
  "graph_top_k": 5,
  "graph_max_depth": 2,
  "graph_query_terms": ["What is GraphRAG?", "GraphRAG"],
  "vector_results": [
    {
      "score": 0.91,
      "chunk_id": "sample_chunk_0000",
      "document_id": "sample",
      "content": "GraphRAG connects vector search and graph traversal.",
      "source": "sample.txt",
      "file_name": "sample.txt",
      "file_type": "txt"
    }
  ],
  "graph_results": [
    {
      "center_id": "Method:graphrag",
      "center_name": "GraphRAG",
      "center_type": "Method",
      "source_id": "Method:graphrag",
      "source_name": "GraphRAG",
      "source_type": "Method",
      "relation_type": "SOLVES_TASK",
      "target_id": "Task:question answering",
      "target_name": "question answering",
      "target_type": "Task",
      "chunk_id": "sample_chunk_0000",
      "document_id": "sample",
      "source": "sample.txt",
      "evidence": "GraphRAG improves question answering.",
      "confidence": 0.9
    }
  ],
  "hybrid_results": [
    {
      "evidence_id": "V1+G1",
      "evidence_type": "hybrid",
      "rank": 1,
      "score": 0.91,
      "fusion_score": 0.937,
      "document_id": "sample",
      "chunk_id": "sample_chunk_0000",
      "source": "sample.txt",
      "content": "GraphRAG connects vector search and graph traversal.\nGraphRAG improves question answering.",
      "metadata": {
        "file_name": "sample.txt",
        "file_type": "txt",
        "center_name": "GraphRAG",
        "center_type": "Method",
        "relation_type": "SOLVES_TASK",
        "target_name": "question answering",
        "target_type": "Task",
        "evidence_ids": "V1,G1",
        "evidence_types": "vector_chunk,graph_relation",
        "source_evidence_count": "2"
      }
    }
  ]
}
```
