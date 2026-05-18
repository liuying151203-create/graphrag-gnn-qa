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
