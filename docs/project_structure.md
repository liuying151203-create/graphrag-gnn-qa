# 项目结构说明

本文档用于说明当前项目目录结构、各模块职责，以及当前已经实现的数据处理流程，方便后续复盘和扩展。

## 当前目录结构

```text
graphrag-gnn-qa/
├── data/
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
│   └── ingest_documents.py
├── src/
│   └── graphrag_gnn_qa/
│       ├── __init__.py
│       ├── config.py
│       ├── main.py
│       ├── api/
│       │   ├── __init__.py
│       │   └── routes_health.py
│       └── ingestion/
│           ├── __init__.py
│           ├── document_loader.py
│           └── text_splitter.py
├── tests/
│   ├── test_document_loader.py
│   ├── test_health.py
│   ├── test_ingest_documents.py
│   └── test_text_splitter.py
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

当前已经注册：

```text
GET /health
```

### `config.py`

配置管理模块，基于 `pydantic-settings` 从 `.env` 读取配置。

后续 Neo4j、Milvus、LLM、Embedding 和检索参数都会统一从这里读取。

### `api/`

API 路由目录。

当前包含：

- `routes_health.py`：健康检查接口

后续计划增加：

- `routes_documents.py`：文档上传和导入接口
- `routes_qa.py`：问答接口
- `routes_retrieval.py`：检索调试接口

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

## `data/`

数据目录。

### `data/raw/`

存放原始文档，例如论文 PDF、Markdown 文档或 TXT 文本。

真实数据文件不会提交到 GitHub。

### `data/processed/`

存放处理后的中间结果，例如：

```text
chunks.jsonl
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
```

## 后续扩展方向

下一阶段会在当前 `chunks.jsonl` 基础上继续实现：

```text
chunks.jsonl
  -> BGE-m3 Embedding
  -> Milvus 向量存储
  -> Vector Search Baseline
```

之后继续扩展：

```text
chunks.jsonl
  -> LLM 实体关系抽取
  -> Neo4j 知识图谱
  -> GraphRAG 混合检索
  -> GNN 节点表示增强
```
