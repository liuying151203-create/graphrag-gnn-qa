# 项目后续开发与演示 Demo 计划

本文档用于维护 GraphRAG-GNN-QA 的后续开发优先级、阶段目标、验收标准和面试演示方案。计划以“可运行、可解释、可评估、可展示”为主线，优先完成能够形成工程闭环和面试展示价值的工作，再推进 GNN 等探索性增强。

## 1. 当前基线

当前项目已经完成 GraphRAG MVP 的核心链路：

- 支持 TXT、Markdown、PDF 文档解析和固定长度重叠切分。
- 使用 BGE-m3 生成文本块向量并写入 Milvus。
- 使用 LLM 按固定 Schema 抽取实体关系并写入 Neo4j。
- 支持 Milvus 向量检索和 Neo4j 图谱邻域检索。
- 使用 `HybridEvidence` 统一向量证据和图谱证据，完成融合排序与去重。
- 支持 keyword overlap reranker 和可配置 BGE Reranker。
- 提供 `/retrieve`、`/graph/retrieve`、`/retrieval/debug` 和 `/qa/ask` API。
- 返回答案、向量来源、图谱来源和 citations。
- 已有样例集、领域 PDF mini set、自动评估脚本和 baseline 结果。
- 已能从 Neo4j 导出节点和边，为后续 GNN 实验准备数据。

当前定位仍是 GraphRAG 工程 MVP。BGE Reranker 已完成工程接入，但尚缺正式消融结果；GNN 目前只有数据导出，不应描述为已经完成训练或提升检索效果。

## 2. 总体目标与优先级

后续开发按以下顺序推进：

1. 面试演示 Demo：把现有能力变成可直观操作和讲解的产品界面。
2. 运行时稳定性与性能：复用模型和数据库连接，减少重复初始化。
3. 文档上传与增量入库：实现端到端文档生命周期。
4. Rerank、融合和引用质量：用实验验证真实收益并增强可解释性。
5. 评估体系：从关键词 smoke test 升级为更可信的对比实验。
6. GNN 节点表示：在稳定 baseline 上验证是否带来可测量收益。

优先级原则：

- 面试官能直接体验的能力优先于大规模调参。
- 可复现的对比结果优先于未经验证的新模型堆叠。
- 先解决运行时重复加载、入库幂等和引用可信度，再扩展复杂算法。
- 新能力必须有测试、配置说明、运行命令和结果记录。

## 3. 面试演示 Demo 设计

### 3.1 演示目标

Demo 不是 README 的可视化版本，而是一个可以在 5 到 8 分钟内让面试官理解项目价值的 GraphRAG 工作台。演示应回答四个问题：

1. 系统解决了什么问题？
2. GraphRAG 与普通 Vector-only RAG 有什么差别？
3. 答案使用了哪些文本和图谱证据？
4. 项目是否有真实评估，而不仅是单次问答效果？

### 3.2 技术方案

第一版推荐使用 Streamlit：

- Python 技术栈与当前项目一致，开发和部署成本低。
- 通过 HTTP 调用现有 FastAPI，业务逻辑继续由后端负责。
- 不直接访问 Milvus、Neo4j 或 LLM，避免前端复制后端编排逻辑。
- 可快速实现聊天、Tabs、指标、表格和图谱可视化。

建议新增：

```text
demo/
  app.py
  api_client.py
  components.py
```

如后续需要公开部署或追求更完整的产品体验，再评估 React/Vite 前端。当前阶段不建议同时维护两套前端方案。

### 3.3 页面结构

第一屏直接进入可用工作台，不制作营销落地页。

#### 顶部状态区

显示：

- API、Milvus、Neo4j、LLM 的就绪状态。
- 当前数据集名称、文档数、chunk 数和图谱节点/关系数。
- 当前检索模式、reranker 类型和 TopK 参数。

状态检查失败时应显示明确原因，不能让用户在点击提问后才发现服务不可用。

#### 左侧控制区

提供：

- 预设问题选择。
- 检索模式：Vector-only、GraphRAG hybrid。
- Reranker：none、keyword、BGE。
- `vector_top_k`、`graph_top_k`、`graph_max_depth` 和 `rerank_top_k`。
- “恢复推荐配置”操作。

Demo MVP 暂不在这里暴露所有 `.env` 参数，避免界面变成配置后台。

#### 主问答区

展示：

- 用户问题。
- LLM 最终答案。
- 总延迟和 retrieval、rerank、generation 阶段耗时。
- 可点击的 citation 标识。
- 证据不足或调用失败时的明确状态。

点击 citation 后，应定位到对应 evidence，展示文档名、chunk、原文片段、证据类型和分数。

#### 证据分析区

使用 Tabs 组织：

- `Hybrid Evidence`：最终进入 prompt 的证据及 fusion/rerank 分数。
- `Vector Results`：Milvus 返回的文本块。
- `Graph Results`：Neo4j 返回的实体关系和 evidence。
- `Graph View`：以中心实体、关系和邻居组成小型子图。
- `Raw Response`：折叠展示 API JSON，供技术追问时使用。

图谱可视化只显示当前问题命中的局部子图，不加载完整知识图谱，确保信息密度和稳定性。

#### 方法对比区

提供一次点击完成的 Vector-only 与 GraphRAG 对比：

| 对比项 | Vector-only | GraphRAG hybrid |
|---|---|---|
| Top evidence | 文本块 | 文本块 + 图谱关系 |
| Evidence keyword recall | 当前问题结果 | 当前问题结果 |
| Citation 数量 | 当前结果 | 当前结果 |
| 延迟 | 当前结果 | 当前结果 |

这一视图是 Demo 的核心，因为它能把“用了 Neo4j”转化为可观察差异。

### 3.4 预设演示问题

准备 3 到 5 个稳定问题，不在面试现场临时碰运气：

1. 单文档事实问题：展示 PDF 解析、向量召回和 citation。
2. 方法与任务关系问题：展示 `Method -> SOLVES_TASK -> Task` 图谱证据。
3. 多实体或多跳问题：展示图谱邻域扩展的价值。
4. 跨论文对比问题：展示多个来源和混合证据组织。
5. 证据不足问题：展示系统会克制回答，而不是强行生成。

每个问题都应记录：

- 推荐参数。
- 预期答案要点。
- 预期命中的文档、chunk 或关系。
- 适合讲解的技术点。
- 已知不稳定因素和备用问题。

### 3.5 推荐演示脚本

面试时按以下顺序进行：

1. 用 30 秒介绍传统 RAG 的信息碎片问题和项目目标。
2. 展示系统状态和已入库领域论文，不现场重建完整索引。
3. 选择一个预设问题，先运行 Vector-only。
4. 切换 GraphRAG hybrid，对比新增图谱证据和最终上下文。
5. 点击 citation，展示答案如何追溯到 PDF chunk 或图谱关系。
6. 打开 Graph View，解释实体 Schema 和邻域检索。
7. 切换 keyword/BGE reranker，展示证据排序变化。
8. 最后展示领域 mini set 的 baseline 表格，并主动说明指标局限。

现场演示重点是解释设计决策和证据链，不建议现场执行 PDF 全量 embedding 或 LLM 图谱抽取，这些步骤耗时长且受网络影响。

### 3.6 Demo 数据与启动策略

Demo 使用固定、可复现的数据快照：

- 选择 2 到 6 篇领域 PDF。
- 固定 chunk 参数、Embedding 模型、fusion 权重和 TopK。
- 提前写入 Milvus 和 Neo4j。
- 保存问题清单和评估摘要。
- 启动后预热 Embedding 和 BGE Reranker。

建议提供统一启动命令：

```powershell
python scripts/start_demo.py
```

该脚本后续负责：

- 检查 Docker 服务。
- 检查 `.env` 和必要配置。
- 启动 FastAPI。
- 等待 `/health` 成功。
- 启动 Streamlit Demo。

如暂不实现启动脚本，至少在文档中提供两条明确命令分别启动后端和 Demo。

### 3.7 失败降级

Demo 必须考虑面试现场网络不稳定：

- Milvus 或 Neo4j 不可用：启动前阻断并提示，不进入半可用状态。
- LLM API 不可用：允许展示 retrieval/debug 和已保存的答案快照。
- BGE Reranker 加载失败：显示 fallback 状态并切换 keyword reranker。
- 图谱为空：隐藏 Graph View 并明确提示需要构建图谱。
- 请求超时：保留上一次成功结果，提供重试操作。

可为预设问题保存经过脱敏的 Demo snapshot，但界面必须明确区分“实时结果”和“已保存结果”。

### 3.8 Demo 验收标准

- 新环境按文档能在 10 分钟内启动已有数据的 Demo。
- 页面能显示服务状态、数据规模和当前配置。
- 至少 3 个预设问题可以稳定返回答案和 citations。
- 能直观看到 Vector-only 与 GraphRAG 的证据差异。
- 能展示当前问题对应的局部知识图谱。
- citation 能定位到具体文档和 chunk 原文。
- 热启动后的单次问答目标延迟控制在 10 秒以内；达不到时必须显示阶段耗时并记录瓶颈。
- LLM 或 BGE 失败时有清晰降级，不出现空白页面或未处理异常。
- 1366×768 和常见笔记本屏幕下不发生核心控件遮挡或文本溢出。

## 4. 分阶段开发计划

### Phase 1：Demo MVP 与运行时复用

优先级：最高。

目标：形成面试官可直接体验的界面，同时解决重复模型加载带来的延迟。

主要任务：

- 使用 FastAPI lifespan 初始化并复用 Embedding 模型、BGE Reranker、Milvus client 和 Neo4j driver。
- 应用关闭时释放数据库连接。
- 为 vector、graph、fusion、rerank 和 LLM 阶段增加耗时记录。
- 实现 Streamlit Demo 的状态区、预设问题、问答和证据 Tabs。
- 增加 Vector-only 与 GraphRAG 对比视图。
- 增加局部图谱可视化。
- 更新 `docs/demo_guide.md` 为实际 Demo 操作手册。

验收：

- 模型不会随每次 API 请求重复加载。
- 预设问题能完成完整演示脚本。
- 页面展示的配置、结果和后端响应一致。
- 有可复现的冷启动和热启动延迟记录。

### Phase 2：文档上传与增量入库

优先级：高。

目标：把命令行脚本整理为可复用服务，实现 `POST /documents/upload`。

主要任务：

- 抽取 `DocumentIngestionService`，统一解析、切分、Embedding、图谱抽取和写库流程。
- 设计稳定的 `document_id`，增加文件 hash 和重复上传检测。
- Milvus 支持按主键 upsert，并支持按 `document_id` 删除。
- Neo4j 支持按文档清理关系，并安全处理孤立实体。
- 增加入库状态：pending、processing、completed、partial_failed、failed。
- 第一版支持单文档同步上传；稳定后再引入后台任务和进度查询。
- Demo 增加上传入口，但预设问题仍使用预构建数据保证稳定。

验收：

- 上传 TXT、Markdown、PDF 后能够立即检索和问答。
- 重复上传不会制造重复 chunk 和关系。
- 任一步骤失败会返回阶段、错误信息和已完成统计。
- 能删除文档并清理对应向量与图谱证据。

### Phase 3：Rerank、融合与引用质量

优先级：高。

目标：验证各检索组件的真实收益，并让 citations 更可信。

主要任务：

- 对比 no-rerank、keyword rerank 和 BGE rerank。
- 对比不同 `RERANK_TOP_K` 和 fusion 权重。
- 评估 RRF 与当前 score/rank 融合方式。
- 为图谱结果增加路径长度、关系类型和 confidence 权重。
- 改进候选实体识别，支持别名和规范化实体名。
- 要求 LLM 输出结构化 citation IDs。
- 校验答案中的 citation 是否存在于实际上下文。
- 增加答案句子与证据对应关系或 unsupported claim 检查。

验收：

- `docs/experiments.md` 有 BGE/no-BGE 和 graph/no-graph 消融结果。
- 每次实验记录完整配置、模型版本、数据版本和阶段耗时。
- citations 表示答案实际引用证据，而不只是全部候选上下文。

### Phase 4：评估体系升级

优先级：中高。

目标：让项目结论经得起面试追问。

主要任务：

- 扩展真实 PDF 问题集，增加单跳、多跳、跨文档和证据不足问题。
- 保留关键词指标，同时增加 answer EM/F1、人工复核和可选 LLM judge。
- 将 retrieval、rerank、generation 延迟拆开统计。
- 避免评估脚本分别调用 debug 和 QA 导致重复检索。
- 增加失败率、fallback 次数和无答案准确率。
- 固化实验输出目录和命名，避免结果相互覆盖。

验收：

- 至少有一组独立于 `sample.txt` 的稳定评估集。
- 能生成 Vector-only、GraphRAG、GraphRAG + BGE 的统一对比表。
- 文档明确说明统计口径、限制和不可声称的结论。

### Phase 5：GNN 节点表示增强

优先级：中低，探索性。

前置条件：Phase 1 至 Phase 4 的 baseline 和评估稳定。

主要任务：

- 使用实体名称和描述生成节点初始语义特征。
- 构造 PyTorch Geometric 数据。
- 尝试 GraphSAGE 或 GAT 学习结构感知节点表示。
- 建立节点向量索引和 GNN-assisted entity retrieval。
- 将节点召回结果转换为 Hybrid Evidence。
- 对长尾实体、多跳问题和跨文档问题做消融。

验收：

- 与无 GNN baseline 使用相同数据和参数完成对比。
- 只有在 Recall@K、MRR 或人工质量上有稳定收益时，才把 GNN 作为项目主要成果。
- 如果没有收益，也应记录负结果和原因，不把训练完成等同于效果提升。

## 5. 横向工程优化

以下工作可随各阶段小步推进：

- 固定关键依赖版本，减少环境漂移。
- 将 PyMilvus 旧 ORM API 逐步迁移到当前 `MilvusClient` API。
- 统一日志格式，加入 request ID、document ID 和阶段耗时。
- 增加模型缓存目录和设备配置，明确 CPU/GPU 行为。
- 对外部 LLM 调用增加超时、重试、限流和错误分类。
- 避免在日志和 API 错误中泄露密钥、完整 prompt 或敏感文档。
- 增加文档大小、类型、页数和文本长度限制。
- 为核心服务增加启动就绪检查，而不仅是进程健康检查。

## 6. 暂不优先事项

- 不优先制作复杂营销首页。
- 不优先接入多个 LLM、Embedding 或向量数据库供应商。
- 不优先进行大规模 GNN 调参。
- 不在同步上传 MVP 稳定前引入复杂分布式任务队列。
- 不把当前关键词指标包装成通用答案准确率。
- 不为了展示技术数量而绕过现有模块边界重复实现检索逻辑。

## 7. 推荐提交拆分

建议每个阶段保持独立提交：

```text
docs: add project development and demo plan / 新增项目开发与演示计划
refactor(api): reuse model and database resources / 复用模型与数据库资源
feat(demo): add GraphRAG interview workbench / 新增 GraphRAG 面试演示工作台
feat(ingestion): add document ingestion service / 新增文档入库服务
feat(api): add document upload endpoint / 新增文档上传接口
test(eval): add reranker ablation evaluation / 新增重排器消融评估
feat(retrieval): validate structured citations / 校验结构化引用
feat(gnn): add GNN-assisted node retrieval / 新增 GNN 辅助节点召回
```

## 8. 文档维护关系

- 开发优先级和阶段状态变化：更新本文档和 `docs/architecture.md`。
- Demo 页面、启动方式或演示问题变化：更新 `docs/demo_guide.md`。
- API 路径、请求或响应变化：更新 `docs/api.md`。
- 新增目录、模块或脚本：更新 `docs/project_structure.md`。
- 指标、数据集或实验结果变化：更新 `docs/experiments.md`。
- 实体类型、关系类型或证据属性变化：更新 `docs/graph_schema.md`。
- 可展示结果和面试表述变化：更新 `README.md` 与 `docs/interview_notes.md`。
