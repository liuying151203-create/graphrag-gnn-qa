# 项目开发工作流与个人偏好

本文档沉淀本项目开发过程中的可复用步骤、协作偏好、文档维护规则、Git 管理方式和验证方法。后续开发新功能、修复问题或准备简历面试展示时，可以优先参考本文档。

## 适用场景

- 新增功能或修改现有模块
- 修复 bug 或排查异常
- 更新 README 和 `docs/` 文档
- 设计长期路线图和阶段性目标
- 准备实验、验证、提交和复盘
- 与 AI coding assistant 协作开发

## 个人偏好

- 默认使用中文沟通。
- 回复使用 Markdown，结构清晰，避免大段无层次文本。
- 开始中等以上任务前，先给出简短 TODO 或计划。
- 执行过程中及时更新计划状态。
- 优先直接实现明确需求，不只给建议。
- 避免无关改动，保持每次修改范围清晰。
- 简历/面试优先级高于大规模调参和大数据集实验。
- 每次完成后给出变更总结、验证方式、测试结果和必要的 Git 建议。
- 下一次提交信息不需要在每次总结里固定提前给出；只有当当前任务已经产生未提交变更、用户要求提交，或明确需要规划提交粒度时再给出。
- 如果只改文档，可以不跑完整测试，但需要做文档 diff 或格式检查。
- 如果改代码，需要给出实际可复现的验证命令。

## 推荐开发流程

### 1. 明确任务边界

开始前先确认：

- 本次目标是什么
- 是否是功能开发、bug 修复、文档维护、实验分析或项目规划
- 涉及哪些模块
- 是否需要修改 API、配置、测试或文档
- 是否影响已有行为和兼容性

推荐输出形式：

```text
本次目标：...
预计修改：...
验证方式：...
风险点：...
```

### 2. 制定短计划

中等以上任务建议维护 2 到 5 个步骤的计划。

示例：

```text
1. 确认当前代码和文档状态
2. 实现核心改动
3. 补充测试和文档
4. 运行验证并总结
```

计划应该是阶段性目标，不要拆成过细的“打开文件”“读某一行”。

### 3. 先查现状，再动代码

修改前优先确认已有实现，避免重复造轮子。

常见检查点：

- README 是否已有相关说明
- `docs/architecture.md` 是否定义了目标和路线
- `docs/api.md` 是否定义接口契约
- `docs/project_structure.md` 是否说明模块职责
- `docs/experiments.md` 是否说明实验和评估方式
- `tests/` 是否已有相关测试
- `.env.example` 是否需要新增配置项

### 4. 小步实现

实现时遵循：

- 一次只解决一个清晰问题
- 优先改核心抽象，再更新调用方
- API 响应结构变更要同步测试和文档
- 配置项变更要同步 `.env.example` 和配置校验
- 检索、QA、评估相关变更要保证可复现实验记录
- 不做与当前任务无关的重构

### 5. 补充测试

代码改动尽量补测试。

常见测试类型：

- 配置测试：验证默认值、非法值和边界条件
- 单元测试：验证核心函数行为
- API 测试：验证请求、响应和依赖注入
- 脚本测试：验证输入输出结构
- 回归测试：覆盖本次修复的问题

### 6. 同步文档

功能变更完成后，检查是否需要同步：

- `README.md`
- `docs/api.md`
- `docs/architecture.md`
- `docs/project_structure.md`
- `docs/experiments.md`
- `.env.example`

文档不能只写最终结论，也要说明如何运行、如何验证、输出在哪里。

### 7. 实际验证

完成后必须给出真实验证方法。

常见验证命令：

```powershell
python -m pytest
```

```powershell
python -m pytest tests/test_xxx.py
```

```powershell
uvicorn graphrag_gnn_qa.main:app --app-dir src --reload
```

```powershell
python scripts/evaluate_retrieval.py --vector-top-k 3 --graph-top-k 5 --graph-max-depth 2 --qa-top-k 3
```

只改文档时可使用：

```powershell
git diff --check
```

## 文档维护规则

### README.md

README 面向 GitHub 访问者和面试官，应该维护：

- 项目目标
- 技术栈
- 当前已实现能力
- 本地运行方式
- 核心 API 示例
- 数据处理流程
- 评估方式
- 文档入口

当新增重要文档时，需要同步 README 的“项目文档”列表。

### docs/architecture.md

架构文档用于说明系统设计和路线图，应该维护：

- 当前项目定位
- 总体架构流程
- 核心模块职责
- 已完成、部分完成、待实现状态
- 长期路线图
- 面试讲解定位

当项目阶段变化时，优先更新此文件。

### docs/api.md

API 文档用于维护接口契约，应该维护：

- 请求路径和方法
- 当前实现状态
- 请求示例
- 响应示例
- 字段含义
- 兼容性说明

当 API 响应新增字段时，必须同步此文件。

### docs/project_structure.md

项目结构文档用于解释目录和模块职责，应该维护：

- 最新目录结构
- 每个模块的职责
- 脚本输入输出
- 核心数据流
- 新增模块的设计位置

当新增文件、目录或脚本时，需要检查是否更新。

### docs/experiments.md

实验文档用于沉淀评估设计，应该维护：

- 数据集设计
- 对比方法
- 指标定义
- 运行方式
- 输出结构
- 结果记录模板
- 消融实验关注点

实验脚本输出结构变化时，需要同步此文件。

## Git 管理规则

### 提交前检查

提交前建议执行：

```powershell
git status --short
```

```powershell
git diff --stat
```

```powershell
git diff --check
```

如改了代码，按影响范围运行测试：

```powershell
python -m pytest tests/test_xxx.py
```

或完整测试：

```powershell
python -m pytest
```

### 提交粒度

推荐小而清晰的提交。

适合单独提交的类型：

- 一个功能闭环
- 一个 bug 修复
- 一次文档更新
- 一组测试补充
- 一次路线图或架构文档更新

避免把无关内容混在一个 commit 中。

### 提交信息格式

推荐使用简化版 Conventional Commits：

```text
type: English summary / 中文简短说明
```

其中 `type` 用于表示提交类型，summary 用英文概括动作，斜杠后用中文补充说明，方便后续复盘和面试讲解。

常用类型：

- `feat`：新增功能或能力
- `fix`：修复 bug
- `docs`：文档更新
- `test`：测试新增或调整
- `refactor`：不改变行为的结构调整
- `chore`：项目配置、依赖、初始化等杂项维护

优先保持格式统一、描述简短、范围清晰。

示例：

```text
feat: add retrieval evaluation script / 新增检索评估脚本
feat: make hybrid fusion weights configurable / 支持混合检索融合权重配置
feat: add QA citations from hybrid evidence / 新增问答混合证据引用
fix: handle QA API external service errors / 处理问答接口外部服务错误
refactor: extract GraphRAG context builder / 抽取 GraphRAG 上下文构建模块
```

如果是文档类提交：

```text
docs: add project structure guide / 新增项目结构说明文档
docs: update architecture status and roadmap / 更新架构状态和路线图
docs: add development workflow guide / 新增项目开发工作流文档
```

如果是测试类提交：

```text
test: add fusion weight validation tests / 新增融合权重校验测试
test: add retrieval debug response tests / 新增检索调试响应测试
```

### 提交后检查

提交后执行：

```powershell
git status --short
```

确认工作区干净。

如需要推送：

```powershell
git push
```

## 长期项目规划方式

长期项目不要只按功能堆叠，建议维护阶段目标。

当前项目推荐路线：

### Stage 1：GraphRAG MVP

状态：已完成。

核心目标：

- 文档读取和切分
- 向量检索
- 图谱构建与检索
- 混合证据融合
- GraphRAG QA
- citations
- 评估记录

### Stage 2：评估指标与项目展示完善

状态：优先推进。

核心目标：

- 自动计算 Recall@K、MRR、citation hit 等指标
- 扩充小规模评估集
- 维护 README 和 docs
- 形成可讲解的实验结果

### Stage 3：Rerank 增强

状态：待实现。

核心目标：

- 新增 reranker 模块
- 对 Hybrid Evidence 二阶段排序
- 比较 rerank 前后效果

### Stage 4：GNN 节点表示增强

状态：待实现。

核心目标：

- 从 Neo4j 导出图结构
- 构造 GNN 训练数据
- 训练 GAT 节点表示
- 接入 GNN-assisted retrieval

### Stage 5：完整消融实验

状态：待实现。

核心目标：

- Vector-only RAG
- GraphRAG
- GraphRAG + Rerank
- GraphRAG + GNN
- GraphRAG + GNN + Rerank
- 不同 fusion 权重对比

## 验证与汇报模板

任务完成后推荐按以下格式汇报：

```text
已完成：
- ...

修改文件：
- ...

验证结果：
- 已运行：...
- 结果：...

未运行的验证：
- 原因：...

Git 建议：
- git add ...
- git commit -m "..."

下一步建议：
- ...
```

## AI 协作提示词模板

### 新功能开发

```text
请先检查当前实现和相关文档，给出简短计划，然后实现这个功能。
要求同步测试、README/docs 和实际验证方法，不要做无关改动。
```

### 文档更新

```text
请根据当前代码和项目状态更新相关文档，重点保证 README、architecture、api、project_structure、experiments 之间一致。
最后给出变更摘要和 git 提交建议。
```

### Git 提交前检查

```text
请检查 git 状态，查看 diff，总结未提交内容，并按合理粒度提交。
提交前如果有代码变更，请建议或执行必要测试。
```

### 路线图规划

```text
请从当前代码、记忆和 docs 中梳理已完成、部分完成、待实现模块，并按简历/面试优先级给出下一步路线图。
```

### 实验和评估

```text
请检查评估脚本和输出结构，说明当前实验的意义、局限、可复现配置，以及下一步应该补哪些指标或样例。
```

## 当前项目的优先级原则

- 先保证 GraphRAG MVP 链路完整、清晰、可验证。
- 简历展示优先于大规模调参。
- 结构完整优先于追求单次实验数值。
- 每个新增能力都要有测试和文档闭环。
- 实验结果要记录配置，避免不可复现。
- GNN 和 Rerank 是重要增强，但可以分阶段实现。

## 检查清单

### 开发前

- 是否明确目标和范围
- 是否确认相关文件和模块
- 是否知道要补哪些测试
- 是否知道要同步哪些文档

### 开发中

- 是否保持改动聚焦
- 是否避免无关重构
- 是否维护配置兼容性
- 是否同步调用方和测试

### 开发后

- 是否运行必要验证
- 是否更新 README/docs
- 是否检查 git diff
- 是否按合理粒度提交
- 是否记录下一步建议
