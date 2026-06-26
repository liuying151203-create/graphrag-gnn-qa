# AGENTS.md

本文件记录本项目中 AI coding assistant 和开发者协作时需要长期遵守的习惯、规矩和交付标准。进入本仓库工作时，优先阅读本文件；涉及更细的流程、Git 规范和文档维护细节时，再参考 `docs/development_workflow.md`。

## 沟通习惯

- 默认使用中文沟通。
- 回复使用 Markdown，结构清晰，避免大段无层次文本。
- 优先直接推进明确需求，不只停留在建议层面。
- 不做与当前任务无关的重构、格式化或文件移动。
- 发现已有未提交改动时，不擅自回滚；先判断是否与当前任务相关。
- 简历、面试和可展示工程闭环的优先级高于大规模调参和大数据集实验。

## 任务执行规则

- 中等以上工程任务开始前，必须先做简短规划，说明目标、预计修改、验证方式和风险点。
- 执行过程中保持小步修改，优先沿用现有模块、接口和文档风格。
- 修改代码前先查现状，重点查看相关源码、测试、README 和 `docs/` 文档，避免重复造轮子。
- 配置项变更必须同步 `.env.example`、配置校验和相关文档。
- API 响应结构变更必须同步测试和 `docs/api.md`。
- 检索、QA、评估相关变更必须保留可复现实验配置和输出说明。

## 长期维护文档

以下文档属于项目长期维护内容，功能或阶段状态变化后需要检查是否同步：

- `README.md`：项目首页，面向 GitHub 访问者和面试官，维护项目目标、运行方式、核心能力、数据流程、API 示例、评估方式和文档入口。
- `docs/architecture.md`：系统架构和路线图，维护当前定位、模块职责、已完成/部分完成/待实现状态和面试讲解定位。
- `docs/api.md`：接口契约，维护路径、请求示例、响应示例、字段含义、当前实现状态和兼容性说明。
- `docs/project_structure.md`：目录和模块职责，新增文件、目录、脚本或模块时需要同步。
- `docs/experiments.md`：实验设计和结果记录，评估脚本、指标、问题集或实测结果变化时需要同步。
- `docs/graph_schema.md`：知识图谱 schema，实体类型、关系类型或证据属性变化时需要同步。

## 验证规则

- 代码变更后，按影响范围运行可复现测试；优先使用：

```powershell
python -m pytest
```

- 范围较小的代码变更可以运行对应测试文件，例如：

```powershell
python -m pytest tests/test_xxx.py
```

- 文档变更可以不跑完整测试，但需要至少检查 diff 或格式问题：

```powershell
git diff --check
```

- 如果无法运行测试或验证命令，完成汇报中必须说明原因。

## 完成汇报规则

每次任务完成后都要汇报，至少包含：

- 已完成内容总结。
- 修改了哪些文件。
- 执行了哪些验证命令，以及结果。
- 未执行的验证及原因。
- 如有文件变动，给出建议的 Git 提交命令。

推荐汇报格式：

```text
已完成：
- ...

修改文件：
- ...

验证：
- 已运行：...
- 结果：...

未运行：
- ...

Git 提交建议：
- git add ...
- git commit -m "type: English summary / 中文简短说明"
```

## Git 提交规则

- 提交前建议检查：

```powershell
git status --short
git diff --stat
git diff --check
```

- 提交粒度要小而清晰，一个提交只覆盖一个功能闭环、一个 bug 修复、一组测试补充或一次文档更新。
- 提交信息使用规范的双语格式，采用简化版 Conventional Commits：

```text
type: English summary / 中文简短说明
```

- 常用类型：
  - `feat`：新增功能或能力
  - `fix`：修复 bug
  - `docs`：文档更新
  - `test`：测试新增或调整
  - `refactor`：不改变行为的结构调整
  - `chore`：项目配置、依赖、初始化等维护

示例：

```text
docs: add agent collaboration guide / 新增协作规则文档
feat: add retrieval evaluation script / 新增检索评估脚本
fix: handle QA API external service errors / 处理问答接口外部服务错误
test: add fusion weight validation tests / 新增融合权重校验测试
refactor: extract GraphRAG context builder / 抽取 GraphRAG 上下文构建模块
```

如果当前任务产生了文件变动，最终汇报中应给出类似命令：

```powershell
git add AGENTS.md
git commit -m "docs: add agent collaboration guide / 新增协作规则文档"
```

