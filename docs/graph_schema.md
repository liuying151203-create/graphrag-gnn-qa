# 知识图谱 Schema 设计

## 设计目标

图谱 Schema 面向科研论文和技术文档问答场景，用于表达论文、作者、机构、方法、任务、数据集、指标和概念之间的关系。

## 节点类型

所有实体节点均包含以下通用属性：

- `id`：由实体类型和规范化名称组成的唯一标识。
- `name`：实体显示名称。
- `description`：可选实体描述。
- `document_ids`：提及该实体的文档 ID 列表，用于共享实体保留和按文档清理。

### `Paper`

表示论文或技术文档。

核心属性：

- `id`
- `title`
- `year`
- `source`
- `abstract`

### `Author`

表示作者。

核心属性：

- `id`
- `name`

### `Institution`

表示机构。

核心属性：

- `id`
- `name`

### `Method`

表示方法、模型或算法。

核心属性：

- `id`
- `name`
- `description`

### `Task`

表示研究任务。

核心属性：

- `id`
- `name`

### `Dataset`

表示数据集。

核心属性：

- `id`
- `name`

### `Metric`

表示评价指标。

核心属性：

- `id`
- `name`

### `Concept`

表示通用技术概念。

核心属性：

- `id`
- `name`
- `description`

## 关系类型

### `AUTHORED_BY`

表示论文由某作者撰写。

```text
(:Paper)-[:AUTHORED_BY]->(:Author)
```

### `AFFILIATED_WITH`

表示作者属于某机构。

```text
(:Author)-[:AFFILIATED_WITH]->(:Institution)
```

### `PROPOSED_METHOD`

表示论文提出某方法。

```text
(:Paper)-[:PROPOSED_METHOD]->(:Method)
```

### `USES_METHOD`

表示论文或方法使用另一方法。

```text
(:Paper)-[:USES_METHOD]->(:Method)
(:Method)-[:USES_METHOD]->(:Method)
```

### `SOLVES_TASK`

表示方法用于解决某任务。

```text
(:Method)-[:SOLVES_TASK]->(:Task)
```

### `USES_DATASET`

表示论文或方法使用某数据集。

```text
(:Paper)-[:USES_DATASET]->(:Dataset)
(:Method)-[:USES_DATASET]->(:Dataset)
```

### `EVALUATED_BY`

表示论文或方法使用某指标评估。

```text
(:Paper)-[:EVALUATED_BY]->(:Metric)
(:Method)-[:EVALUATED_BY]->(:Metric)
```

### `RELATED_TO`

表示两个概念、方法或任务之间存在一般关联。

```text
(:Concept)-[:RELATED_TO]->(:Concept)
(:Method)-[:RELATED_TO]->(:Concept)
```

## 关系通用属性

每条关系建议保留以下属性：

- `source`
- `document_id`
- `chunk_id`
- `evidence`
- `confidence`

这样可以在回答时返回证据来源，降低大模型幻觉风险。

## 文档归属与清理

实体按 `type + normalized name` 全局合并，因此同一实体可以被多篇文档共享。每次图谱写入会幂等追加实体的 `document_ids`；关系把 `document_id + chunk_id` 纳入 `MERGE` 身份，避免不同文档的同类证据相互覆盖。

删除文档时按以下顺序处理：

1. 删除 `relationship.document_id` 匹配的关系。
2. 从相关实体的 `document_ids` 中移除当前文档。
3. 仅删除 `document_ids` 为空且没有任何剩余关系的实体。

旧数据如果尚无 `document_ids` 属性，不会被孤立实体清理逻辑直接删除；通过覆盖重建重新写入后会获得文档归属信息。
