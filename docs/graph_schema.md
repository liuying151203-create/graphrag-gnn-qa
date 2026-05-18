# 知识图谱 Schema 设计

## 设计目标

图谱 Schema 面向科研论文和技术文档问答场景，用于表达论文、作者、机构、方法、任务、数据集、指标和概念之间的关系。

## 节点类型

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
- `chunk_id`
- `evidence`
- `confidence`

这样可以在回答时返回证据来源，降低大模型幻觉风险。
