import json
from pathlib import Path

from graphrag_gnn_qa.gnn.graph_dataset import GraphDataset, GraphEdge, GraphNode
from scripts.export_graph_dataset import export_graph_dataset, parse_args, write_graph_dataset


class FakeGraphStore:
    def export_graph_dataset(self) -> GraphDataset:
        return GraphDataset(
            nodes=[GraphNode(node_id="Method:graphrag", name="GraphRAG", node_type="Method")],
            edges=[
                GraphEdge(
                    source_id="Method:graphrag",
                    target_id="Task:question answering",
                    relation_type="SOLVES_TASK",
                )
            ],
        )


def test_write_graph_dataset_writes_json(tmp_path: Path) -> None:
    output_file = tmp_path / "graph_dataset.json"
    dataset = FakeGraphStore().export_graph_dataset()

    write_graph_dataset(dataset=dataset, output_file=output_file)

    assert json.loads(output_file.read_text(encoding="utf-8")) == dataset.to_dict()


def test_export_graph_dataset_writes_store_output(tmp_path: Path) -> None:
    output_file = tmp_path / "graph_dataset.json"

    dataset = export_graph_dataset(output_file=output_file, graph_store=FakeGraphStore())

    assert len(dataset.nodes) == 1
    assert len(dataset.edges) == 1
    assert json.loads(output_file.read_text(encoding="utf-8"))["summary"] == {
        "node_count": 1,
        "edge_count": 1,
    }


def test_parse_args_for_export_graph_dataset(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["export_graph_dataset.py", "--output-file", "data/tmp/graph_dataset.json"],
    )

    args = parse_args()

    assert args.output_file == Path("data/tmp/graph_dataset.json")
