from graphrag_gnn_qa.gnn.graph_dataset import GraphDataset, GraphEdge, GraphNode, graph_dataset_from_records


def test_graph_dataset_to_dict_includes_summary() -> None:
    dataset = GraphDataset(
        nodes=[
            GraphNode(
                node_id="Method:graphrag",
                name="GraphRAG",
                node_type="Method",
                description="Graph retrieval augmented generation.",
            )
        ],
        edges=[
            GraphEdge(
                source_id="Method:graphrag",
                target_id="Task:question answering",
                relation_type="SOLVES_TASK",
                chunk_id="sample_chunk_0000",
                document_id="sample",
                source="sample.txt",
                evidence="GraphRAG improves question answering.",
                confidence=0.9,
            )
        ],
    )

    assert dataset.to_dict() == {
        "nodes": [
            {
                "node_id": "Method:graphrag",
                "name": "GraphRAG",
                "node_type": "Method",
                "description": "Graph retrieval augmented generation.",
            }
        ],
        "edges": [
            {
                "source_id": "Method:graphrag",
                "target_id": "Task:question answering",
                "relation_type": "SOLVES_TASK",
                "chunk_id": "sample_chunk_0000",
                "document_id": "sample",
                "source": "sample.txt",
                "evidence": "GraphRAG improves question answering.",
                "confidence": 0.9,
            }
        ],
        "summary": {"node_count": 1, "edge_count": 1},
    }


def test_graph_dataset_from_records_normalizes_missing_optional_values() -> None:
    dataset = graph_dataset_from_records(
        node_records=[
            {
                "node_id": "Concept:rag",
                "name": "RAG",
                "node_type": "Concept",
                "description": None,
            }
        ],
        edge_records=[
            {
                "source_id": "Concept:rag",
                "target_id": "Method:graphrag",
                "relation_type": "RELATED_TO",
                "confidence": None,
            }
        ],
    )

    assert dataset.nodes[0].description == ""
    assert dataset.edges[0].chunk_id == ""
    assert dataset.edges[0].confidence == 0.0
