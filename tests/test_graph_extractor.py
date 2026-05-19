import pytest

from graphrag_gnn_qa.graph.extractor import (
    GraphEntity,
    GraphExtractor,
    GraphRelation,
    build_extraction_prompt,
    graph_result_to_dict,
    parse_extraction_response,
)


class FakeLLMClient:
    def generate(self, prompt: str) -> str:
        return """
        {
          "entities": [
            {"name": "GraphRAG", "type": "Method", "description": "Graph-based retrieval augmented generation"},
            {"name": "question answering", "type": "Task", "description": "Answering user questions"}
          ],
          "relations": [
            {
              "source_entity": "GraphRAG",
              "source_type": "Method",
              "relation_type": "SOLVES_TASK",
              "target_entity": "question answering",
              "target_type": "Task",
              "evidence": "GraphRAG improves question answering.",
              "confidence": 0.87
            }
          ]
        }
        """


def test_build_extraction_prompt_contains_text_and_schema() -> None:
    prompt = build_extraction_prompt("GraphRAG improves question answering.")

    assert "GraphRAG improves question answering." in prompt
    assert "Entity types must be one of" in prompt
    assert "Relation types must be one of" in prompt


def test_parse_extraction_response() -> None:
    response = """
    ```json
    {
      "entities": [{"name": "GraphRAG", "type": "Method", "description": "A RAG method"}],
      "relations": [
        {
          "source_entity": "GraphRAG",
          "source_type": "Method",
          "relation_type": "SOLVES_TASK",
          "target_entity": "question answering",
          "target_type": "Task",
          "evidence": "GraphRAG solves question answering.",
          "confidence": 0.9
        }
      ]
    }
    ```
    """

    entities, relations = parse_extraction_response(response)

    assert entities == [GraphEntity(name="GraphRAG", type="Method", description="A RAG method")]
    assert relations == [
        GraphRelation(
            source_entity="GraphRAG",
            source_type="Method",
            relation_type="SOLVES_TASK",
            target_entity="question answering",
            target_type="Task",
            evidence="GraphRAG solves question answering.",
            confidence=0.9,
        )
    ]


def test_parse_extraction_response_rejects_unknown_entity_type() -> None:
    response = '{"entities": [{"name": "Unknown", "type": "UnknownType"}], "relations": []}'

    with pytest.raises(ValueError):
        parse_extraction_response(response)


def test_graph_extractor_extracts_from_chunk() -> None:
    extractor = GraphExtractor(llm_client=FakeLLMClient())
    chunk = {
        "chunk_id": "sample_chunk_0000",
        "document_id": "sample",
        "source": "sample.txt",
        "content": "GraphRAG improves question answering.",
    }

    result = extractor.extract_from_chunk(chunk)

    assert result.chunk_id == "sample_chunk_0000"
    assert result.entities[0].name == "GraphRAG"
    assert result.relations[0].relation_type == "SOLVES_TASK"
    assert graph_result_to_dict(result)["entities"][0]["type"] == "Method"
