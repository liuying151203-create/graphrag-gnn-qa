import json
from pathlib import Path

from scripts.build_hotpotqa_mini import (
    build_question_record,
    build_raw_document,
    load_hotpotqa_records,
    select_records,
    write_hotpotqa_mini,
)


def build_record(record_id: str = "abc123") -> dict:
    return {
        "_id": record_id,
        "question": "Which city is connected to the bridge fact?",
        "answer": "Paris",
        "type": "bridge",
        "level": "medium",
        "supporting_facts": [["Alpha", 1], ["Beta", 0]],
        "context": [
            ["Alpha", ["Distractor sentence.", "Alpha supporting sentence mentions Paris."]],
            ["Beta", ["Beta supporting sentence connects the evidence.", "Another sentence."]],
        ],
    }


def test_build_raw_document_writes_context_without_question_or_answer() -> None:
    document = build_raw_document(build_record())

    assert "# Alpha" in document
    assert "[1] Alpha supporting sentence mentions Paris." in document
    assert "Which city is connected" not in document
    assert "answer" not in document.casefold()


def test_build_question_record_preserves_supporting_facts_and_keywords() -> None:
    raw_file = Path("data/raw/hotpotqa_mini/hotpot_0001_abc123.txt")

    question = build_question_record(build_record(), raw_file)

    assert question["id"] == "abc123"
    assert question["question"] == "Which city is connected to the bridge fact?"
    assert question["expected_answer_keywords"] == ["Paris"]
    assert question["expected_evidence_keywords"] == [
        "Alpha",
        "Beta",
        "Alpha supporting sentence mentions Paris.",
        "Beta supporting sentence connects the evidence.",
    ]
    assert question["metadata"]["source"] == "hotpotqa"
    assert question["metadata"]["supporting_facts"] == [["Alpha", 1], ["Beta", 0]]


def test_select_records_filters_type_level_and_unusable_records() -> None:
    records = [
        {"_id": "missing-context", "question": "Q", "answer": "A", "supporting_facts": []},
        build_record("bridge-medium"),
        {**build_record("comparison-hard"), "type": "comparison", "level": "hard"},
    ]

    selected = select_records(records, limit=2, question_type="comparison", level="hard")

    assert [record["_id"] for record in selected] == ["comparison-hard"]


def test_write_hotpotqa_mini_creates_raw_documents_and_questions(tmp_path: Path) -> None:
    output_raw_dir = tmp_path / "raw"
    output_questions_file = tmp_path / "questions.jsonl"

    write_hotpotqa_mini([build_record()], output_raw_dir, output_questions_file)

    raw_files = list(output_raw_dir.glob("*.txt"))
    question_records = [json.loads(line) for line in output_questions_file.read_text(encoding="utf-8").splitlines()]

    assert len(raw_files) == 1
    assert raw_files[0].read_text(encoding="utf-8").startswith("# Alpha")
    assert len(question_records) == 1
    assert question_records[0]["metadata"]["raw_file"].endswith("hotpot_0001_abc123.txt")


def test_load_hotpotqa_records_accepts_huggingface_rows_format(tmp_path: Path) -> None:
    input_file = tmp_path / "hf_rows.json"
    input_file.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "row": {
                            "id": "hf-id",
                            "question": "Q?",
                            "answer": "A",
                            "type": "comparison",
                            "level": "hard",
                            "context": {
                                "title": ["Title A"],
                                "sentences": [["Sentence A0.", "Sentence A1."]],
                            },
                            "supporting_facts": {
                                "title": ["Title A"],
                                "sent_id": [1],
                            },
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    records = load_hotpotqa_records(input_file)

    assert records == [
        {
            "id": "hf-id",
            "_id": "hf-id",
            "question": "Q?",
            "answer": "A",
            "type": "comparison",
            "level": "hard",
            "context": [["Title A", ["Sentence A0.", "Sentence A1."]]],
            "supporting_facts": [["Title A", 1]],
        }
    ]
