import argparse
import json
import re
from pathlib import Path
from typing import Any


HOTPOTQA_DEV_DISTRACTOR_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"


def load_hotpotqa_records(input_file: Path) -> list[dict[str, Any]]:
    if not input_file.exists():
        raise FileNotFoundError(f"HotpotQA file not found: {input_file}")

    data = json.loads(input_file.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return [normalize_hotpotqa_record(row.get("row", {})) for row in data["rows"] if isinstance(row, dict)]
    if not isinstance(data, list):
        raise ValueError(f"Expected HotpotQA top-level JSON list: {input_file}")
    return [normalize_hotpotqa_record(record) for record in data if isinstance(record, dict)]


def normalize_hotpotqa_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    if "_id" not in normalized and "id" in normalized:
        normalized["_id"] = normalized["id"]
    normalized["context"] = normalize_context(normalized.get("context", []))
    normalized["supporting_facts"] = normalize_supporting_facts(normalized.get("supporting_facts", []))
    return normalized


def normalize_context(value: Any) -> list[list[Any]]:
    if isinstance(value, list):
        return value
    if not isinstance(value, dict):
        return []

    titles = value.get("title", [])
    sentences_list = value.get("sentences", [])
    if not isinstance(titles, list) or not isinstance(sentences_list, list):
        return []
    return [[title, sentences] for title, sentences in zip(titles, sentences_list, strict=False)]


def normalize_supporting_facts(value: Any) -> list[list[Any]]:
    if isinstance(value, list):
        return value
    if not isinstance(value, dict):
        return []

    titles = value.get("title", [])
    sentence_ids = value.get("sent_id", [])
    if not isinstance(titles, list) or not isinstance(sentence_ids, list):
        return []
    return [[title, sentence_id] for title, sentence_id in zip(titles, sentence_ids, strict=False)]


def select_records(
    records: list[dict[str, Any]],
    limit: int,
    start_index: int = 0,
    question_type: str | None = None,
    level: str | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for record in records[start_index:]:
        if question_type is not None and record.get("type") != question_type:
            continue
        if level is not None and record.get("level") != level:
            continue
        if not is_usable_record(record):
            continue
        selected.append(record)
        if len(selected) >= limit:
            break
    return selected


def is_usable_record(record: dict[str, Any]) -> bool:
    return bool(
        str(record.get("_id", "")).strip()
        and str(record.get("question", "")).strip()
        and str(record.get("answer", "")).strip()
        and isinstance(record.get("context"), list)
        and isinstance(record.get("supporting_facts"), list)
    )


def write_hotpotqa_mini(
    records: list[dict[str, Any]],
    output_raw_dir: Path,
    output_questions_file: Path,
) -> None:
    output_raw_dir.mkdir(parents=True, exist_ok=True)
    output_questions_file.parent.mkdir(parents=True, exist_ok=True)

    with output_questions_file.open("w", encoding="utf-8") as question_writer:
        for index, record in enumerate(records, start=1):
            raw_file = output_raw_dir / build_raw_file_name(index=index, record_id=str(record["_id"]))
            raw_file.write_text(build_raw_document(record), encoding="utf-8")
            question_writer.write(json.dumps(build_question_record(record, raw_file), ensure_ascii=False) + "\n")


def build_raw_file_name(index: int, record_id: str) -> str:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", record_id).strip("_")
    return f"hotpot_{index:04d}_{safe_id}.txt"


def build_raw_document(record: dict[str, Any]) -> str:
    sections = []
    for title, sentences in iter_context_paragraphs(record):
        lines = [f"# {title}", ""]
        for sentence_index, sentence in enumerate(sentences):
            sentence_text = str(sentence).strip()
            if sentence_text:
                lines.append(f"[{sentence_index}] {sentence_text}")
        sections.append("\n".join(lines).strip())
    return "\n\n".join(section for section in sections if section).strip() + "\n"


def iter_context_paragraphs(record: dict[str, Any]) -> list[tuple[str, list[str]]]:
    paragraphs = []
    for item in record.get("context", []):
        if not isinstance(item, list) or len(item) != 2:
            continue
        title, sentences = item
        if not isinstance(sentences, list):
            continue
        title_text = str(title).strip()
        if title_text:
            paragraphs.append((title_text, [str(sentence).strip() for sentence in sentences]))
    return paragraphs


def build_question_record(record: dict[str, Any], raw_file: Path) -> dict[str, Any]:
    supporting_sentences = find_supporting_sentences(record)
    supporting_titles = [title for title, _ in parse_supporting_facts(record)]
    answer = str(record.get("answer", "")).strip()
    record_id = str(record.get("_id") or record.get("id") or "").strip()

    return {
        "id": record_id,
        "question": str(record["question"]).strip(),
        "expected_evidence_keywords": unique_non_empty([*supporting_titles, *supporting_sentences]),
        "expected_answer_keywords": unique_non_empty([answer]),
        "metadata": {
            "source": "hotpotqa",
            "split": "dev_distractor",
            "type": record.get("type"),
            "level": record.get("level"),
            "answer": answer,
            "supporting_facts": [[title, sentence_id] for title, sentence_id in parse_supporting_facts(record)],
            "raw_file": raw_file.as_posix(),
        },
    }


def find_supporting_sentences(record: dict[str, Any]) -> list[str]:
    context_by_title = {title: sentences for title, sentences in iter_context_paragraphs(record)}
    sentences = []
    for title, sentence_id in parse_supporting_facts(record):
        paragraph = context_by_title.get(title)
        if paragraph is None or sentence_id < 0 or sentence_id >= len(paragraph):
            continue
        sentences.append(paragraph[sentence_id])
    return sentences


def parse_supporting_facts(record: dict[str, Any]) -> list[tuple[str, int]]:
    facts = []
    for item in record.get("supporting_facts", []):
        if not isinstance(item, list) or len(item) != 2:
            continue
        title, sentence_id = item
        try:
            facts.append((str(title).strip(), int(sentence_id)))
        except (TypeError, ValueError):
            continue
    return [(title, sentence_id) for title, sentence_id in facts if title]


def unique_non_empty(values: list[str]) -> list[str]:
    result = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build project-ready raw documents and eval questions from an official HotpotQA JSON file."
    )
    parser.add_argument("--input-file", type=Path, required=True, help="Official HotpotQA JSON file.")
    parser.add_argument("--output-raw-dir", type=Path, default=Path("data/raw/hotpotqa_mini"))
    parser.add_argument("--output-questions-file", type=Path, default=Path("data/eval/questions.hotpotqa_mini.jsonl"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--type", choices=["bridge", "comparison"], default=None)
    parser.add_argument("--level", choices=["easy", "medium", "hard"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_hotpotqa_records(args.input_file)
    selected_records = select_records(
        records=records,
        limit=args.limit,
        start_index=args.start_index,
        question_type=args.type,
        level=args.level,
    )
    if not selected_records:
        raise ValueError("No usable HotpotQA records matched the selection criteria.")

    write_hotpotqa_mini(
        records=selected_records,
        output_raw_dir=args.output_raw_dir,
        output_questions_file=args.output_questions_file,
    )
    print(f"Wrote {len(selected_records)} raw documents: {args.output_raw_dir}")
    print(f"Wrote {len(selected_records)} questions: {args.output_questions_file}")
    print(f"Official dev distractor URL: {HOTPOTQA_DEV_DISTRACTOR_URL}")


if __name__ == "__main__":
    main()
