import re

QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "can",
    "does",
    "for",
    "from",
    "how",
    "explain",
    "show",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "solve",
    "solves",
    "use",
    "uses",
    "with",
}


def extract_query_entities(question: str, max_entities: int = 5) -> list[str]:
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question must not be empty")
    if max_entities <= 0:
        raise ValueError("max_entities must be greater than 0")

    candidates: list[str] = []
    candidates.extend(extract_quoted_phrases(normalized_question))
    candidates.extend(extract_mixed_case_terms(normalized_question))
    candidates.extend(extract_keyword_phrases(normalized_question))
    return deduplicate_candidates(candidates)[:max_entities]


def extract_quoted_phrases(text: str) -> list[str]:
    return [match.strip() for match in re.findall(r'"([^"\n]+)"|\'([^\'\n]+)\'', text) for match in match if match.strip()]


def extract_mixed_case_terms(text: str) -> list[str]:
    terms = []
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*\b", text):
        if token.lower() in QUESTION_STOPWORDS:
            continue
        has_internal_uppercase = any(character.isupper() for character in token[1:])
        has_digit = any(character.isdigit() for character in token)
        if token.isupper() or has_internal_uppercase or has_digit:
            terms.append(token)
    return terms


def extract_keyword_phrases(text: str) -> list[str]:
    words = [word for word in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*\b", text) if word.lower() not in QUESTION_STOPWORDS]
    phrases: list[str] = []
    for size in range(min(3, len(words)), 1, -1):
        for index in range(0, len(words) - size + 1):
            phrases.append(" ".join(words[index : index + size]))
    phrases.extend(words)
    return phrases


def deduplicate_candidates(candidates: list[str]) -> list[str]:
    seen = set()
    deduplicated = []
    for candidate in candidates:
        normalized_candidate = re.sub(r"\s+", " ", candidate.strip(" .,:;!?()[]{}\n\t"))
        key = normalized_candidate.lower()
        if normalized_candidate and key not in seen:
            seen.add(key)
            deduplicated.append(normalized_candidate)
    return deduplicated
