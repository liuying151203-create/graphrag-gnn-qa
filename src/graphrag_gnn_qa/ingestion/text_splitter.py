from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    content: str
    start_index: int
    end_index: int


class TextSplitter:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, document_id: str = "doc") -> list[TextChunk]:
        normalized_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if not normalized_text:
            return []

        chunks = []
        start = 0
        chunk_index = 0
        text_length = len(normalized_text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            content = normalized_text[start:end].strip()

            if content:
                chunks.append(
                    TextChunk(
                        chunk_id=f"{document_id}_chunk_{chunk_index:04d}",
                        content=content,
                        start_index=start,
                        end_index=end,
                    )
                )
                chunk_index += 1

            if end == text_length:
                break

            start = end - self.chunk_overlap

        return chunks
