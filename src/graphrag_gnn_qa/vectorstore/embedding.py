from typing import Protocol


class EmbeddingModel(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class SentenceTransformerEmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(float).tolist()


class HashEmbeddingModel:
    def __init__(self, dimension: int = 16) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for index, character in enumerate(text):
            bucket = (ord(character) + index) % self.dimension
            vector[bucket] += 1.0

        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return vector
        return [value / norm for value in vector]
