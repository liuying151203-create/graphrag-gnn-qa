from io import BytesIO
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadedDocument:
    content: str
    source: str
    file_name: str
    file_type: str


class DocumentLoader:
    supported_extensions = {".txt", ".md", ".markdown", ".pdf"}

    def load(self, file_path: str | Path) -> LoadedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        return self.load_bytes(
            content=path.read_bytes(),
            file_name=path.name,
            source=str(path),
        )

    def load_bytes(
        self,
        content: bytes,
        file_name: str,
        source: str | None = None,
    ) -> LoadedDocument:
        suffix = Path(file_name).suffix.lower()
        if suffix not in self.supported_extensions:
            raise ValueError(f"Unsupported document type: {suffix}")

        if suffix == ".pdf":
            try:
                text = self._load_pdf(content)
            except Exception as exc:
                raise ValueError("Failed to parse PDF document") from exc
        else:
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError("Text documents must use UTF-8 encoding") from exc

        return LoadedDocument(
            content=text.strip(),
            source=source or file_name,
            file_name=file_name,
            file_type=suffix.lstrip("."),
        )

    def _load_pdf(self, content: bytes) -> str:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
        return "\n\n".join(pages)
