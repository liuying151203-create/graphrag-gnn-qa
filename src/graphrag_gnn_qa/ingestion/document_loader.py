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

        suffix = path.suffix.lower()
        if suffix not in self.supported_extensions:
            raise ValueError(f"Unsupported document type: {suffix}")

        if suffix == ".pdf":
            content = self._load_pdf(path)
        else:
            content = path.read_text(encoding="utf-8")

        return LoadedDocument(
            content=content.strip(),
            source=str(path),
            file_name=path.name,
            file_type=suffix.lstrip("."),
        )

    def _load_pdf(self, path: Path) -> str:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
        return "\n\n".join(pages)
