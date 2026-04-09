import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass, field
import hashlib


@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content[:500].encode()).hexdigest()


class DocumentLoader:

    @staticmethod
    def load_pdf(path: Path) -> Document:
        doc = fitz.open(str(path))
        pages = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")
            if text.strip():
                pages.append({"text": text, "page": page_num})

        full_text = "\n\n".join(p["text"] for p in pages)
        return Document(
            content=full_text,
            metadata={
                "source": path.name,
                "total_pages": len(doc),
                "file_path": str(path)
            }
        )

    @staticmethod
    def load_text(path: Path) -> Document:
        content = path.read_text(encoding="utf-8")
        return Document(
            content=content,
            metadata={"source": path.name, "file_path": str(path)}
        )

    def load_directory(self, dir_path: Path) -> list:
        docs = []
        for file_path in sorted(dir_path.rglob("*")):
            if file_path.suffix.lower() == ".pdf":
                docs.append(self.load_pdf(file_path))
            elif file_path.suffix.lower() in (".txt", ".md"):
                docs.append(self.load_text(file_path))
        return docs
