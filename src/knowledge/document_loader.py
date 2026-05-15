"""Document loader for processing DOCX, PDF, and HTML files."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import hashlib

import fitz  # PyMuPDF
from docx import Document
from bs4 import BeautifulSoup


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""

    content: str
    document_id: str
    chunk_id: str
    source_file: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "section_title": self.section_title,
            **self.metadata,
        }


class DocumentLoader:
    """Loads and processes documents in various formats."""

    SUPPORTED_EXTENSIONS = {".docx", ".pdf", ".html", ".htm"}

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_document(self, file_path: str | Path) -> list[DocumentChunk]:
        """Load a document and return chunks."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        extension = path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {self.SUPPORTED_EXTENSIONS}"
            )

        document_id = self._generate_document_id(path)

        if extension == ".pdf":
            return self._load_pdf(path, document_id)
        elif extension == ".docx":
            return self._load_docx(path, document_id)
        elif extension in {".html", ".htm"}:
            return self._load_html(path, document_id)

        return []

    def load_directory(self, directory_path: str | Path) -> list[DocumentChunk]:
        """Load all supported documents from a directory."""
        path = Path(directory_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        all_chunks = []
        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in path.rglob(f"*{ext}"):
                try:
                    chunks = self.load_document(file_path)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")

        return all_chunks

    def _generate_document_id(self, path: Path) -> str:
        """Generate a unique document ID based on file path and content hash."""
        content_hash = hashlib.md5(path.read_bytes()).hexdigest()[:8]
        return f"{path.stem}_{content_hash}"

    def _load_pdf(self, path: Path, document_id: str) -> list[DocumentChunk]:
        """Load and chunk a PDF document."""
        chunks = []
        doc = fitz.open(path)

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                page_chunks = self._chunk_text(
                    text=text,
                    document_id=document_id,
                    source_file=str(path),
                    page_number=page_num,
                )
                chunks.extend(page_chunks)

        doc.close()
        return chunks

    def _load_docx(self, path: Path, document_id: str) -> list[DocumentChunk]:
        """Load and chunk a DOCX document."""
        doc = Document(path)
        chunks = []

        current_section = None
        section_text = []

        for para in doc.paragraphs:
            if para.style.name.startswith("Heading"):
                if section_text:
                    text = "\n".join(section_text)
                    section_chunks = self._chunk_text(
                        text=text,
                        document_id=document_id,
                        source_file=str(path),
                        section_title=current_section,
                    )
                    chunks.extend(section_chunks)
                    section_text = []

                current_section = para.text.strip()
            else:
                if para.text.strip():
                    section_text.append(para.text)

        if section_text:
            text = "\n".join(section_text)
            section_chunks = self._chunk_text(
                text=text,
                document_id=document_id,
                source_file=str(path),
                section_title=current_section,
            )
            chunks.extend(section_chunks)

        return chunks

    def _load_html(self, path: Path, document_id: str) -> list[DocumentChunk]:
        """Load and chunk an HTML document."""
        with open(path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "lxml")

        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        chunks = []
        current_section = None

        for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "div"]):
            if element.name.startswith("h"):
                current_section = element.get_text(strip=True)
            else:
                text = element.get_text(strip=True)
                if text and len(text) > 50:
                    element_chunks = self._chunk_text(
                        text=text,
                        document_id=document_id,
                        source_file=str(path),
                        section_title=current_section,
                    )
                    chunks.extend(element_chunks)

        return chunks

    def _chunk_text(
        self,
        text: str,
        document_id: str,
        source_file: str,
        page_number: Optional[int] = None,
        section_title: Optional[str] = None,
    ) -> list[DocumentChunk]:
        """Split text into overlapping chunks."""
        text = text.strip()
        if not text:
            return []

        def make_chunk_id(content: str, index: int) -> str:
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            return f"{document_id}_{content_hash}_{index}"

        if len(text) <= self.chunk_size:
            chunk_id = make_chunk_id(text, 0)
            return [
                DocumentChunk(
                    content=text,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    source_file=source_file,
                    page_number=page_number,
                    section_title=section_title,
                )
            ]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            if end < len(text):
                last_period = text.rfind(".", start, end)
                last_newline = text.rfind("\n", start, end)
                break_point = max(last_period, last_newline)

                if break_point > start:
                    end = break_point + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = make_chunk_id(chunk_text, chunk_index)
                chunks.append(
                    DocumentChunk(
                        content=chunk_text,
                        document_id=document_id,
                        chunk_id=chunk_id,
                        source_file=source_file,
                        page_number=page_number,
                        section_title=section_title,
                    )
                )
                chunk_index += 1

            start = end - self.chunk_overlap
            if start >= len(text) - self.chunk_overlap:
                break

        return chunks
