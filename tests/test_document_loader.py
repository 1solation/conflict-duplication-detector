"""Tests for document loader."""

import pytest
from pathlib import Path
import tempfile

from src.knowledge.document_loader import DocumentLoader, DocumentChunk


class TestDocumentLoader:
    """Tests for DocumentLoader class."""

    def test_init_default_values(self):
        loader = DocumentLoader()
        assert loader.chunk_size == 1000
        assert loader.chunk_overlap == 200

    def test_init_custom_values(self):
        loader = DocumentLoader(chunk_size=500, chunk_overlap=100)
        assert loader.chunk_size == 500
        assert loader.chunk_overlap == 100

    def test_supported_extensions(self):
        loader = DocumentLoader()
        assert ".pdf" in loader.SUPPORTED_EXTENSIONS
        assert ".docx" in loader.SUPPORTED_EXTENSIONS
        assert ".html" in loader.SUPPORTED_EXTENSIONS
        assert ".htm" in loader.SUPPORTED_EXTENSIONS

    def test_load_document_file_not_found(self):
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_document("nonexistent_file.pdf")

    def test_load_document_unsupported_format(self):
        loader = DocumentLoader()
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load_document(temp_path)

        Path(temp_path).unlink()

    def test_chunk_text_small_text(self):
        loader = DocumentLoader(chunk_size=1000)
        chunks = loader._chunk_text(
            text="Short text",
            document_id="doc1",
            source_file="test.pdf",
        )
        assert len(chunks) == 1
        assert chunks[0].content == "Short text"

    def test_chunk_text_large_text(self):
        loader = DocumentLoader(chunk_size=100, chunk_overlap=20)
        long_text = "This is a sentence. " * 50
        chunks = loader._chunk_text(
            text=long_text,
            document_id="doc1",
            source_file="test.pdf",
        )
        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)

    def test_document_chunk_to_dict(self):
        chunk = DocumentChunk(
            content="Test content",
            document_id="doc1",
            chunk_id="chunk1",
            source_file="test.pdf",
            page_number=1,
            section_title="Introduction",
        )
        result = chunk.to_dict()
        assert result["content"] == "Test content"
        assert result["document_id"] == "doc1"
        assert result["page_number"] == 1


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""

    def test_create_chunk(self):
        chunk = DocumentChunk(
            content="Sample content",
            document_id="doc123",
            chunk_id="chunk456",
            source_file="/path/to/file.pdf",
        )
        assert chunk.content == "Sample content"
        assert chunk.document_id == "doc123"
        assert chunk.chunk_id == "chunk456"
        assert chunk.page_number is None
        assert chunk.section_title is None

    def test_chunk_with_metadata(self):
        chunk = DocumentChunk(
            content="Content",
            document_id="doc1",
            chunk_id="chunk1",
            source_file="test.pdf",
            metadata={"custom_field": "value"},
        )
        result = chunk.to_dict()
        assert result["custom_field"] == "value"
