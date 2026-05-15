"""Vector store using ChromaDB for document embeddings and retrieval."""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from .document_loader import DocumentChunk


@dataclass
class SearchResult:
    """Represents a search result from the vector store."""

    chunk_id: str
    document_id: str
    content: str
    source_file: str
    similarity_score: float
    page_number: Optional[int] = None
    section_title: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "source_file": self.source_file,
            "similarity_score": self.similarity_score,
            "page_number": self.page_number,
            "section_title": self.section_title,
        }


class VectorStore:
    """ChromaDB-based vector store for document embeddings."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "knowledge_base",
        embedding_model: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None,
    ):
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIRECTORY", "./chroma_db"
        )
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass openai_api_key parameter."
            )

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        self.embedding_function = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=self.embedding_model,
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Add document chunks to the vector store."""
        if not chunks:
            return 0

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "source_file": chunk.source_file,
                "page_number": chunk.page_number or -1,
                "section_title": chunk.section_title or "",
            }
            for chunk in chunks
        ]

        existing_ids = set()
        try:
            existing = self.collection.get(ids=ids)
            existing_ids = set(existing["ids"])
        except Exception:
            pass

        new_indices = [i for i, id_ in enumerate(ids) if id_ not in existing_ids]

        if not new_indices:
            return 0

        new_ids = [ids[i] for i in new_indices]
        new_documents = [documents[i] for i in new_indices]
        new_metadatas = [metadatas[i] for i in new_indices]

        self.collection.add(
            ids=new_ids,
            documents=new_documents,
            metadatas=new_metadatas,
        )

        return len(new_ids)

    def search(
        self,
        query: str,
        n_results: int = 10,
        filter_document_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        where_filter = None
        if filter_document_id:
            where_filter = {"document_id": filter_document_id}

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                content = results["documents"][0][i] if results["documents"] else ""

                search_results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        document_id=metadata.get("document_id", ""),
                        content=content,
                        source_file=metadata.get("source_file", ""),
                        similarity_score=similarity,
                        page_number=(
                            metadata.get("page_number")
                            if metadata.get("page_number", -1) != -1
                            else None
                        ),
                        section_title=metadata.get("section_title") or None,
                    )
                )

        return search_results

    def find_similar_chunks(
        self,
        chunk: DocumentChunk,
        n_results: int = 5,
        exclude_same_document: bool = True,
    ) -> list[SearchResult]:
        """Find chunks similar to a given chunk."""
        results = self.collection.query(
            query_texts=[chunk.content],
            n_results=n_results + 10,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, result_chunk_id in enumerate(results["ids"][0]):
                if result_chunk_id == chunk.chunk_id:
                    continue

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                if (
                    exclude_same_document
                    and metadata.get("document_id") == chunk.document_id
                ):
                    continue

                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance

                content = results["documents"][0][i] if results["documents"] else ""

                search_results.append(
                    SearchResult(
                        chunk_id=result_chunk_id,
                        document_id=metadata.get("document_id", ""),
                        content=content,
                        source_file=metadata.get("source_file", ""),
                        similarity_score=similarity,
                        page_number=(
                            metadata.get("page_number")
                            if metadata.get("page_number", -1) != -1
                            else None
                        ),
                        section_title=metadata.get("section_title") or None,
                    )
                )

                if len(search_results) >= n_results:
                    break

        return search_results

    def get_all_chunks(self) -> list[dict]:
        """Retrieve all chunks from the collection."""
        results = self.collection.get(include=["documents", "metadatas"])

        chunks = []
        for i, chunk_id in enumerate(results["ids"]):
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i],
                }
            )

        return chunks

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document."""
        results = self.collection.get(
            where={"document_id": document_id}, include=["metadatas"]
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])

        return 0

    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        """Return the number of chunks in the collection."""
        return self.collection.count()
