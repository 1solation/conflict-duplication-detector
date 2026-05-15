"""Duplication detection using semantic similarity."""

from dataclasses import dataclass
from typing import Optional

from ..knowledge.document_loader import DocumentChunk
from ..knowledge.vector_store import VectorStore, SearchResult


@dataclass
class Duplication:
    """Represents a detected duplication between document chunks."""

    source_chunk: DocumentChunk
    similar_chunk: SearchResult
    similarity_score: float
    duplication_type: str  # "exact", "near_duplicate", "thematic"

    def to_dict(self) -> dict:
        return {
            "source": {
                "chunk_id": self.source_chunk.chunk_id,
                "document_id": self.source_chunk.document_id,
                "source_file": self.source_chunk.source_file,
                "content": self.source_chunk.content[:200] + "..."
                if len(self.source_chunk.content) > 200
                else self.source_chunk.content,
                "section_title": self.source_chunk.section_title,
            },
            "similar": {
                "chunk_id": self.similar_chunk.chunk_id,
                "document_id": self.similar_chunk.document_id,
                "source_file": self.similar_chunk.source_file,
                "content": self.similar_chunk.content[:200] + "..."
                if len(self.similar_chunk.content) > 200
                else self.similar_chunk.content,
                "section_title": self.similar_chunk.section_title,
            },
            "similarity_score": round(self.similarity_score, 4),
            "duplication_type": self.duplication_type,
        }

    def __str__(self) -> str:
        return (
            f"Duplication ({self.duplication_type}, score: {self.similarity_score:.2%}):\n"
            f"  Source: {self.source_chunk.source_file} - {self.source_chunk.section_title or 'N/A'}\n"
            f"  Similar: {self.similar_chunk.source_file} - {self.similar_chunk.section_title or 'N/A'}"
        )


class DuplicationDetector:
    """Detects semantic/thematic duplications across documents."""

    EXACT_THRESHOLD = 0.98
    NEAR_DUPLICATE_THRESHOLD = 0.90
    THEMATIC_THRESHOLD = 0.80

    def __init__(
        self,
        vector_store: VectorStore,
        similarity_threshold: float = 0.85,
    ):
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold

    def find_duplications(
        self,
        chunks: list[DocumentChunk],
        check_within_document: bool = False,
        n_similar: int = 5,
    ) -> list[Duplication]:
        """Find duplications for a list of chunks against the knowledge base."""
        duplications = []
        seen_pairs = set()

        for chunk in chunks:
            similar_results = self.vector_store.find_similar_chunks(
                chunk=chunk,
                n_results=n_similar,
                exclude_same_document=not check_within_document,
            )

            for result in similar_results:
                if result.similarity_score < self.similarity_threshold:
                    continue

                pair_key = tuple(sorted([chunk.chunk_id, result.chunk_id]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                duplication_type = self._classify_duplication(result.similarity_score)

                duplications.append(
                    Duplication(
                        source_chunk=chunk,
                        similar_chunk=result,
                        similarity_score=result.similarity_score,
                        duplication_type=duplication_type,
                    )
                )

        duplications.sort(key=lambda d: d.similarity_score, reverse=True)
        return duplications

    def find_all_duplications(
        self,
        check_within_document: bool = False,
        n_similar: int = 5,
    ) -> list[Duplication]:
        """Find all duplications in the knowledge base."""
        all_chunks = self.vector_store.get_all_chunks()

        chunks = [
            DocumentChunk(
                content=c["content"],
                document_id=c["metadata"]["document_id"],
                chunk_id=c["chunk_id"],
                source_file=c["metadata"]["source_file"],
                page_number=(
                    c["metadata"].get("page_number")
                    if c["metadata"].get("page_number", -1) != -1
                    else None
                ),
                section_title=c["metadata"].get("section_title") or None,
            )
            for c in all_chunks
        ]

        return self.find_duplications(
            chunks=chunks,
            check_within_document=check_within_document,
            n_similar=n_similar,
        )

    def _classify_duplication(self, similarity_score: float) -> str:
        """Classify the type of duplication based on similarity score."""
        if similarity_score >= self.EXACT_THRESHOLD:
            return "exact"
        elif similarity_score >= self.NEAR_DUPLICATE_THRESHOLD:
            return "near_duplicate"
        else:
            return "thematic"

    def get_duplication_summary(
        self, duplications: list[Duplication]
    ) -> dict:
        """Generate a summary of detected duplications."""
        summary = {
            "total_duplications": len(duplications),
            "by_type": {"exact": 0, "near_duplicate": 0, "thematic": 0},
            "affected_documents": set(),
            "highest_similarity": 0.0,
            "average_similarity": 0.0,
        }

        if not duplications:
            return summary

        for dup in duplications:
            summary["by_type"][dup.duplication_type] += 1
            summary["affected_documents"].add(dup.source_chunk.source_file)
            summary["affected_documents"].add(dup.similar_chunk.source_file)

            if dup.similarity_score > summary["highest_similarity"]:
                summary["highest_similarity"] = dup.similarity_score

        summary["average_similarity"] = sum(d.similarity_score for d in duplications) / len(
            duplications
        )
        summary["affected_documents"] = list(summary["affected_documents"])

        return summary
