"""Conflict detection using LLM analysis."""

import os
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from openai import OpenAI

from ..knowledge.document_loader import DocumentChunk
from ..knowledge.vector_store import VectorStore, SearchResult


class ConflictSeverity(Enum):
    """Severity levels for detected conflicts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Conflict:
    """Represents a detected conflict between document chunks."""

    chunk_a: DocumentChunk
    chunk_b: SearchResult
    conflict_type: str
    description: str
    severity: ConflictSeverity
    confidence: float
    recommendation: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "chunk_a": {
                "chunk_id": self.chunk_a.chunk_id,
                "document_id": self.chunk_a.document_id,
                "source_file": self.chunk_a.source_file,
                "content": self.chunk_a.content[:300] + "..."
                if len(self.chunk_a.content) > 300
                else self.chunk_a.content,
                "section_title": self.chunk_a.section_title,
            },
            "chunk_b": {
                "chunk_id": self.chunk_b.chunk_id,
                "document_id": self.chunk_b.document_id,
                "source_file": self.chunk_b.source_file,
                "content": self.chunk_b.content[:300] + "..."
                if len(self.chunk_b.content) > 300
                else self.chunk_b.content,
                "section_title": self.chunk_b.section_title,
            },
            "conflict_type": self.conflict_type,
            "description": self.description,
            "severity": self.severity.value,
            "confidence": round(self.confidence, 2),
            "recommendation": self.recommendation,
        }

    def __str__(self) -> str:
        return (
            f"Conflict [{self.severity.value.upper()}] - {self.conflict_type}\n"
            f"  Description: {self.description}\n"
            f"  Between: {self.chunk_a.source_file} <-> {self.chunk_b.source_file}\n"
            f"  Confidence: {self.confidence:.0%}"
        )


CONFLICT_ANALYSIS_PROMPT = """You are an expert document analyst specializing in identifying conflicts and contradictions in guidance documents.

Analyse the following two text passages and determine if there are any conflicts, contradictions, or inconsistencies between them.

PASSAGE A (from {source_a}):
{content_a}

PASSAGE B (from {source_b}):
{content_b}

Analyse these passages for:
1. Direct contradictions (opposite statements about the same topic)
2. Numerical discrepancies (different numbers, dates, percentages)
3. Procedural conflicts (different steps or sequences for the same process)
4. Policy conflicts (different rules or requirements for the same situation)
5. Definitional conflicts (different definitions for the same term)

Respond in the following JSON format:
{{
    "has_conflict": true/false,
    "conflict_type": "contradiction|numerical|procedural|policy|definitional|none",
    "description": "Brief description of the conflict",
    "severity": "low|medium|high|critical",
    "confidence": 0.0-1.0,
    "recommendation": "How to resolve this conflict"
}}

If there is no conflict, set has_conflict to false and provide empty values for other fields.
Be precise and only report genuine conflicts, not mere differences in wording or emphasis."""


class ConflictDetector:
    """Detects conflicts and contradictions using LLM analysis."""

    def __init__(
        self,
        vector_store: VectorStore,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o",
        confidence_threshold: float = 0.7,
    ):
        self.vector_store = vector_store
        self.model = model
        self.confidence_threshold = confidence_threshold

        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.client = OpenAI(api_key=api_key)

    def detect_conflicts(
        self,
        chunks: list[DocumentChunk],
        similarity_threshold: float = 0.5,
        n_candidates: int = 5,
    ) -> list[Conflict]:
        """Detect conflicts between chunks and the knowledge base."""
        conflicts = []
        analysed_pairs = set()

        for chunk in chunks:
            similar_chunks = self.vector_store.find_similar_chunks(
                chunk=chunk,
                n_results=n_candidates,
                exclude_same_document=True,
            )

            candidates = [
                c for c in similar_chunks if c.similarity_score >= similarity_threshold
            ]

            for candidate in candidates:
                pair_key = tuple(sorted([chunk.chunk_id, candidate.chunk_id]))
                if pair_key in analysed_pairs:
                    continue
                analysed_pairs.add(pair_key)

                conflict = self._analyse_conflict(chunk, candidate)
                if conflict and conflict.confidence >= self.confidence_threshold:
                    conflicts.append(conflict)

        conflicts.sort(
            key=lambda c: (
                -["low", "medium", "high", "critical"].index(c.severity.value),
                -c.confidence,
            )
        )

        return conflicts

    def _analyse_conflict(
        self,
        chunk_a: DocumentChunk,
        chunk_b: SearchResult,
    ) -> Optional[Conflict]:
        """Use LLM to analyse potential conflict between two chunks."""
        prompt = CONFLICT_ANALYSIS_PROMPT.format(
            source_a=chunk_a.source_file,
            content_a=chunk_a.content,
            source_b=chunk_b.source_file,
            content_b=chunk_b.content,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            import json

            result = json.loads(response.choices[0].message.content)

            if not result.get("has_conflict", False):
                return None

            severity_map = {
                "low": ConflictSeverity.LOW,
                "medium": ConflictSeverity.MEDIUM,
                "high": ConflictSeverity.HIGH,
                "critical": ConflictSeverity.CRITICAL,
            }

            return Conflict(
                chunk_a=chunk_a,
                chunk_b=chunk_b,
                conflict_type=result.get("conflict_type", "unknown"),
                description=result.get("description", ""),
                severity=severity_map.get(
                    result.get("severity", "medium"), ConflictSeverity.MEDIUM
                ),
                confidence=float(result.get("confidence", 0.5)),
                recommendation=result.get("recommendation"),
            )

        except Exception as e:
            print(f"Warning: Failed to analyse conflict: {e}")
            return None

    def get_conflict_summary(self, conflicts: list[Conflict]) -> dict:
        """Generate a summary of detected conflicts."""
        summary = {
            "total_conflicts": len(conflicts),
            "by_severity": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
            },
            "by_type": {},
            "affected_documents": set(),
            "average_confidence": 0.0,
        }

        if not conflicts:
            return summary

        for conflict in conflicts:
            summary["by_severity"][conflict.severity.value] += 1

            conflict_type = conflict.conflict_type
            summary["by_type"][conflict_type] = (
                summary["by_type"].get(conflict_type, 0) + 1
            )

            summary["affected_documents"].add(conflict.chunk_a.source_file)
            summary["affected_documents"].add(conflict.chunk_b.source_file)

        summary["average_confidence"] = sum(c.confidence for c in conflicts) / len(
            conflicts
        )
        summary["affected_documents"] = list(summary["affected_documents"])

        return summary
