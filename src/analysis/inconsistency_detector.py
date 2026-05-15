"""Inconsistency detection for document guidance analysis."""

import os
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from openai import OpenAI

from ..knowledge.document_loader import DocumentChunk
from ..knowledge.vector_store import VectorStore


class InconsistencyType(Enum):
    """Types of inconsistencies that can be detected."""

    TERMINOLOGY = "terminology"  # Different terms for the same concept
    FORMAT = "format"  # Inconsistent formatting or structure
    TONE = "tone"  # Inconsistent tone or voice
    VERSIONING = "versioning"  # Outdated vs current information
    SCOPE = "scope"  # Overlapping but different scopes
    REFERENCE = "reference"  # Broken or inconsistent references


@dataclass
class Inconsistency:
    """Represents a detected inconsistency."""

    chunks_involved: list[dict]
    inconsistency_type: InconsistencyType
    description: str
    impact: str
    suggestion: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "chunks_involved": self.chunks_involved,
            "inconsistency_type": self.inconsistency_type.value,
            "description": self.description,
            "impact": self.impact,
            "suggestion": self.suggestion,
            "confidence": round(self.confidence, 2),
        }

    def __str__(self) -> str:
        files = ", ".join(c.get("source_file", "unknown") for c in self.chunks_involved)
        return (
            f"Inconsistency [{self.inconsistency_type.value}]\n"
            f"  Description: {self.description}\n"
            f"  Files: {files}\n"
            f"  Impact: {self.impact}\n"
            f"  Suggestion: {self.suggestion}"
        )


INCONSISTENCY_ANALYSIS_PROMPT = """You are an expert document analyst specializing in identifying inconsistencies in guidance documents.

Analyse the following set of related passages from different documents and identify any inconsistencies.

PASSAGES:
{passages}

Look for these types of inconsistencies:
1. TERMINOLOGY: Different terms used for the same concept
2. FORMAT: Inconsistent formatting, numbering, or structure
3. TONE: Inconsistent tone, voice, or level of formality
4. VERSIONING: Mix of outdated and current information
5. SCOPE: Overlapping but inconsistently defined scopes
6. REFERENCE: Broken or inconsistent cross-references

For each inconsistency found, respond with a JSON array:
{{
    "inconsistencies": [
        {{
            "type": "terminology|format|tone|versioning|scope|reference",
            "description": "Clear description of the inconsistency",
            "passages_involved": [0, 1],  // indices of passages involved
            "impact": "How this inconsistency affects users",
            "suggestion": "How to resolve this inconsistency",
            "confidence": 0.0-1.0
        }}
    ]
}}

If no inconsistencies are found, return {{"inconsistencies": []}}
Be thorough but precise - only report genuine inconsistencies that would confuse or mislead users."""


class InconsistencyDetector:
    """Detects inconsistencies across related document sections."""

    def __init__(
        self,
        vector_store: VectorStore,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o",
        confidence_threshold: float = 0.6,
    ):
        self.vector_store = vector_store
        self.model = model
        self.confidence_threshold = confidence_threshold

        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.client = OpenAI(api_key=api_key)

    def detect_inconsistencies(
        self,
        topic_query: str,
        n_chunks: int = 10,
    ) -> list[Inconsistency]:
        """Find inconsistencies among chunks related to a specific topic."""
        search_results = self.vector_store.search(
            query=topic_query,
            n_results=n_chunks,
        )

        if len(search_results) < 2:
            return []

        return self._analyse_inconsistencies(search_results)

    def detect_inconsistencies_for_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[Inconsistency]:
        """Find inconsistencies among a specific set of chunks."""
        if len(chunks) < 2:
            return []

        chunk_dicts = [
            {
                "content": chunk.content,
                "source_file": chunk.source_file,
                "section_title": chunk.section_title,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        ]

        return self._analyse_chunk_list(chunk_dicts)

    def _analyse_inconsistencies(
        self,
        search_results: list,
    ) -> list[Inconsistency]:
        """Analyse search results for inconsistencies."""
        chunk_dicts = [
            {
                "content": result.content,
                "source_file": result.source_file,
                "section_title": result.section_title,
                "chunk_id": result.chunk_id,
            }
            for result in search_results
        ]

        return self._analyse_chunk_list(chunk_dicts)

    def _analyse_chunk_list(
        self,
        chunk_dicts: list[dict],
    ) -> list[Inconsistency]:
        """Use LLM to analyse chunks for inconsistencies."""
        passages_text = ""
        for i, chunk in enumerate(chunk_dicts):
            passages_text += f"\n[Passage {i}] From: {chunk['source_file']}"
            if chunk.get("section_title"):
                passages_text += f" - Section: {chunk['section_title']}"
            passages_text += f"\n{chunk['content']}\n"

        prompt = INCONSISTENCY_ANALYSIS_PROMPT.format(passages=passages_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            import json

            result = json.loads(response.choices[0].message.content)
            inconsistencies_data = result.get("inconsistencies", [])

            inconsistencies = []
            type_map = {
                "terminology": InconsistencyType.TERMINOLOGY,
                "format": InconsistencyType.FORMAT,
                "tone": InconsistencyType.TONE,
                "versioning": InconsistencyType.VERSIONING,
                "scope": InconsistencyType.SCOPE,
                "reference": InconsistencyType.REFERENCE,
            }

            for inc_data in inconsistencies_data:
                confidence = float(inc_data.get("confidence", 0.5))
                if confidence < self.confidence_threshold:
                    continue

                passage_indices = inc_data.get("passages_involved", [])
                chunks_involved = [
                    chunk_dicts[i]
                    for i in passage_indices
                    if i < len(chunk_dicts)
                ]

                inc_type = type_map.get(
                    inc_data.get("type", ""), InconsistencyType.TERMINOLOGY
                )

                inconsistencies.append(
                    Inconsistency(
                        chunks_involved=chunks_involved,
                        inconsistency_type=inc_type,
                        description=inc_data.get("description", ""),
                        impact=inc_data.get("impact", ""),
                        suggestion=inc_data.get("suggestion", ""),
                        confidence=confidence,
                    )
                )

            return inconsistencies

        except Exception as e:
            print(f"Warning: Failed to analyse inconsistencies: {e}")
            return []

    def get_inconsistency_summary(
        self, inconsistencies: list[Inconsistency]
    ) -> dict:
        """Generate a summary of detected inconsistencies."""
        summary = {
            "total_inconsistencies": len(inconsistencies),
            "by_type": {},
            "affected_documents": set(),
            "average_confidence": 0.0,
        }

        if not inconsistencies:
            return summary

        for inc in inconsistencies:
            inc_type = inc.inconsistency_type.value
            summary["by_type"][inc_type] = summary["by_type"].get(inc_type, 0) + 1

            for chunk in inc.chunks_involved:
                if "source_file" in chunk:
                    summary["affected_documents"].add(chunk["source_file"])

        summary["average_confidence"] = sum(
            i.confidence for i in inconsistencies
        ) / len(inconsistencies)
        summary["affected_documents"] = list(summary["affected_documents"])

        return summary
