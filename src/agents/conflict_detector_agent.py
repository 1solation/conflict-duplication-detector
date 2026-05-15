"""AutoGen-based agent for conflict and duplication detection."""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from ..knowledge.document_loader import DocumentLoader
from ..knowledge.vector_store import VectorStore
from ..analysis.duplication_detector import DuplicationDetector, Duplication
from ..analysis.conflict_detector import ConflictDetector, Conflict
from ..analysis.inconsistency_detector import InconsistencyDetector, Inconsistency


@dataclass
class AnalysisResult:
    """Result of document analysis."""

    document_path: str
    chunks_added: int
    duplications: list[Duplication]
    conflicts: list[Conflict]
    inconsistencies: list[Inconsistency]
    summary: dict

    def to_dict(self) -> dict:
        return {
            "document_path": self.document_path,
            "chunks_added": self.chunks_added,
            "duplications": [d.to_dict() for d in self.duplications],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "inconsistencies": [i.to_dict() for i in self.inconsistencies],
            "summary": self.summary,
        }


AGENT_SYSTEM_MESSAGE = """You are an expert document analysis agent specialized in detecting conflicts, duplications, and inconsistencies in guidance documents.

You have access to the following tools:
1. analyze_document - Analyze a document for conflicts, duplications, and inconsistencies against the knowledge base
2. add_to_knowledge - Add a document to the knowledge base
3. search_knowledge - Search the knowledge base for relevant content
4. find_duplications - Find duplications across the knowledge base
5. check_conflicts - Check for conflicts on a specific topic
6. check_inconsistencies - Check for inconsistencies on a specific topic

When analyzing documents:
- First, check for semantic duplications with existing content
- Then, identify any conflicting information
- Finally, look for inconsistencies in terminology, formatting, or guidance

Provide clear, actionable reports highlighting issues and suggesting resolutions."""


class ConflictDetectorAgent:
    """AutoGen agent for document conflict and duplication detection."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "knowledge_base",
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        similarity_threshold: float = 0.85,
        conflict_confidence_threshold: float = 0.7,
    ):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.model = model
        self.similarity_threshold = similarity_threshold

        self.document_loader = DocumentLoader()
        self.vector_store = VectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_model=embedding_model,
            openai_api_key=self.api_key,
        )
        self.duplication_detector = DuplicationDetector(
            vector_store=self.vector_store,
            similarity_threshold=similarity_threshold,
        )
        self.conflict_detector = ConflictDetector(
            vector_store=self.vector_store,
            openai_api_key=self.api_key,
            model=model,
            confidence_threshold=conflict_confidence_threshold,
        )
        self.inconsistency_detector = InconsistencyDetector(
            vector_store=self.vector_store,
            openai_api_key=self.api_key,
            model=model,
        )

        self.model_client = OpenAIChatCompletionClient(
            model=model,
            api_key=self.api_key,
        )

        self._tools = self._create_tools()
        self._agent: Optional[AssistantAgent] = None

    def _create_tools(self) -> list[Callable]:
        """Create tool functions for the agent."""

        async def analyze_document(file_path: str) -> str:
            """Analyze a document for conflicts, duplications, and inconsistencies."""
            try:
                result = self.analyze_document_sync(file_path)
                return json.dumps(result.to_dict(), indent=2)
            except Exception as e:
                return json.dumps({"error": str(e)})

        async def add_to_knowledge(file_path: str) -> str:
            """Add a document to the knowledge base."""
            try:
                chunks = self.document_loader.load_document(file_path)
                added = self.vector_store.add_chunks(chunks)
                return json.dumps({
                    "success": True,
                    "file": file_path,
                    "chunks_added": added,
                    "total_chunks": self.vector_store.count,
                })
            except Exception as e:
                return json.dumps({"error": str(e)})

        async def search_knowledge(query: str, n_results: int = 5) -> str:
            """Search the knowledge base for relevant content."""
            try:
                results = self.vector_store.search(query, n_results=n_results)
                return json.dumps([r.to_dict() for r in results], indent=2)
            except Exception as e:
                return json.dumps({"error": str(e)})

        async def find_duplications(check_within_document: bool = False) -> str:
            """Find all duplications in the knowledge base."""
            try:
                duplications = self.duplication_detector.find_all_duplications(
                    check_within_document=check_within_document
                )
                summary = self.duplication_detector.get_duplication_summary(duplications)
                return json.dumps({
                    "summary": summary,
                    "duplications": [d.to_dict() for d in duplications[:20]],
                }, indent=2)
            except Exception as e:
                return json.dumps({"error": str(e)})

        async def check_conflicts(topic: str) -> str:
            """Check for conflicts related to a specific topic."""
            try:
                results = self.vector_store.search(topic, n_results=10)
                if len(results) < 2:
                    return json.dumps({"message": "Not enough related content found"})

                from ..knowledge.document_loader import DocumentChunk

                chunks = [
                    DocumentChunk(
                        content=r.content,
                        document_id=r.document_id,
                        chunk_id=r.chunk_id,
                        source_file=r.source_file,
                        page_number=r.page_number,
                        section_title=r.section_title,
                    )
                    for r in results
                ]

                conflicts = self.conflict_detector.detect_conflicts(chunks)
                summary = self.conflict_detector.get_conflict_summary(conflicts)

                return json.dumps({
                    "topic": topic,
                    "summary": summary,
                    "conflicts": [c.to_dict() for c in conflicts],
                }, indent=2)
            except Exception as e:
                return json.dumps({"error": str(e)})

        async def check_inconsistencies(topic: str) -> str:
            """Check for inconsistencies related to a specific topic."""
            try:
                inconsistencies = self.inconsistency_detector.detect_inconsistencies(
                    topic_query=topic,
                    n_chunks=10,
                )
                summary = self.inconsistency_detector.get_inconsistency_summary(
                    inconsistencies
                )

                return json.dumps({
                    "topic": topic,
                    "summary": summary,
                    "inconsistencies": [i.to_dict() for i in inconsistencies],
                }, indent=2)
            except Exception as e:
                return json.dumps({"error": str(e)})

        return [
            analyze_document,
            add_to_knowledge,
            search_knowledge,
            find_duplications,
            check_conflicts,
            check_inconsistencies,
        ]

    def get_agent(self) -> AssistantAgent:
        """Get or create the AutoGen agent."""
        if self._agent is None:
            self._agent = AssistantAgent(
                name="ConflictDetector",
                model_client=self.model_client,
                system_message=AGENT_SYSTEM_MESSAGE,
                tools=self._tools,
            )
        return self._agent

    def analyze_document_sync(self, file_path: str) -> AnalysisResult:
        """Synchronously analyze a document."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        chunks = self.document_loader.load_document(path)

        duplications = self.duplication_detector.find_duplications(chunks)
        conflicts = self.conflict_detector.detect_conflicts(chunks)
        inconsistencies = []

        if chunks:
            inconsistencies = self.inconsistency_detector.detect_inconsistencies_for_chunks(
                chunks[:10]
            )

        added = self.vector_store.add_chunks(chunks)

        summary = {
            "document": str(path),
            "total_chunks": len(chunks),
            "chunks_added_to_knowledge": added,
            "duplications_found": len(duplications),
            "conflicts_found": len(conflicts),
            "inconsistencies_found": len(inconsistencies),
            "duplication_summary": self.duplication_detector.get_duplication_summary(
                duplications
            ),
            "conflict_summary": self.conflict_detector.get_conflict_summary(conflicts),
            "inconsistency_summary": self.inconsistency_detector.get_inconsistency_summary(
                inconsistencies
            ),
        }

        return AnalysisResult(
            document_path=str(path),
            chunks_added=added,
            duplications=duplications,
            conflicts=conflicts,
            inconsistencies=inconsistencies,
            summary=summary,
        )

    async def analyze_document_async(self, file_path: str) -> AnalysisResult:
        """Asynchronously analyze a document."""
        return await asyncio.to_thread(self.analyze_document_sync, file_path)

    async def run(self, task: str) -> str:
        """Run the agent with a specific task."""
        agent = self.get_agent()
        result = await agent.run(task=task)
        return str(result.messages[-1].content) if result.messages else ""

    async def chat(self, message: str) -> str:
        """Send a message to the agent and get a response."""
        return await self.run(message)

    def add_knowledge_directory(self, directory_path: str) -> dict:
        """Add all documents from a directory to the knowledge base."""
        chunks = self.document_loader.load_directory(directory_path)
        added = self.vector_store.add_chunks(chunks)

        return {
            "directory": directory_path,
            "total_chunks_loaded": len(chunks),
            "chunks_added": added,
            "total_in_knowledge_base": self.vector_store.count,
        }

    def clear_knowledge_base(self) -> None:
        """Clear all documents from the knowledge base."""
        self.vector_store.clear()

    @property
    def knowledge_base_count(self) -> int:
        """Return the number of chunks in the knowledge base."""
        return self.vector_store.count
