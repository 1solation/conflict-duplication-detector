"""Main entry point for the Conflict and Duplication Detection Agent."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

from .agents.conflict_detector_agent import ConflictDetectorAgent

load_dotenv()

app = typer.Typer(
    name="conflict-detector",
    help="Detect conflicts, duplications, and inconsistencies in documents.",
)
console = Console()


def get_agent(
    persist_dir: Optional[str] = None,
    collection: str = "knowledge_base",
) -> ConflictDetectorAgent:
    """Create and return the agent instance."""
    return ConflictDetectorAgent(
        persist_directory=persist_dir or os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
        collection_name=collection,
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.85")),
        conflict_confidence_threshold=float(
            os.getenv("CONFLICT_CONFIDENCE_THRESHOLD", "0.7")
        ),
    )


@app.command()
def analyse(
    file_path: str = typer.Argument(..., help="Path to the document to analyse"),
    persist_dir: Optional[str] = typer.Option(
        None, "--persist-dir", "-d", help="ChromaDB persistence directory"
    ),
    collection: str = typer.Option(
        "knowledge_base", "--collection", "-c", help="ChromaDB collection name"
    ),
    output_json: bool = typer.Option(
        False, "--json", "-j", help="Output results as JSON"
    ),
):
    """Analyse a document for conflicts, duplications, and inconsistencies."""
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Analyzing document...", total=None)

        try:
            agent = get_agent(persist_dir, collection)
            result = agent.analyse_document_sync(str(path))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    if output_json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    console.print()
    console.print(Panel(f"[bold]Analysis Results for {path.name}[/bold]"))

    summary_table = Table(title="Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Total Chunks", str(result.summary["total_chunks"]))
    summary_table.add_row(
        "Chunks Added to Knowledge Base",
        str(result.summary["chunks_added_to_knowledge"]),
    )
    summary_table.add_row("Duplications Found", str(len(result.duplications)))
    summary_table.add_row("Conflicts Found", str(len(result.conflicts)))
    summary_table.add_row("Inconsistencies Found", str(len(result.inconsistencies)))

    console.print(summary_table)

    if result.duplications:
        console.print()
        dup_table = Table(title="[yellow]Duplications[/yellow]")
        dup_table.add_column("Type", style="yellow")
        dup_table.add_column("Similarity", style="green")
        dup_table.add_column("Source", style="cyan")
        dup_table.add_column("Similar To", style="cyan")

        for dup in result.duplications[:10]:
            dup_table.add_row(
                dup.duplication_type,
                f"{dup.similarity_score:.1%}",
                Path(dup.source_chunk.source_file).name,
                Path(dup.similar_chunk.source_file).name,
            )

        console.print(dup_table)

    if result.conflicts:
        console.print()
        conflict_table = Table(title="[red]Conflicts[/red]")
        conflict_table.add_column("Severity", style="red")
        conflict_table.add_column("Type", style="yellow")
        conflict_table.add_column("Description", style="white")
        conflict_table.add_column("Confidence", style="green")

        for conflict in result.conflicts[:10]:
            conflict_table.add_row(
                conflict.severity.value.upper(),
                conflict.conflict_type,
                conflict.description,
                f"{conflict.confidence:.0%}",
            )

        console.print(conflict_table)

    if result.inconsistencies:
        console.print()
        inc_table = Table(title="[magenta]Inconsistencies[/magenta]")
        inc_table.add_column("Type", style="magenta")
        inc_table.add_column("Description", style="white")
        inc_table.add_column("Confidence", style="green")

        for inc in result.inconsistencies[:10]:
            inc_table.add_row(
                inc.inconsistency_type.value,
                inc.description,
                f"{inc.confidence:.0%}",
            )

        console.print(inc_table)


@app.command()
def add(
    path: str = typer.Argument(..., help="Path to file or directory to add"),
    persist_dir: Optional[str] = typer.Option(
        None, "--persist-dir", "-d", help="ChromaDB persistence directory"
    ),
    collection: str = typer.Option(
        "knowledge_base", "--collection", "-c", help="ChromaDB collection name"
    ),
):
    """Add documents to the knowledge base."""
    target = Path(path)
    if not target.exists():
        console.print(f"[red]Error: Path not found: {path}[/red]")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Adding to knowledge base...", total=None)

        agent = get_agent(persist_dir, collection)

        if target.is_dir():
            result = agent.add_knowledge_directory(str(target))
            console.print(
                f"[green]Added {result['chunks_added']} chunks from {result['directory']}[/green]"
            )
        else:
            from .knowledge.document_loader import DocumentLoader

            loader = DocumentLoader()
            chunks = loader.load_document(target)
            added = agent.vector_store.add_chunks(chunks)
            console.print(f"[green]Added {added} chunks from {target.name}[/green]")

    console.print(f"Total chunks in knowledge base: {agent.knowledge_base_count}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    n_results: int = typer.Option(5, "--results", "-n", help="Number of results"),
    persist_dir: Optional[str] = typer.Option(
        None, "--persist-dir", "-d", help="ChromaDB persistence directory"
    ),
    collection: str = typer.Option(
        "knowledge_base", "--collection", "-c", help="ChromaDB collection name"
    ),
):
    """Search the knowledge base."""
    agent = get_agent(persist_dir, collection)
    results = agent.vector_store.search(query, n_results=n_results)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f"Search Results for: {query}")
    table.add_column("Score", style="green")
    table.add_column("Source", style="cyan")
    table.add_column("Section", style="yellow")
    table.add_column("Content", style="white")

    for result in results:
        content = result.content[:100] + "..." if len(result.content) > 100 else result.content
        table.add_row(
            f"{result.similarity_score:.2%}",
            Path(result.source_file).name,
            result.section_title or "N/A",
            content,
        )

    console.print(table)


@app.command()
def duplications(
    persist_dir: Optional[str] = typer.Option(
        None, "--persist-dir", "-d", help="ChromaDB persistence directory"
    ),
    collection: str = typer.Option(
        "knowledge_base", "--collection", "-c", help="ChromaDB collection name"
    ),
    within_document: bool = typer.Option(
        False, "--within", "-w", help="Check duplications within same document"
    ),
    output_json: bool = typer.Option(
        False, "--json", "-j", help="Output results as JSON"
    ),
):
    """Find duplications across the knowledge base."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Finding duplications...", total=None)

        agent = get_agent(persist_dir, collection)
        dups = agent.duplication_detector.find_all_duplications(
            check_within_document=within_document
        )
        summary = agent.duplication_detector.get_duplication_summary(dups)

    if output_json:
        print(json.dumps({
            "summary": summary,
            "duplications": [d.to_dict() for d in dups],
        }, indent=2))
        return

    console.print(Panel(f"[bold]Found {len(dups)} duplications[/bold]"))

    if dups:
        table = Table(title="Duplications")
        table.add_column("Type", style="yellow")
        table.add_column("Similarity", style="green")
        table.add_column("File A", style="cyan")
        table.add_column("File B", style="cyan")

        for dup in dups[:20]:
            table.add_row(
                dup.duplication_type,
                f"{dup.similarity_score:.1%}",
                Path(dup.source_chunk.source_file).name,
                Path(dup.similar_chunk.source_file).name,
            )

        console.print(table)


@app.command()
def conflicts(
    topic: str = typer.Argument(..., help="Topic to check for conflicts"),
    persist_dir: Optional[str] = typer.Option(
        None, "--persist-dir", "-d", help="ChromaDB persistence directory"
    ),
    collection: str = typer.Option(
        "knowledge_base", "--collection", "-c", help="ChromaDB collection name"
    ),
    output_json: bool = typer.Option(
        False, "--json", "-j", help="Output results as JSON"
    ),
):
    """Check for conflicts on a specific topic."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Checking for conflicts...", total=None)

        agent = get_agent(persist_dir, collection)

        results = agent.vector_store.search(topic, n_results=10)
        if len(results) < 2:
            console.print("[yellow]Not enough related content found.[/yellow]")
            return

        from .knowledge.document_loader import DocumentChunk

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

        conflicts_found = agent.conflict_detector.detect_conflicts(chunks)
        summary = agent.conflict_detector.get_conflict_summary(conflicts_found)

    if output_json:
        print(json.dumps({
            "topic": topic,
            "summary": summary,
            "conflicts": [c.to_dict() for c in conflicts_found],
        }, indent=2))
        return

    console.print(Panel(f"[bold]Conflicts for topic: {topic}[/bold]"))
    console.print(f"Found {len(conflicts_found)} conflicts")

    if conflicts_found:
        table = Table(title="Conflicts")
        table.add_column("Severity", style="red")
        table.add_column("Type", style="yellow")
        table.add_column("Description", style="white")
        table.add_column("Confidence", style="green")

        for conflict in conflicts_found:
            table.add_row(
                conflict.severity.value.upper(),
                conflict.conflict_type,
                conflict.description,
                f"{conflict.confidence:.0%}",
            )

        console.print(table)


@app.command()
def clear(
    persist_dir: Optional[str] = typer.Option(
        None, "--persist-dir", "-d", help="ChromaDB persistence directory"
    ),
    collection: str = typer.Option(
        "knowledge_base", "--collection", "-c", help="ChromaDB collection name"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip confirmation"
    ),
):
    """Clear the knowledge base."""
    if not force:
        confirm = typer.confirm("Are you sure you want to clear the knowledge base?")
        if not confirm:
            raise typer.Abort()

    agent = get_agent(persist_dir, collection)
    agent.clear_knowledge_base()
    console.print("[green]Knowledge base cleared.[/green]")


@app.command()
def chat(
    persist_dir: Optional[str] = typer.Option(
        None, "--persist-dir", "-d", help="ChromaDB persistence directory"
    ),
    collection: str = typer.Option(
        "knowledge_base", "--collection", "-c", help="ChromaDB collection name"
    ),
):
    """Start an interactive chat session with the agent."""
    agent = get_agent(persist_dir, collection)

    console.print(Panel(
        "[bold]Conflict Detection Agent[/bold]\n\n"
        "Commands:\n"
        "  /analyse <file>  - Analyse a document\n"
        "  /add <path>      - Add to knowledge base\n"
        "  /search <query>  - Search knowledge base\n"
        "  /quit            - Exit chat\n\n"
        "Or just ask questions about conflicts and duplications.",
        title="Interactive Mode"
    ))

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not user_input.strip():
            continue

        if user_input.strip().lower() in ["/quit", "/exit", "quit", "exit"]:
            console.print("[yellow]Goodbye![/yellow]")
            break

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Thinking...", total=None)
            response = asyncio.run(agent.chat(user_input))

        console.print(f"[bold green]Agent:[/bold green] {response}")
        console.print()


@app.command()
def info(
    persist_dir: Optional[str] = typer.Option(
        None, "--persist-dir", "-d", help="ChromaDB persistence directory"
    ),
    collection: str = typer.Option(
        "knowledge_base", "--collection", "-c", help="ChromaDB collection name"
    ),
):
    """Show information about the knowledge base."""
    agent = get_agent(persist_dir, collection)

    table = Table(title="Knowledge Base Info")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Persistence Directory", persist_dir or os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"))
    table.add_row("Collection Name", collection)
    table.add_row("Total Chunks", str(agent.knowledge_base_count))

    console.print(table)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
