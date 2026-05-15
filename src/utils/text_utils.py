"""Text processing utilities."""

import re
from typing import Optional


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    separator: str = "\n",
) -> list[str]:
    """Split text into overlapping chunks."""
    text = clean_text(text)

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            last_sep = text.rfind(separator, start, end)
            if last_sep > start:
                end = last_sep + len(separator)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text) - overlap:
            break

    return chunks


def extract_key_phrases(text: str, max_phrases: int = 10) -> list[str]:
    """Extract key phrases from text using simple heuristics."""
    sentences = re.split(r"[.!?]+", text)
    phrases = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20:
            continue

        words = sentence.split()
        if 5 <= len(words) <= 30:
            phrases.append(sentence)

    phrase_scores = []
    for phrase in phrases:
        score = 0
        if phrase[0].isupper():
            score += 1
        if any(word in phrase.lower() for word in ["must", "should", "required", "shall"]):
            score += 2
        if any(word in phrase.lower() for word in ["important", "note", "warning", "caution"]):
            score += 2

        phrase_scores.append((phrase, score))

    phrase_scores.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in phrase_scores[:max_phrases]]


def calculate_text_similarity(text_a: str, text_b: str) -> float:
    """Calculate simple text similarity using Jaccard index."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b

    return len(intersection) / len(union) if union else 0.0


def truncate_text(
    text: str,
    max_length: int = 500,
    suffix: str = "...",
) -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text

    truncated = text[: max_length - len(suffix)]
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        truncated = truncated[:last_space]

    return truncated + suffix


def extract_section_title(text: str) -> Optional[str]:
    """Try to extract a section title from the beginning of text."""
    lines = text.split("\n")

    for line in lines[:3]:
        line = line.strip()
        if not line:
            continue

        if len(line) < 100 and (
            line.endswith(":")
            or line.isupper()
            or re.match(r"^\d+\.\s+\w+", line)
            or re.match(r"^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$", line)
        ):
            return line.rstrip(":")

    return None
