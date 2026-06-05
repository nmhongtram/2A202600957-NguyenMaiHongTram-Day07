from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        # Split on sentence-ending punctuation followed by whitespace or newline
        raw = re.split(r'(?<=[.!?]) +|(?<=\.)\n', text)
        sentences = [s.strip() for s in raw if s.strip()]
        if not sentences:
            return [text]

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, list(self.separators))

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            return [current_text]

        sep = remaining_separators[0]
        rest = remaining_separators[1:]

        # Empty string = last-resort character-level fixed-size split
        if sep == "":
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        parts = current_text.split(sep)
        if len(parts) <= 1:
            # Separator not found — try the next one
            return self._split(current_text, rest)

        result: list[str] = []
        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue
            if len(stripped) <= self.chunk_size:
                result.append(stripped)
            else:
                result.extend(self._split(stripped, rest))

        return result if result else [current_text]


class SectionChunker:
    """
    Split text at Vietnamese legal section/heading boundaries.

    Detects heading lines that start with:  Chương / Mục / Điều  (and uppercase variants).
    Each chunk = heading + following content lines until the next heading.
    Oversized sections (> max_size chars) are subdivided with SentenceChunker(4).
    Undersized sections (< min_size chars) are merged into the following chunk.
    """

    _HEADING_RE = re.compile(
        r'^(Chương|CHƯƠNG|Mục|MỤC|Điều|ĐIỀU)\s+'
    )

    def __init__(self, max_size: int = 800, min_size: int = 80) -> None:
        self.max_size = max_size
        self.min_size = min_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # ── 1. Split into raw sections at each heading boundary ──────────────
        lines = text.split('\n')
        raw_sections: list[str] = []
        current: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if self._HEADING_RE.match(stripped):
                if current:
                    raw_sections.append('\n'.join(current))
                current = [stripped]
            else:
                current.append(stripped)

        if current:
            raw_sections.append('\n'.join(current))

        if not raw_sections:
            return [text]

        # ── 2. Post-process: subdivide oversized, merge undersized ────────────
        result: list[str] = []
        pending = ''          # accumulator for undersized sections

        for section in raw_sections:
            if pending:
                merged = pending + '\n' + section
                if len(merged) <= self.max_size:
                    pending = merged
                    continue
                else:
                    # Flush pending before starting fresh
                    result.append(pending)
                    pending = ''

            if len(section) > self.max_size:
                subs = SentenceChunker(max_sentences_per_chunk=4).chunk(section)
                result.extend(subs)
            elif len(section) < self.min_size:
                pending = section
            else:
                result.append(section)

        if pending:
            result.append(pending)

        return [c.strip() for c in result if c.strip()]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=0),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }
        result: dict = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            avg_length = sum(len(c) for c in chunks) / len(chunks) if chunks else 0.0
            result[name] = {
                "count": len(chunks),
                "avg_length": avg_length,
                "chunks": chunks,
            }
        return result
