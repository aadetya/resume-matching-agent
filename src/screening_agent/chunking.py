"""Overlapping windows whose evidence remains an exact source substring."""

from __future__ import annotations

import re

from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: str
    candidate_id: str
    text: str
    start: int
    end: int


def chunk_text(
    text: str, candidate_id: str, *, window: int = 180, overlap: int = 40
) -> list[Chunk]:
    """Select whitespace-token windows; never reconstruct source whitespace.

    These are extraction windows, not a claim about an embedding tokenizer's
    budget. The embedding provider applies its own tokenizer before inference.
    """
    if window < 8 or not 0 <= overlap < window:
        raise ValueError("window must be at least 8 and 0 <= overlap < window")
    tokens = list(re.finditer(r"\S+", text))
    output: list[Chunk] = []
    index = 0
    while index < len(tokens):
        stop = min(index + window, len(tokens))
        start, end = tokens[index].start(), tokens[stop - 1].end()
        output.append(
            Chunk(
                chunk_id=f"{candidate_id}:{start}:{end}",
                candidate_id=candidate_id,
                text=text[start:end],
                start=start,
                end=end,
            )
        )
        if stop == len(tokens):
            break
        index = stop - overlap
    return output
