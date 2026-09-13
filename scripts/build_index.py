"""Build the versioned local index; malformed inputs remain visible in a report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from screening_agent.retrieval import build_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--overlap", type=int, default=25)
    parser.add_argument("--ner-backend", choices=["spacy", "rules"], default="spacy")
    parser.add_argument("--lexical-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    manifest = build_index(
        args.root,
        ner_backend=args.ner_backend,
        semantic=not args.lexical_only,
        window=args.window,
        overlap=args.overlap,
        index_dir=args.index_dir,
        strict=not args.allow_partial,
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in [
                    "candidate_count",
                    "chunk_count",
                    "ner_model",
                    "embedding_model",
                    "window",
                    "overlap",
                ]
            },
            indent=2,
        )
    )
