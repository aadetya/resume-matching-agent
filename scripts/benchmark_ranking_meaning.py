"""Run controlled meaning comparisons through the application's active ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from screening_agent.neural_ranking import (
    RANKING_INSTRUCTION,
    RERANKER_MODEL,
    RERANKER_REVISION,
    score_complete_pairs,
)
from screening_agent.retrieval import _reranker

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/ranking_validation/meaning.json"
    )
    args = parser.parse_args()
    path = ROOT / "data/evaluation/ranking_meaning_cases.json"
    source = json.loads(path.read_text())
    cases = source["cases"]
    pairs = [(case["query"], case[key]) for case in cases for key in ("supported", "insufficient")]
    started = time.perf_counter()
    model = _reranker()
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    values, tokens = score_complete_pairs(model, pairs)
    elapsed = time.perf_counter() - started
    rows = [
        {
            "id": case["id"],
            "category": case["category"],
            "domain": case["domain"],
            "supported_logit": float(values[2 * i]),
            "insufficient_logit": float(values[2 * i + 1]),
            "correct": bool(values[2 * i] > values[2 * i + 1]),
        }
        for i, case in enumerate(cases)
    ]
    receipt = {
        "schema_version": 1,
        "model": RERANKER_MODEL,
        "revision": RERANKER_REVISION,
        "device": str(model.device),
        "instruction": RANKING_INSTRUCTION,
        "scope": "Model-selection diagnostics, including occupational transfer cases. Authored comparisons, no independent human review or universal accuracy claim.",
        "cases_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            (ROOT / "src/screening_agent/neural_ranking.py").read_bytes()
        ).hexdigest(),
        "load_seconds": load_seconds,
        "inference_seconds": elapsed,
        "max_pair_tokens": max(tokens),
        "passed": sum(row["correct"] for row in rows),
        "total": len(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "rows"}), flush=True)
    return 0 if receipt["passed"] == receipt["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
