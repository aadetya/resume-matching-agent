"""Measure real three-round conversations with isolated, initially empty assessment caches."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matching_agent import MatchingSession  # noqa: E402
from screening_agent.planner import OpenAIPlanner  # noqa: E402
from screening_agent.policy import fingerprint  # noqa: E402
from screening_agent.retrieval import ResumeIndex  # noqa: E402

QUERIES = [
    "Need a backend developer with 4+ years of experience and Python tech stack.",
    "Find software developers with Python and SQL and at least 3 years total professional experience.",
    "Keep Python, software development and 3 years total experience. SQL means practical relational database work: PostgreSQL development or SQLAlchemy querying and modeling counts; handwritten SQL is not required.",
    "Find React web developers with at least 3 years total employment. Mobile-only React Native work does not meet the web requirement.",
    "Find candidates with 3 years of hands-on Python as a substantive job responsibility; isolated scripts do not count for the whole job.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--initial-only",
        action="store_true",
        help="Benchmark retrieval and interpretation without detailed rounds.",
    )
    parser.add_argument("--query-limit", type=int, default=5, choices=range(1, 6))
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = root / "artifacts/stage_benchmarks" / uuid4().hex
    cache.mkdir(parents=True)
    initial_index_started = time.perf_counter()
    index = ResumeIndex(root)
    index_seconds = time.perf_counter() - initial_index_started
    receipt = {
        "queries": QUERIES[: args.query_limit],
        "initial_index_load_seconds": index_seconds,
        "index_fingerprint": index.engine_fingerprint,
        "candidate_count": len(index.candidates),
        "assessment_cache_initially_empty": True,
        "scope": "Development conversations; no human ground-truth accuracy claim.",
        "runs": [],
    }
    previous = None
    for n, query in enumerate(QUERIES[: args.query_limit], 1):
        if n == 3:
            session = previous
        else:
            planner = OpenAIPlanner.from_env(root)
            session = MatchingSession(
                root, index=index, planner=planner, db_path=cache / "sessions.sqlite"
            )
            session.registry.requirement_compiler.cache_dir = cache / "standards"
            session.registry.requirement_compiler.cache_dir.mkdir(exist_ok=True)
            session.registry.deep_reviewer.root = cache
            session.registry.deep_reviewer.cache_dir = cache / "assessments"
            session.registry.deep_reviewer.cache_dir.mkdir(exist_ok=True)
        stages = [("initial", query)]
        if not args.initial_only:
            stages += [
                ("deep", "Run detailed screening of the top 10"),
                ("final", "Generate final screening recommendations"),
            ]
        for stage, request in stages:
            before_events = len(session.snapshot().get("tool_events", []))
            started = time.perf_counter()
            progress = []
            for event in session.stream(request):
                if "__progress__" in event:
                    progress.append(
                        {
                            "seconds": round(time.perf_counter() - started, 3),
                            **event["__progress__"],
                        }
                    )
            state = session.snapshot()
            duration = time.perf_counter() - started
            (output / f"q{n}-{stage}.json").write_text(json.dumps(state, indent=2))
            events = state["tool_events"][before_events:]
            record = {
                "query": n,
                "stage": stage,
                "seconds": round(duration, 3),
                "error": state.get("error"),
                "model": session.registry.deep_reviewer.model,
                "reviewer_signature": session.registry.deep_reviewer.signature,
                "shortlist": [m["candidate_id"] for m in state["shortlist"]],
                "tool_events": events,
                "progress": progress,
            }
            if stage == "initial" and not state.get("error") and not state.get("clarification"):
                record["candidate_assessments"] = len(state.get("detailed_reviews", []))
                assert not state.get("detailed_reviews") and not any(
                    m["assessments"] for m in state["shortlist"]
                )
                assert all(e["tool"] != "contextual_screen" for e in events)
            if stage == "deep":
                record["pending"] = sum(
                    r["stage"] == "pending" for r in state.get("detailed_reviews", [])
                )
                record["model_calls"] = sum(
                    r["model_calls"] for r in state.get("detailed_reviews", [])
                )
                record["cache_hits"] = sum(
                    r["cache_hits"] for r in state.get("detailed_reviews", [])
                )
            if stage == "final" and not state.get("error"):
                record["model_calls"] = state["decision_memo"]["model_calls"]
                record["comparisons"] = len(state["decision_memo"]["comparisons"])
                record["checked_claims"] = len(state["decision_memo"]["audit"]["claims"])
            receipt["runs"].append(record)
            receipt["integrity"] = fingerprint(
                {k: v for k, v in receipt.items() if k != "integrity"}
            )
            (output / "results.json").write_text(json.dumps(receipt, indent=2))
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in record.items()
                        if k not in {"tool_events", "progress", "reviewer_signature"}
                    }
                ),
                flush=True,
            )
            if state.get("error"):
                break
        previous = session
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
