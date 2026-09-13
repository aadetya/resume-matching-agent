"""Evaluate ranking separately from real conversational interpretation.

Frozen mode isolates retrieval under supplied requirements. Conversation mode
sends the actual query through MatchingSession, without injecting expected
requirements. Labels are used only after the application returns its result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from screening_agent.contracts import Requirements
from screening_agent.retrieval import ResumeIndex, build_index

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collection(name):
    if name == "holdout":
        directory = ROOT / "data/holdout"
        query_path = directory / "queries.json"
        label_path = directory / "judgments.json"
        queries = json.loads(query_path.read_text())
        labels = json.loads(label_path.read_text())
        grades = {
            q["job_id"]: {
                row["candidate_id"]: next(
                    j["grade"] for j in row["judgments"] if j["job_id"] == q["job_id"]
                )
                for row in labels["candidates"]
            }
            for q in queries
        }
        files = [
            ROOT / row["path"]
            for row in json.loads((directory / "corpus_manifest.json").read_text())["files"]
        ]
        label_scope = "Development corpus: 50 People profiles, two independent agent reviews and adjudication; no human review. Frozen legacy rubric, including its original date convention."
    else:
        directory = ROOT / "data/regression/data"
        query_path = label_path = directory / "ground_truth/queries.json"
        queries = json.loads(query_path.read_text())
        grades = {q["job_id"]: q["grades"] for q in queries}
        files = sorted((directory / "resumes").glob("*"))
        label_scope = "Original 100 synthetic fixtures with generator-authored grades; constructed regression material, not independent human judgments."
    return (
        queries,
        grades,
        files,
        {
            "collection": name,
            "query_sha256": sha(query_path),
            "labels_sha256": sha(label_path),
            "label_scope": label_scope,
            "source_sha256": {p.stem: sha(p) for p in files},
        },
    )


def prepare_index(destination, files):
    (destination / "config").mkdir(parents=True)
    shutil.copyfile(ROOT / "config/skills.yaml", destination / "config/skills.yaml")
    resumes = destination / "data/resumes"
    resumes.mkdir(parents=True)
    for path in files:
        shutil.copyfile(path, resumes / path.name)
    # Preserve the dataset's identity authority. Guessing display names from a
    # rendered header would change identity masking and therefore model inputs.
    manifest_path = files[0].parent.parent / "corpus_manifest.json" if files else None
    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") == 2:
            selected = {p.name for p in files}
            rows = [row for row in manifest["files"] if Path(row["path"]).name in selected]
            if len(rows) != len(files):
                raise ValueError(
                    "Benchmark source identities must cover every document exactly once."
                )
            manifest["files"] = [
                {**row, "path": "data/resumes/" + Path(row["path"]).name, "split": "production"}
                for row in rows
            ]
            # 'production' is the index allowlist role inside this temporary root.
            # Original development split/provenance remains in the benchmark receipt.
            (destination / "data/corpus_manifest.json").write_text(json.dumps(manifest))
    build_index(destination, ner_backend="spacy", semantic=True)
    return ResumeIndex(destination)


def selection_metrics(ids, grades, *, k=10):
    selected = set(ids[:k])
    if not selected <= grades.keys():
        raise ValueError("Every retrieved candidate needs a frozen judgment for this query.")
    relevant = {cid for cid, grade in grades.items() if grade >= 2}
    strongest = {cid for cid, grade in grades.items() if grade == 3}
    ideal_gain = sum(sorted((2**grade - 1 for grade in grades.values()), reverse=True)[:k])
    return {
        "relevant_count": len(relevant),
        "relevant_selected": len(selected & relevant),
        "recall_at_10": len(selected & relevant) / len(relevant) if relevant else None,
        "attainable_recall_at_10": len(selected & relevant) / min(k, len(relevant))
        if relevant
        else None,
        "strongest_recall_at_10": len(selected & strongest) / len(strongest) if strongest else None,
        "set_gain_at_10": sum(2 ** grades[cid] - 1 for cid in selected) / ideal_gain
        if ideal_gain
        else None,
        "positive_selected_ids": sorted(selected & relevant),
    }


def evaluate(index, queries, grades, *, expand=None, rank=None):
    records = []
    # Warm model loading separately from request timing. Index construction is also separate.
    if queries:
        req = Requirements.model_validate(queries[0]["requirements"])
        index.retrieve(req)
        from screening_agent.retrieval import _reranker

        _reranker()
    for q in queries:
        req = Requirements.model_validate(q["requirements"])
        start = time.perf_counter()
        batch = index.retrieve(req)
        batch = expand(index, batch, req) if expand else index.expand_retrieval(batch, req)
        result = rank(index, batch, req) if rank else index.rank(batch, req)
        seconds = time.perf_counter() - start
        ids = [m.candidate_id for m in result.matches]
        rows = [m.model_dump(mode="json") for m in result.matches]
        assert len(ids) == min(10, len(index.candidates)) and result.retrieved_count == len(
            index.candidates
        )
        assert not any(m.assessments or m.eligible for m in result.matches)
        for row in rows:
            for lead in row["retrieval_leads"]:
                for ev in lead["evidence"]:
                    assert (
                        index.documents[row["candidate_id"]]["text"][ev["start"] : ev["end"]]
                        == ev["quote"]
                    )
        record = {
            "query_id": q["job_id"],
            "query": q["query"],
            "requirements": req.model_dump(mode="json"),
            "top10": ids,
            "local_seconds": round(seconds, 4),
            "candidate_assessment_calls": 0,
            "metrics": selection_metrics(ids, grades[q["job_id"]]),
            "search_result": result.model_dump(mode="json"),
            "execution_mode": "frozen",
            "interpretation_tested": False,
            "ranking_input": result.retrieval_audit.get("context_query", ""),
        }
        records.append(record)
        print(q["job_id"], record["metrics"], round(seconds, 2), flush=True)
    return records


def interpretation_errors(actual, expected):
    """Check the independently supplied structural labels, not model wording.

    These checks do not claim to judge every source-clause meaning. Actual
    meanings and standards remain in the receipt for separate semantic review.
    """
    issues = []

    def groups(req):
        return {tuple(sorted(group)) for group in req.must_have}

    for name, got, wanted in (
        ("mandatory alternatives", groups(actual), groups(expected)),
        ("preferred skills", set(actual.nice_to_have), set(expected.nice_to_have)),
        ("overall years", actual.minimum_years, expected.minimum_years),
        ("skill years", actual.skill_years, expected.skill_years),
    ):
        if got != wanted:
            issues.append(f"{name}: expected {wanted}, received {got}")
    if expected.role and actual.role != expected.role:
        issues.append(f"role: expected {expected.role}, received {actual.role}")
    return issues


def evaluate_conversations(index, queries, grades, *, session_factory=None):
    """One fresh real graph per query; no frozen requirements enter the graph."""
    if session_factory is None:
        sys.path.insert(0, str(ROOT))
        from matching_agent import MatchingSession
        from screening_agent.planner import OpenAIPlanner

        def session_factory(directory):
            session = MatchingSession(
                ROOT,
                index=index,
                planner=OpenAIPlanner.from_env(ROOT),
                db_path=directory / "sessions.sqlite",
            )
            compiler = session.registry.requirement_compiler
            compiler.cache_dir = directory / "standards"
            compiler.cache_dir.mkdir()
            return session

    records = []
    for q in queries:
        with tempfile.TemporaryDirectory(prefix="screening-conversation-") as temp:
            session = session_factory(Path(temp))
            started = time.perf_counter()
            try:
                state = session.send(q["query"])
            finally:
                session.close()
            elapsed = time.perf_counter() - started
        expected = Requirements.model_validate(q["requirements"])
        actual = Requirements.model_validate(state.get("requirements") or {})
        issues = interpretation_errors(actual, expected)
        if state.get("error") or state.get("clarification"):
            issues.append(str(state.get("error") or state["clarification"]))
        ids = [m["candidate_id"] for m in state.get("shortlist", [])]
        if len(ids) != min(10, len(index.candidates)):
            issues.append("The initial graph did not produce the complete review pool.")
        if state.get("detailed_reviews") or any(
            m.get("assessments") for m in state.get("shortlist", [])
        ):
            raise AssertionError("Initial screening performed candidate assessment.")
        audit = state.get("retrieval_audit", {})
        # Some state versions store the audit inside search_result.
        if not audit:
            audit = (state.get("search_result") or {}).get("retrieval_audit", {})
        row = {
            "query_id": q["job_id"],
            "query": q["query"],
            "execution_mode": "conversation",
            "interpretation_tested": True,
            "requirements": actual.model_dump(mode="json"),
            "expected_requirements": expected.model_dump(mode="json"),
            "interpretation_errors": issues,
            "top10": ids,
            "end_to_end_seconds": round(elapsed, 4),
            "ranking_input": audit.get("context_query", ""),
            "metrics": selection_metrics(ids, grades[q["job_id"]]) if not issues else None,
            "state": state,
        }
        records.append(row)
        print(
            q["job_id"], {"interpretation_errors": issues, "seconds": round(elapsed, 2)}, flush=True
        )
    return records


def summarize(records):
    positive = [r for r in records if r["metrics"]["relevant_count"]]
    strongest = [
        r["metrics"]["strongest_recall_at_10"]
        for r in records
        if r["metrics"]["strongest_recall_at_10"] is not None
    ]
    return {
        "macro_strongest_recall_at_10": float(np.mean(strongest)) if strongest else None,
        "query_count": len(records),
        "positive_query_count": len(positive),
        "macro_attainable_recall_at_10": float(
            np.mean([r["metrics"]["attainable_recall_at_10"] for r in positive])
        )
        if positive
        else None,
        "relevant_selected": sum(r["metrics"]["relevant_selected"] for r in records),
        "relevant_available": sum(r["metrics"]["relevant_count"] for r in records),
        "maximum_selectable_positives": sum(
            min(10, r["metrics"]["relevant_count"]) for r in records
        ),
        "mean_set_gain_at_10": float(
            np.mean(
                [
                    r["metrics"]["set_gain_at_10"]
                    for r in records
                    if r["metrics"]["set_gain_at_10"] is not None
                ]
            )
        ),
        "median_local_seconds": float(np.median([r["local_seconds"] for r in records])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", choices=["holdout", "regression"], default="holdout")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=["frozen", "conversation"], default="frozen")
    parser.add_argument(
        "--query-id",
        action="append",
        default=[],
        help="Repeat to run a named subset; the receipt retains each selected original request.",
    )
    args = parser.parse_args()
    queries, grades, files, provenance = collection(args.collection)
    if args.query_id:
        if set(args.query_id) - {q["job_id"] for q in queries}:
            raise ValueError("A selected query ID is absent from this collection.")
        queries = [q for q in queries if q["job_id"] in args.query_id]
    with tempfile.TemporaryDirectory(prefix="screening-selection-") as temp:
        start = time.perf_counter()
        index = prepare_index(Path(temp), files)
        setup_seconds = time.perf_counter() - start
        rows = (evaluate if args.mode == "frozen" else evaluate_conversations)(
            index, queries, grades
        )
        engine = index.engine_fingerprint
    output = args.output or ROOT / f"reports/ranking_validation/{args.collection}-{args.mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "provenance": provenance,
                "engine_fingerprint": engine,
                "ranker_sha256": sha(ROOT / "src/screening_agent/candidate_retrieval.py"),
                "benchmark_sha256": sha(Path(__file__)),
                "machine": {
                    "system": platform.system(),
                    "architecture": platform.machine(),
                    "python": platform.python_version(),
                },
                "scope": "Frozen requirements; interpretation is not tested."
                if args.mode == "frozen"
                else "Real interpreter, shared-standard compiler and initial graph. Expected requirements and labels are never injected into the application. Structural interpretation checks do not establish complete semantic correctness.",
                "timing_scope": "Local warmed stages; excludes interpretation and index construction."
                if args.mode == "frozen"
                else "Complete initial graph, including interpretation and shared standards; excludes index construction and browser rendering.",
                "execution_mode": args.mode,
                "query_count": len(rows),
                "distinct_ranking_inputs": len(
                    {r["ranking_input"] for r in rows if r["ranking_input"]}
                ),
                "index_build_seconds": setup_seconds,
                "summary": summarize(rows)
                if args.mode == "frozen"
                else {
                    "structural_interpretation_passed": sum(
                        not r["interpretation_errors"] for r in rows
                    ),
                    "failed_query_ids": [r["query_id"] for r in rows if r["interpretation_errors"]],
                    "median_end_to_end_seconds": float(
                        np.median([r["end_to_end_seconds"] for r in rows])
                    ),
                },
                "queries": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
