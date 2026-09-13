"""Run frozen author-reviewed semantic diagnostics without modifying their labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from openai import OpenAI

from screening_agent.contextual import ContextualReviewer
from screening_agent.contracts import Candidate, Match, Requirements
from screening_agent.criteria import RequirementCompiler
from screening_agent.model_config import DEFAULT_MODEL
from screening_agent.policy import fingerprint, requirements_fingerprint


def evaluate(expected, review):
    actual = {r["criterion"]: r for r in review["criteria"]}
    results = []
    for label, judgment in expected.items():
        row = actual.get(label)
        failures = []
        if row is None:
            failures.append("Criterion missing")
        else:
            if row["status"] not in judgment["allowed_statuses"]:
                failures.append("Unexpected evidence status")
            if (
                judgment["allowed_relationships"] is not None
                and row["relationship"] not in judgment["allowed_relationships"]
            ):
                failures.append("Unexpected semantic relationship")
            if judgment["duration_basis"] is not None:
                duration = row.get("duration") or {}
                if duration.get("basis") != judgment["duration_basis"]:
                    failures.append("Unexpected duration basis")
                lower, upper = (
                    judgment["duration_minimum_months"],
                    judgment["duration_maximum_months"],
                )
                if lower is None and upper is None:
                    if duration.get("months") is not None:
                        failures.append("Unknown duration was assigned a value")
                elif duration.get("months") is None or not lower <= duration["months"] <= upper:
                    failures.append("Duration outside source-supported bounds")
        association_checks = []
        excluded_credit = []
        if row and judgment.get("positive_associations"):
            units = {u["unit_id"]: u for u in review["evidence_graph"]["work_units"]}
            credited = set((row.get("duration") or {}).get("unit_ids", []))
            links = []
            for association in row.get("duration_associations", []):
                unit = units[association["unit_id"]]
                text = " ".join(
                    " ".join(e["quote"] for e in unit["evidence"] + association["evidence"]).split()
                )
                links.append((association, text, association["unit_id"] in credited))
            for expected_link in judgment["positive_associations"]:
                found = any(
                    is_credited
                    and all(
                        " ".join(fragment.split()) in text
                        for fragment in expected_link["source_fragments"]
                    )
                    for _, text, is_credited in links
                )
                association_checks.append(
                    {"title": expected_link["title"], "found_source_context": found}
                )
            if judgment.get("must_find_all_positive_associations") and not all(
                c["found_source_context"] for c in association_checks
            ):
                failures.append("Required positive association context missing")
        if row and judgment.get("excluded_duration_source_fragments"):
            units = {u["unit_id"]: u for u in review["evidence_graph"]["work_units"]}
            for uid in (row.get("duration") or {}).get("unit_ids", []):
                text = " ".join(" ".join(e["quote"] for e in units[uid]["evidence"]).split())
                excluded_credit += [
                    f
                    for f in judgment["excluded_duration_source_fragments"]
                    if " ".join(f.split()) in text
                ]
            if excluded_credit:
                failures.append("Excluded duration context received credit")
        results.append(
            {
                "criterion": label,
                "association_checks": association_checks,
                "excluded_credit": excluded_credit,
                "passed": not failures,
                "failures": failures,
                "actual_status": row["status"] if row else None,
                "actual_relationship": row["relationship"] if row else None,
                "duration": row.get("duration") if row else None,
                "finding": row["finding"] if row else None,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--fixture", default="data/contextual_diagnostics.json")
    parser.add_argument(
        "--run-id",
        default="default",
        help="A new ID forces fresh assessment calls; standards remain shared.",
    )
    parser.add_argument("--output", default="reports/semantic_evaluation/diagnostics.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / args.fixture
    original = path.read_bytes()
    dataset = json.loads(original)
    digest = hashlib.sha256(original).hexdigest()
    cases = [c for c in dataset["cases"] if not args.ids or c["id"] in args.ids]
    work = root / "artifacts/contextual_diagnostics" / args.run_id
    sources = work / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    documents, candidates = {}, {}
    for case in cases:
        cid, content = case["id"], case["resume_text"]
        source = sources / f"{cid}.txt"
        source.write_text(content)
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        documents[cid] = {
            "text": content,
            "sha256": sha,
            "source_path": str(source.relative_to(work)),
        }
        candidates[cid] = Candidate(
            candidate_id=cid,
            name=content.splitlines()[0],
            skills=[],
            source_path=documents[cid]["source_path"],
        )

    def check_sources(ids=None):
        for cid in ids or candidates:
            record = documents[cid]
            if (
                hashlib.sha256((work / record["source_path"]).read_bytes()).hexdigest()
                != record["sha256"]
            ):
                raise ValueError("Diagnostic source changed during assessment.")

    index = SimpleNamespace(
        candidates=candidates,
        documents=documents,
        manifest={"snapshot_date": dataset["snapshot_date"]},
        engine_fingerprint=fingerprint({"fixture": digest}),
        _check_sources=check_sources,
    )
    client = OpenAI()
    reviewer = ContextualReviewer(root=work, client=client, model=args.model)
    compiler = RequirementCompiler(root, client=client, model=args.model)
    compiled_requests = {}
    # Freeze all shared standards before assessing any candidate; never tune them to labels.
    for case in cases:
        raw = Requirements.model_validate(case["requirements"])
        key = fingerprint(raw.model_dump(mode="json"))
        if key not in compiled_requests:
            compiled_requests[key] = compiler.compile_with_metadata(raw)
    print("Shared standards frozen:", len(compiled_requests), flush=True)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    run = {
        "fixture_sha256": digest,
        "fixture_path": str(path.relative_to(root)),
        "model": args.model,
        "engine_signature": reviewer.signature,
        "generation_options": reviewer.options,
        "annotation_provenance": dataset["annotation_provenance"],
        "run_id": args.run_id,
        "compiled_requests": {
            k: {"requirements": req.model_dump(mode="json"), **meta}
            for k, (req, meta) in compiled_requests.items()
        },
        "cases": [],
        "errors": [],
    }

    def one(case):
        cid = case["id"]
        req, _ = compiled_requests[
            fingerprint(Requirements.model_validate(case["requirements"]).model_dump(mode="json"))
        ]
        started = time.perf_counter()
        match = Match(
            candidate_id=cid,
            name=candidates[cid].name,
            score=0,
            eligible=False,
            requirements_fingerprint=requirements_fingerprint(req),
            engine_fingerprint=index.engine_fingerprint,
        )
        final = reviewer.review(index, [match], req)[0]
        stage_errors = []
        if final.get("stage") == "pending":
            stage_errors.append(
                {
                    "id": cid,
                    "stage": "deep",
                    "error": final["error"],
                    "failure_record": final["failure_record"],
                }
            )
            draft = None
        else:
            from screening_agent.contextual import ContextualResponse, compile_response

            draft = compile_response(
                ContextualResponse.model_validate(final["draft"]), final["packet"]
            )
        record = {
            "id": cid,
            "family": case["family"],
            "stage_errors": stage_errors,
            "elapsed_seconds": time.perf_counter() - started,
            "counterpart_id": case["counterpart_id"],
            "draft": evaluate(case["expected"], draft) if draft else None,
            "deep": evaluate(case["expected"], final) if final else None,
            "deep_usage": final["usage"] if final else {},
        }
        detail = output.parent / (output.stem + "-details")
        detail.mkdir(exist_ok=True)
        (detail / f"{cid}.json").write_text(json.dumps({"draft": draft, "deep": final}, indent=2))
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, c): c for c in cases}
        for future in as_completed(futures):
            case = futures[future]
            try:
                record = future.result()
                run["cases"].append(record)
                run["errors"].extend(record["stage_errors"])
                scored = record["deep"] or []
                print(case["id"], sum(r["passed"] for r in scored), "/", len(scored), flush=True)
            except Exception as exc:
                run["errors"].append({"id": case["id"], "error": str(exc)})
                print(case["id"], "ERROR", str(exc)[:400], flush=True)
            output.write_text(json.dumps(run, indent=2))
    assert path.read_bytes() == original, "Frozen expectations changed during evaluation"
    run["cases"].sort(key=lambda c: c["id"])
    run["summary"] = {
        "requested_cases": len(cases),
        "completed_cases": len(run["cases"]),
        "errors": len(run["errors"]),
    }
    for stage in (
        "draft",
        "deep",
    ):
        rows = [r for c in run["cases"] for r in c.get(stage) or []]
        run["summary"][stage] = {
            "criteria": len(rows),
            "passed": sum(r["passed"] for r in rows),
            "failures": [
                {"id": c["id"], **r}
                for c in run["cases"]
                for r in c.get(stage) or []
                if not r["passed"]
            ],
        }
    output.write_text(json.dumps(run, indent=2))
    print(
        json.dumps(
            {
                k: (
                    {field: value for field, value in v.items() if field != "failures"}
                    if isinstance(v, dict)
                    else v
                )
                for k, v in run["summary"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
