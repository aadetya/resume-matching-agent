"""Separate evidence judgments, temporal accuracy and schema labels in frozen runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def summarize(root: Path, run_path: Path) -> dict:
    run = json.loads(run_path.read_text())
    fixture_bytes = (
        root / run.get("fixture_path", "data/contextual_diagnostics.json")
    ).read_bytes()
    fixture = json.loads(fixture_bytes)
    assert hashlib.sha256(fixture_bytes).hexdigest() == run["fixture_sha256"]
    cases = {c["id"]: c for c in fixture["cases"]}
    observed = {c["id"]: c for c in run["cases"]}
    output = {
        k: run[k]
        for k in (
            "fixture_sha256",
            "model",
            "engine_signature",
            "generation_options",
            "annotation_provenance",
        )
    }
    output.update(
        requested_cases=run["summary"]["requested_cases"],
        completed_cases=len(observed),
        errors=run["errors"],
        stages={},
    )
    details = run_path.parent / (run_path.stem + "-details")
    records = {cid: json.loads((details / (cid + ".json")).read_text()) for cid in observed}
    checked, mismatches = set(), []
    stages = [
        stage
        for stage in ("draft", "initial", "deep")
        if any(c.get(stage) for c in observed.values())
    ] + [
        stage
        for stage in ("independent", "audit_initial", "audit_independent", "audit_initial_repaired")
        if any(c.get(stage) for c in observed.values())
    ]
    for stage in stages:
        rows = [(cid, r) for cid, c in observed.items() for r in c.get(stage) or []]
        durations = [
            (cid, r)
            for cid, r in rows
            if cases[cid]["expected"][r["criterion"]]["duration_basis"] is not None
        ]
        label_only = [
            (cid, r) for cid, r in rows if r["failures"] == ["Unexpected semantic relationship"]
        ]
        temporal_errors = [
            (cid, r)
            for cid, r in durations
            if any(
                x in r["failures"]
                for x in (
                    "Unknown duration was assigned a value",
                    "Duration outside source-supported bounds",
                )
            )
        ]
        pairs = []

        def signature(cid, include_labels, stage=stage):
            return [
                (
                    r["criterion"],
                    r["actual_status"],
                    (r["duration"] or {}).get("months"),
                    (r["duration"] or {}).get("maximum_months"),
                    *(
                        [(r["duration"] or {}).get("basis"), r["actual_relationship"]]
                        if include_labels
                        else []
                    ),
                )
                for r in observed[cid].get(stage) or []
            ]

        for cid, c in cases.items():
            other = c.get("counterpart_id")
            if c.get("pair_relation") != "invariant" or not other or cid >= other:
                continue
            complete = cid in observed and other in observed
            pairs.append(
                {
                    "case_ids": [cid, other],
                    "completed": complete,
                    "status_and_duration_equal": complete
                    and signature(cid, False) == signature(other, False),
                    "including_category_labels_equal": complete
                    and signature(cid, True) == signature(other, True),
                }
            )
        output["stages"][stage] = dict(
            criteria=len(rows),
            strict_passes=sum(r["passed"] for _, r in rows),
            single_status_expectations=sum(
                len(cases[cid]["expected"][r["criterion"]]["allowed_statuses"]) == 1
                for cid, r in rows
            ),
            single_status_agreements=sum(
                len(cases[cid]["expected"][r["criterion"]]["allowed_statuses"]) == 1
                and "Unexpected evidence status" not in r["failures"]
                for cid, r in rows
            ),
            required_association_contexts=sum(
                len(r.get("association_checks", [])) for _, r in rows
            ),
            found_association_contexts=sum(
                c["found_source_context"] for _, r in rows for c in r.get("association_checks", [])
            ),
            excluded_context_credit_flags=[
                {"id": cid, "criterion": r["criterion"], "source_fragments": r["excluded_credit"]}
                for cid, r in rows
                if r.get("excluded_credit")
            ],
            status_agreements=sum(
                r["actual_status"] is not None and "Unexpected evidence status" not in r["failures"]
                for _, r in rows
            ),
            duration_expectations=len(durations),
            duration_value_agreements=len(durations) - len(temporal_errors),
            relationship_label_only_disagreements=[{"id": cid, **r} for cid, r in label_only],
            other_disagreements=[
                {"id": cid, **r}
                for cid, r in rows
                if not r["passed"] and r["failures"] != ["Unexpected semantic relationship"]
            ],
            invariant_pairs=pairs,
            incremental_usage={
                token: sum(
                    (records[cid].get(stage) or {}).get("usage", {}).get(token, 0)
                    for cid in observed
                )
                for token in ("input_tokens", "output_tokens", "total_tokens")
            },
            usage_scope="Incremental stage usage in this run. Cache reuse means these are not comparable from-scratch path costs.",
        )

        def check(cid, value):
            if isinstance(value, dict):
                if {"quote", "start", "end", "document_sha256"} <= value.keys():
                    key = (
                        cid,
                        value["start"],
                        value["end"],
                        value["quote"],
                        value["document_sha256"],
                    )
                    checked.add(key)
                    text = cases[cid]["resume_text"]
                    if (
                        text[value["start"] : value["end"]] != value["quote"]
                        or hashlib.sha256(text.encode()).hexdigest() != value["document_sha256"]
                    ):
                        mismatches.append({"id": cid, "start": value["start"], "end": value["end"]})
                for child in value.values():
                    check(cid, child)
            elif isinstance(value, list):
                for child in value:
                    check(cid, child)

        for cid in observed:
            check(cid, records[cid].get(stage))
    output["source_provenance"] = {
        "unique_original_spans_checked": len(checked),
        "mismatches": mismatches,
        "scope": "Exact source positions and hashes; this does not measure semantic entailment.",
    }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = summarize(root, args.run)
    output = args.run.with_name(args.run.stem + "-summary.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
