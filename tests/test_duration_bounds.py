"""Numerical eligibility depends on interval bounds, not an experience label."""

from hashlib import sha256
from types import SimpleNamespace

import pytest

from screening_agent.contextual import (
    ContextualResponse,
    ContextualReviewer,
    compile_response,
    source_packet,
)
from screening_agent.contracts import Candidate, Requirements


@pytest.mark.parametrize(
    "basis",
    [
        "total_employment",
        "skill_related_employment",
        "explicit_usage",
        "explicit_usage_union_bounds",
    ],
)
@pytest.mark.parametrize("endpoint", ["2020-12", None])
@pytest.mark.parametrize("proposed_negative", [False, True])
def test_closed_insufficient_duration_and_open_uncertainty(basis, endpoint, proposed_negative):
    text = "Anonymous\nEngineer | 2020-01" + (" to " + endpoint if endpoint else "")
    text += "\nUsed Python throughout the employment period.\n"
    index = SimpleNamespace(
        documents={
            "A": {"text": text, "sha256": sha256(text.encode()).hexdigest(), "source_path": "A.txt"}
        },
        candidates={
            "A": Candidate(candidate_id="A", name="Anonymous", skills=[], source_path="A.txt")
        },
        manifest={"snapshot_date": "2026-09-01"},
        engine_fingerprint="test",
        _check_sources=lambda *_: None,
    )
    req = (
        Requirements(minimum_years=2)
        if basis == "total_employment"
        else Requirements(skill_years={"python": 2})
    )
    packet = source_packet(index, "A", req)
    raw = ContextualResponse.model_validate(
        dict(
            work_units=[
                dict(
                    unit_id="employment",
                    scope="employment",
                    title="Engineer",
                    source_lines=["L2", "L3"],
                    start_date="2020-01",
                    end_date=endpoint,
                    end_kind="calendar" if endpoint else "unknown",
                )
            ],
            findings=[
                dict(
                    criterion_id=packet["semantic_criteria"][0]["criterion_id"],
                    status="not_demonstrated" if proposed_negative else "supported",
                    evidence_depth="contradictory" if proposed_negative else "dated_association",
                    relationship="direct",
                    finding="Personal Python usage throughout this employment period.",
                    unit_ids=["employment"],
                    source_lines=["L2", "L3"],
                    usage_claims=[],
                    unit_decisions=[
                        dict(
                            unit_id="employment",
                            decision="include",
                            reason="The fixture credits this documented employment.",
                            source_lines=["L2", "L3"],
                        )
                    ],
                    duration_basis=basis,
                    duration_associations=[
                        dict(
                            unit_id="employment",
                            basis="explicit_usage" if basis.startswith("explicit_usage") else basis,
                            assertion="The source describes the candidate's employment and Python usage throughout that period.",
                            source_lines=["L2", "L3"],
                            temporal_scope="whole_role",
                        )
                    ],
                )
            ],
            observations=[],
        )
    )
    packet["source_inventory"] = {
        "work_units": raw.model_dump(mode="json")["work_units"],
        "complete": True,
    }
    row = compile_response(raw, packet)["criteria"][0]
    assert row["duration"]["maximum_months"] == (12 if endpoint else None)
    assert row["status"] == ("not_demonstrated" if endpoint else "uncertain")

    # A numeric failure needs verified coverage before driving a recommendation.
    # Its source can remain a positive employment association, not a contrary claim.
    row["evidence_depth"] = "dated_association"
    reviewer = ContextualReviewer.__new__(ContextualReviewer)
    reviews, matches = [{"candidate_id": "A", "criteria": [row]}], [{"candidate_id": "A"}]
    row["duration"]["extraction_complete"] = False
    assert reviewer.apply_evidence(reviews, matches)[0]["screening_status"] == "needs_review"
    row["duration"]["extraction_complete"] = True
    row["duration"]["coverage_complete"] = False
    assert reviewer.apply_evidence(reviews, matches)[0]["screening_status"] == "needs_review"
    row["duration"]["coverage_complete"] = bool(endpoint)
    assert reviewer.apply_evidence(reviews, matches)[0]["screening_status"] == (
        "does_not_meet" if endpoint else "needs_review"
    )
