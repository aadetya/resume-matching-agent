"""Regressions for omitted jobs, query-independent facts, disputes and partial failures."""

import json
from copy import deepcopy

import pytest
import test_contextual
from test_contextual import bind_fixture_ids, compile_fixture, fake_client

from screening_agent.contextual import (
    ContextualResponse,
    ContextualReviewer,
    compile_response,
    source_packet,
)
from screening_agent.contracts import Match, Requirements, SearchResult
from screening_agent.inventory import prepare_inventory
from screening_agent.model_config import DEFAULT_MODEL
from screening_agent.policy import fingerprint
from screening_agent.ui_views import conversation_reply, detailed_review_markdown, stage_html

case = test_contextual.case


def complete_case(case):
    _, packet, raw = case
    bind_fixture_ids(raw, packet["semantic_criteria"])
    packet["source_inventory"] = {
        "work_units": deepcopy(raw["work_units"]),
        "complete": True,
        "version": "fixture",
    }
    return packet, raw


def test_a_recognized_job_cannot_silently_vanish_from_duration_decisions(case):
    packet, raw = complete_case(case)
    raw["findings"][0]["unit_decisions"].pop()
    with pytest.raises(ValueError, match="missing=\\['second'\\]"):
        compile_response(ContextualResponse.model_validate(raw), packet)


def test_narrative_acknowledgement_cannot_replace_an_included_job_association(case):
    packet, raw = complete_case(case)
    raw["findings"][0]["duration_associations"].pop()
    with pytest.raises(ValueError, match="Included duration decisions"):
        compile_response(ContextualResponse.model_validate(raw), packet)


def test_uncertain_employment_never_produces_a_closed_total(case):
    packet, raw = complete_case(case)
    row = raw["findings"][0]
    row["duration_associations"].pop()
    row["unit_decisions"][1].update(
        decision="uncertain", reason="The source leaves this job's Python responsibility ambiguous."
    )
    compiled = compile_response(ContextualResponse.model_validate(raw), packet)["criteria"][0]
    assert compiled["duration"]["months"] == 36
    assert compiled["duration"]["maximum_months"] is None
    assert not compiled["duration"]["coverage_complete"]
    assert "Other employment decisions" in compiled["finding"]
    assert "ambiguous" in compiled["finding"]


def test_incomplete_source_inventory_cannot_qualify_via_precise_dates(case):
    packet, raw = complete_case(case)
    packet["source_inventory"]["complete"] = False
    row = compile_response(ContextualResponse.model_validate(raw), packet)["criteria"][0]
    assert row["status"] == "uncertain"
    assert row["duration"]["minimum_months"] is None
    assert row["duration"]["maximum_months"] is None
    assert row["duration"]["measurable_dates_complete"]
    assert not row["duration"]["extraction_complete"]


def test_assessment_cannot_rewrite_frozen_job_dates(case):
    packet, raw = complete_case(case)
    raw["work_units"][0]["end_date"] = "2021-01"
    with pytest.raises(ValueError, match="frozen source inventory"):
        compile_response(ContextualResponse.model_validate(raw), packet)


def test_inventory_is_shared_across_unrelated_queries_and_excludes_query_text(case, tmp_path):
    index, packet, raw = case
    calls = []
    reviewer = ContextualReviewer(
        root=tmp_path, client=fake_client(raw, calls), model=DEFAULT_MODEL
    )
    first = prepare_inventory(reviewer, packet)
    other = source_packet(index, "A", Requirements(role="designer", must_have=[["novel material"]]))
    second = prepare_inventory(reviewer, other)
    assert first["version"] == second["version"]
    assert first["work_units"] == second["work_units"]
    assert len(calls) == 2
    assert second["generation"]["cache_hit"] and second["audit_generation"]["cache_hit"]
    for call in calls:
        payload = json.loads(call["input"])
        assert payload["criteria"] == [] and payload["semantic_brief"] == ""
        assert "designer" not in call["input"]
    expected = {p["id"] for p in packet["passages"]}
    assert {line for u in first["work_units"] for line in u["source_lines"]} == expected


def test_auditor_cannot_silently_replace_a_supported_judgment(case, tmp_path):
    index, _, raw = case
    packet = source_packet(index, "A", Requirements(must_have=[["python"]]))
    packet["semantic_criteria"][0]["standard"] = {"capability": "Personal Python development work."}
    row = raw["findings"][0]
    row.update(
        criterion_id=packet["semantic_criteria"][0]["criterion_id"],
        duration_associations=[],
        duration_basis="none",
        evidence_depth="personal_work",
    )
    compiled = compile_fixture(raw, packet)

    def disagree(data, payload):
        data["criteria"][0].update(
            verdict=dict(
                assessed_status="not_demonstrated", relationship="none", evidence_depth="not_found"
            ),
            reason="The auditor disagrees with the documented developer responsibility.",
        )

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(raw, [], grounding_transform=disagree),
        model=DEFAULT_MODEL,
    )
    result = reviewer.audit_grounding(packet, {**compiled, "raw": raw})
    assert result["assessment"]["criteria"][0]["status"] == "uncertain"
    assert result["disputes"][0]["initial"]["status"] == "supported"
    assert result["disputes"][0]["audit"]["verdict"]["assessed_status"] == "not_demonstrated"
    assert result["disputes"][0]["resolution"] == "needs_review"


def test_rejected_duration_link_remains_an_explicit_uncertain_unit(case, tmp_path):
    packet, raw = complete_case(case)

    def reject_one(data, payload):
        data["associations"][1].update(
            entailed=False,
            reason="The evidence does not establish substantive Python work for this whole job.",
        )

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(raw, [], grounding_transform=reject_one),
        model=DEFAULT_MODEL,
    )
    compiled = compile_response(ContextualResponse.model_validate(raw), packet)
    result = reviewer.audit_grounding(packet, {**compiled, "raw": raw})
    row = result["assessment"]["criteria"][0]
    assert row["duration"]["months"] == 36 and row["duration"]["maximum_months"] is None
    assert row["unit_decisions"][1]["decision"] == "uncertain"
    assert len(result["rejected_associations"]) == 1
    assert "repair_duration_inventory" not in reviewer.review_graph.get_graph().nodes


def test_one_failure_retains_completed_results_and_the_failed_payload(case, tmp_path):
    index, packet, raw = case
    reviewer = ContextualReviewer(root=tmp_path, client=fake_client(raw, []), model=DEFAULT_MODEL)
    req = Requirements(skill_years={"python": 3})
    index.candidates["B"] = index.candidates["A"].model_copy(
        update={"candidate_id": "B", "source_path": "B.txt"}
    )
    index.documents["B"] = {**index.documents["A"], "source_path": "B.txt"}
    matches = [Match(candidate_id=cid, name=cid, score=50, eligible=False) for cid in ("A", "B")]
    index.rank = lambda *a, **kw: SearchResult(
        matches=deepcopy(matches),
        corpus_size=2,
        retrieved_count=2,
        not_shortlisted_count=0,
        backend="test",
    )

    original = reviewer.review_graph.invoke

    def invoke(value):
        if value["packet"]["candidate_id"] == "A":
            return original(value)
        error = ValueError("A finding's citation is outside its linked work context.")
        error.usage = {"total_tokens": 123}
        error.repair_trace = [
            {
                "attempt": 1,
                "response": {"findings": [{"criterion_id": "broken", "source_lines": ["L99"]}]},
                "validation_error": str(error),
            }
        ]
        raise error

    reviewer.review_graph.invoke = invoke
    reviews = reviewer.review(index, matches, req)
    assert [r["candidate_id"] for r in reviews] == ["A", "B"]
    assert reviews[0]["stage"] == "deep"
    pending = reviews[1]
    assert pending["stage"] == "pending"
    receipt = json.loads((tmp_path / pending["failure_record"]).read_text())
    assert receipt["usage"]["total_tokens"] == 123
    assert receipt["attempts"][0]["response"]["findings"][0]["source_lines"] == ["L99"]
    assert (
        fingerprint({k: v for k, v in receipt.items() if k != "integrity"}) == receipt["integrity"]
    )
    state = {
        "requirements": req.model_dump(),
        "plan": {"action": "deep_screen"},
        "shortlist": [m.model_dump() for m in matches],
        "detailed_reviews": reviews,
    }
    assert "pending" in conversation_reply(state, 2).lower()
    assert "pending" in stage_html(state)
    assert "pending" in detailed_review_markdown(pending)
    failed_match = Match(candidate_id="B", name="B", eligible=False, score=49).model_dump()
    assert (
        reviewer.apply_recommendations([pending], [failed_match], req)[0]["recommendation"]
        == "hold"
    )


def test_nonemployment_annotation_issue_does_not_erase_verified_employment(case):
    packet, raw = complete_case(case)
    packet["source_inventory"].update(complete=False, employment_complete=True)
    row = compile_response(ContextualResponse.model_validate(raw), packet)["criteria"][0]
    assert row["status"] == "supported"
    assert row["duration"]["months"] == row["duration"]["maximum_months"] == 48


def test_inventory_generation_schema_rejects_copied_line_text_as_an_id(case, tmp_path):
    from pydantic import ValidationError

    reviewer = ContextualReviewer(
        root=tmp_path, client=fake_client(case[2], calls := []), model=DEFAULT_MODEL
    )
    prepare_inventory(reviewer, case[1])
    schema = calls[0]["text_format"]
    bad = {"work_units": deepcopy(case[2]["work_units"])}
    bad["work_units"][0]["source_lines"][0] = "L2: Python Developer | 2020-01 to 2022-12"
    with pytest.raises(ValidationError, match="enum"):
        schema.model_validate(bad)


def test_assessment_schema_binds_criterion_identity_and_count(case, tmp_path):
    from pydantic import ValidationError

    from screening_agent.contextual import AssessmentResponse

    reviewer = ContextualReviewer(
        root=tmp_path, client=fake_client(case[2], calls := []), model=DEFAULT_MODEL
    )
    result = reviewer.assess(case[1])
    schema = next(
        c["text_format"] for c in calls if issubclass(c["text_format"], AssessmentResponse)
    )
    response = {k: deepcopy(result["raw"][k]) for k in ("findings", "observations")}
    response["findings"][0]["criterion_id"] += "d"
    with pytest.raises(ValidationError, match="enum"):
        schema.model_validate(response)
    response["findings"] = []
    with pytest.raises(ValidationError, match="too_short"):
        schema.model_validate(response)


@pytest.mark.parametrize("clause", ["", "The candidate must have a senior title."])
def test_changed_support_requires_a_real_frozen_standard_clause(case, tmp_path, clause):
    index, _, raw = case
    packet = source_packet(index, "A", Requirements(must_have=[["python"]]))
    packet["semantic_criteria"][0]["standard"] = {"capability": "Personal Python development work."}
    raw["findings"][0].update(
        criterion_id=packet["semantic_criteria"][0]["criterion_id"],
        duration_associations=[],
        duration_basis="none",
        evidence_depth="personal_work",
    )
    compiled = compile_fixture(raw, packet)

    def disagree(data, payload):
        data["criteria"][0].update(
            verdict=dict(
                assessed_status="not_demonstrated", relationship="none", evidence_depth="not_found"
            ),
            standard_clause=clause,
        )

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(raw, [], grounding_transform=disagree),
        model=DEFAULT_MODEL,
    )
    with pytest.raises(ValueError, match="exact frozen-standard clause"):
        reviewer.audit_grounding(packet, {**compiled, "raw": raw})
