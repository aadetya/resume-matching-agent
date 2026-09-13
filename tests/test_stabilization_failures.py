"""Independent cross-stage failure, source-inventory and audit-authority regressions."""

import json
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import pytest
import test_contextual
from test_contextual import compile_fixture, fake_client

from screening_agent.contextual import (
    MAX_SOURCE_CHARS,
    AssessmentResponse,
    ContextualReviewer,
    GroundingChecks,
    source_packet,
)
from screening_agent.contracts import Candidate, Match, Requirements, SearchResult
from screening_agent.inventory import InventoryCheck, SourceInventory, prepare_inventory
from screening_agent.model_config import DEFAULT_MODEL
from screening_agent.policy import fingerprint

case = test_contextual.case


def routed_client(raw, calls, override):
    """Use valid source fixtures except for the explicitly selected failing stage."""
    delegate = fake_client(raw, calls).responses.parse

    def parse(**kwargs):
        result = override(kwargs)
        if result is not None:
            calls.append(kwargs)
            if isinstance(result, Exception):
                raise result
            return result
        return delegate(**kwargs)

    return SimpleNamespace(responses=SimpleNamespace(parse=parse))


def provider_result(parsed, *, usage=True):
    return SimpleNamespace(
        output_parsed=parsed,
        usage=SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30)
        if usage
        else None,
        model=DEFAULT_MODEL,
    )


def receipt(root, error):
    saved = json.loads((root / error.failure_record).read_text())
    assert fingerprint({k: v for k, v in saved.items() if k != "integrity"}) == saved["integrity"]
    return saved


def test_inventory_audit_failure_retains_successful_extraction_cost(case, tmp_path):
    calls = []
    client = routed_client(
        case[2],
        calls,
        lambda request: (
            RuntimeError("private provider diagnostic")
            if issubclass(request["text_format"], InventoryCheck)
            else None
        ),
    )
    reviewer = ContextualReviewer(root=tmp_path, client=client, model=DEFAULT_MODEL)
    with pytest.raises(RuntimeError) as caught:
        reviewer.assess(case[1])
    error = caught.value
    assert error.usage == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    assert error.model_calls == len(calls) == 2
    assert error.usage_unknown_calls == 1
    saved = receipt(tmp_path, error)
    assert [step["stage"] for step in saved["completed_steps"]] == ["source_inventory"]
    assert saved["preceding_failure_record"]
    assert "private provider diagnostic" not in json.dumps(saved)


def test_invalid_initial_responses_keep_inventory_and_both_attempt_costs(case, tmp_path):
    calls = []

    def invalid(request):
        if not issubclass(request["text_format"], AssessmentResponse):
            return None
        parsed = fake_client(deepcopy(case[2]), []).responses.parse(**request).output_parsed
        parsed.findings[0].source_lines = ["L999"]
        return provider_result(parsed)

    reviewer = ContextualReviewer(
        root=tmp_path, client=routed_client(case[2], calls, invalid), model=DEFAULT_MODEL
    )
    with pytest.raises(ValueError, match="after one structural repair") as caught:
        reviewer.assess(case[1])
    error = caught.value
    assert error.usage["total_tokens"] == 120
    assert error.model_calls == len(calls) == 4 and error.usage_unknown_calls == 0
    assert len(error.repair_trace) == 2
    saved = receipt(tmp_path, error)
    assert {step["stage"] for step in saved["completed_steps"]} == {
        "generation",
        "audit_generation",
    }
    assert all(step["usage"]["total_tokens"] == 30 for step in saved["completed_steps"])
    assert all(
        attempt["response"]["findings"][0]["source_lines"] == ["L999"]
        for attempt in saved["attempts"]
    )


@pytest.mark.parametrize("warm_cache", [False, True])
def test_graph_final_audit_failure_accounts_for_completed_stages_and_cache(
    case, tmp_path, warm_cache
):
    calls = []
    reviewer = ContextualReviewer(
        root=tmp_path, client=fake_client(case[2], calls), model=DEFAULT_MODEL
    )
    if warm_cache:
        reviewer.assess(case[1])
        # Each public request constructs a fresh packet; receipts from the previous
        # request must be loaded as cache hits, not counted again.
        case[1].pop("source_inventory")
        calls.clear()
    reviewer.client = routed_client(
        case[2],
        calls,
        lambda request: (
            RuntimeError("private provider diagnostic")
            if request["text_format"] is GroundingChecks
            else None
        ),
    )
    with pytest.raises(RuntimeError) as caught:
        reviewer.review_graph.invoke({"packet": case[1]})
    error = caught.value
    assert error.usage.get("total_tokens", 0) == (0 if warm_cache else 90)
    assert error.model_calls == len(calls) == (1 if warm_cache else 4)
    assert error.usage_unknown_calls == 1
    saved = receipt(tmp_path, error)
    assert {step["stage"] for step in saved["completed_steps"]} == {
        "assessment",
        "generation",
        "audit_generation",
    }
    if warm_cache:
        assert sum(step["model_calls"] for step in saved["completed_steps"]) == 0
    assert "private provider diagnostic" not in json.dumps(saved)


def test_successful_audit_with_unreported_usage_is_recorded_as_unknown(case, tmp_path):
    calls = []
    delegate = fake_client(case[2], calls).responses.parse

    def parse(**kwargs):
        result = delegate(**kwargs)
        if kwargs["text_format"] is GroundingChecks:
            result.usage = None
        return result

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),
        model=DEFAULT_MODEL,
    )
    first = reviewer.assess(case[1])
    audit = reviewer.audit_grounding(case[1], first)
    assert audit["usage"] == {} and audit["model_calls"] == 1
    assert audit["usage_unknown_calls"] == 1
    cached = reviewer.audit_grounding(case[1], first)
    assert cached["usage_unknown_calls"] == 0 and cached["cached_usage_unknown_calls"] == 1


@pytest.mark.parametrize(
    "bad_text", ["x" * (MAX_SOURCE_CHARS + 1), "\n \n\t"], ids=["oversized", "empty"]
)
def test_invalid_source_size_or_empty_source_is_pending_without_losing_completed_candidate(
    case, tmp_path, bad_text
):
    index, _, raw = case
    req = Requirements(skill_years={"python": 3})
    index.candidates["B"] = index.candidates["A"].model_copy(
        update={"candidate_id": "B", "source_path": "B.txt"}
    )
    index.documents["B"] = {
        "text": bad_text,
        "sha256": sha256(bad_text.encode()).hexdigest(),
        "source_path": "B.txt",
    }
    matches = [Match(candidate_id=cid, name=cid, score=50, eligible=False) for cid in ("A", "B")]
    index.rank = lambda *args, **kwargs: SearchResult(
        matches=deepcopy(matches),
        corpus_size=2,
        retrieved_count=2,
        not_shortlisted_count=0,
        backend="test",
    )
    calls = []
    reviewer = ContextualReviewer(
        root=tmp_path, client=fake_client(raw, calls), model=DEFAULT_MODEL
    )
    reviews = reviewer.review(index, matches, req)
    assert [r["candidate_id"] for r in reviews] == ["A", "B"]
    assert reviews[0]["stage"] == "deep" and len(calls) == 4
    pending = reviews[1]
    assert pending["candidate_id"] == "B" and pending["packet"]["coverage"]["complete"] is False
    assert pending["model_calls"] == 0 and pending["usage"] == {}
    assert all(row["status"] == "uncertain" for row in pending["criteria"])
    assert (tmp_path / pending["failure_record"]).is_file()
    final = reviewer.apply_recommendations([pending], [matches[1].model_dump()], req)[0]
    assert final["recommendation"] == "hold"


def test_distinct_jobs_on_one_extracted_line_retain_distinct_source_identities(tmp_path):
    text = "Anonymous profile\nDeveloper at A | 2020-01 to 2020-12; Analyst at B | 2021-01 to 2021-12\n"
    candidate = Candidate(
        candidate_id="X", name="Anonymous profile", source_path="X.txt", skills=[]
    )
    index = SimpleNamespace(
        candidates={"X": candidate},
        documents={
            "X": {"text": text, "sha256": sha256(text.encode()).hexdigest(), "source_path": "X.txt"}
        },
        manifest={"snapshot_date": "2026-09-01"},
        engine_fingerprint="fixture",
        _check_sources=lambda *args: None,
    )
    units = [
        dict(
            unit_id="a",
            scope="employment",
            title="Developer at A",
            source_lines=["L2"],
            start_date="2020-01",
            end_date="2020-12",
            end_kind="calendar",
        ),
        dict(
            unit_id="b",
            scope="employment",
            title="Analyst at B",
            source_lines=["L2"],
            start_date="2021-01",
            end_date="2021-12",
            end_kind="calendar",
        ),
        dict(
            unit_id="heading",
            scope="other",
            title="Heading",
            source_lines=["L1"],
            start_date=None,
            end_date=None,
            end_kind="unknown",
        ),
    ]
    calls = []
    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client({"work_units": units, "findings": []}, calls),
        model=DEFAULT_MODEL,
    )
    packet = source_packet(index, "X", Requirements(minimum_years=1))
    first = prepare_inventory(reviewer, packet)
    second = prepare_inventory(reviewer, packet)
    jobs = [u for u in first["work_units"] if u["scope"] == "employment"]
    assert len(jobs) == 2 and len({u["unit_id"] for u in jobs}) == 2
    assert {u["title"] for u in jobs} == {"Developer at A", "Analyst at B"}
    assert {u["start_date"] for u in jobs} == {"2020-01", "2021-01"}
    assert jobs[0]["source_lines"] == jobs[1]["source_lines"] == ["L2"]
    assert first["work_units"] == second["work_units"] and first["version"] == second["version"]
    assert len(calls) == 2 and second["generation"]["cache_hit"]


def test_duplicate_inventory_entry_is_structurally_rejected_with_both_attempt_receipts(
    case, tmp_path
):
    raw = deepcopy(case[2])
    raw["work_units"].append({**deepcopy(raw["work_units"][0]), "unit_id": "duplicate_alias"})
    calls = []
    reviewer = ContextualReviewer(
        root=tmp_path, client=fake_client(raw, calls), model=DEFAULT_MODEL
    )
    with pytest.raises(ValueError) as caught:
        prepare_inventory(reviewer, case[1])
    error = caught.value
    assert error.usage["total_tokens"] == 60 and error.model_calls == 2
    assert all(issubclass(call["text_format"], SourceInventory) for call in calls)
    saved = receipt(tmp_path, error)
    assert len(saved["attempts"]) == 2
    assert all("duplicate" in attempt["validation_error"].lower() for attempt in saved["attempts"])


@pytest.mark.parametrize("change", ["label", "incidental_narrative"])
def test_agreed_support_retains_grounded_correction_without_becoming_a_dispute(
    case, tmp_path, change
):
    index, _, raw = case
    packet = source_packet(index, "A", Requirements(must_have=[["python"]]))
    finding = raw["findings"][0]
    finding.update(
        criterion_id=packet["semantic_criteria"][0]["criterion_id"],
        duration_associations=[],
        duration_basis="none",
        evidence_depth="personal_work",
        finding="Python is demonstrated by a personally owned project and dated service work.",
    )
    packet["semantic_criteria"][0]["standard"] = {"capability": "Personal Python development work."}
    compiled = compile_fixture(raw, packet)
    corrected = (
        "The dated developer responsibility explicitly describes personal Python service work."
    )

    def correction(data, payload):
        check = data["criteria"][0]
        check.update(reason=corrected, source_lines=["L2", "L3"], standard_clause="")
        if change == "label":
            check["standard_clause"] = ""
            check["verdict"]["relationship"] = "equivalent"
        else:
            check["claim_supported"] = False

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(raw, [], grounding_transform=correction),
        model=DEFAULT_MODEL,
    )
    result = reviewer.audit_grounding(packet, {**compiled, "raw": raw})
    row = result["assessment"]["criteria"][0]
    assert row["status"] == "supported" and row["finding"] == corrected
    assert [citation["passage_id"] for citation in row["citations"]] == ["L2", "L3"]
    assert not result["disputes"] and len(result["corrections"]) == 1
    saved = result["corrections"][0]
    assert saved["initial"]["finding"] == finding["finding"]
    assert saved["audit"]["reason"] == corrected
    if change == "incidental_narrative":
        assert saved["audit"]["claim_supported"] is False
    else:
        assert saved["initial"]["relationship"] == "direct"
        assert saved["audit"]["verdict"]["relationship"] == "equivalent"
