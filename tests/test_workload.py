"""Stage boundaries and the grounding contract for cross-candidate decisions."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from screening_agent.recommendation import DecisionMemo, MemoAudit, validate_memo


def memo_response(schema, payload):
    facts = {p["id"]: json.loads(p["text"]) for p in payload["source_lines"]}
    proposal = payload["proposal"]
    if issubclass(schema, MemoAudit):
        return schema.model_validate(
            {
                "claims": [
                    {
                        "claim_id": c["claim_id"],
                        "supported": True,
                        "reason": "Recorded fixture confirms this statement against its cited facts.",
                    }
                    for c in proposal["claims"]
                ],
                "priorities_supported": True,
                "priority_reason": "Equal support stays tied under the specified criteria.",
            }
        )
    decisions = proposal["fixed_decisions"]

    def claim(cid):
        key = next(k for k, f in facts.items() if f["candidate_id"] == cid)
        return {
            "text": "Verify this recorded evidence before an employment decision.",
            "fact_ids": [key],
        }

    ids = [d["candidate_id"] for d in decisions]
    advance = [d["candidate_id"] for d in decisions if d["recommendation"] == "hire"]
    pair = {
        "candidate_ids": ids[:2],
        "conclusion": {
            "text": "The supplied evidence provides no additional justified priority between these candidates.",
            "fact_ids": [claim(cid)["fact_ids"][0] for cid in ids[:2]],
        },
    }
    return schema.model_validate(
        {
            "summary": claim(ids[0]),
            "candidates": [
                {"candidate_id": cid, "rationale": claim(cid), "what_would_change": claim(cid)}
                for cid in ids
            ],
            "comparisons": [pair] if len(ids) > 1 else [],
            "advance_tiers": [advance] if advance else [],
        }
    )


@pytest.fixture
def memo_case():
    facts = {
        f"{cid}:criterion:0": {"candidate_id": cid, "finding": "The criterion is supported."}
        for cid in ["A", "B"]
    }
    decisions = [{"candidate_id": cid, "recommendation": "hire"} for cid in ["A", "B"]]
    payload = {
        "source_lines": [{"id": k, "text": json.dumps(v)} for k, v in facts.items()],
        "proposal": {"fixed_decisions": decisions},
    }
    return memo_response(DecisionMemo, payload).model_dump(), facts, decisions


def test_memo_accepts_supported_tie_with_complete_citations(memo_case):
    validate_memo(*memo_case)


@pytest.mark.parametrize(
    "mutation",
    [
        "invented_fact",
        "missing_side",
        "wrong_candidate",
        "duplicate_candidate",
        "unknown_priority",
        "missing_comparison",
        "duplicate_pair",
    ],
)
def test_memo_rejects_untraceable_or_incomplete_decisions(memo_case, mutation):
    memo, facts, decisions = deepcopy(memo_case)
    if mutation == "invented_fact":
        memo["summary"]["fact_ids"] = ["fake"]
    elif mutation == "missing_side":
        memo["comparisons"][0]["conclusion"]["fact_ids"] = ["A:criterion:0"]
    elif mutation == "wrong_candidate":
        memo["candidates"][0]["rationale"]["fact_ids"] = ["B:criterion:0"]
    elif mutation == "duplicate_candidate":
        memo["candidates"][1] = memo["candidates"][0]
    elif mutation == "unknown_priority":
        memo["advance_tiers"] = [["A", "C"]]
    elif mutation == "missing_comparison":
        memo["comparisons"] = []
    else:
        memo["comparisons"] *= 2
    with pytest.raises(ValueError):
        validate_memo(memo, facts, decisions)


def test_initial_graph_has_no_candidate_assessment_node(flow):  # noqa: F811
    # Fixture defined in test_agent_flows is deliberately reused below through import.
    state = flow.send("search", requirements=flow.requirements)
    assert flow.counts["deep"] == 0
    assert "contextual_screen" not in flow.session.graph.get_graph().nodes
    assert all(not m["assessments"] and not m["reviewed_assessments"] for m in state["shortlist"])
    assert state["detailed_reviews"] == [] and state["decision_memo"] == {}


from test_agent_flows import flow  # noqa: E402,F401


def test_rejected_memo_cannot_publish_or_mutate_decisions(memo_case):
    from screening_agent.contracts import Requirements
    from screening_agent.recommendation import recommend_cohort

    memo, _, decisions = memo_case
    for decision in decisions:
        decision["recommendation_reason"] = "All recorded mandatory criteria are supported."
    source = {"semantic_criteria": [], "snapshot_date": "2026-09-01"}
    reviews = [
        {
            "candidate_id": cid,
            "stage": "deep",
            "integrity": cid,
            "packet": source,
            "criteria": [
                {
                    "criterion_id": "c_test",
                    "criterion": "Python",
                    "mandatory": True,
                    "status": "supported",
                    "finding": "The source supports this criterion.",
                }
            ],
            "observations": [],
        }
        for cid in ["A", "B"]
    ]
    calls = []

    def generate(packet, **kwargs):
        calls.append(kwargs["stage"])
        if issubclass(kwargs["schema"], DecisionMemo):
            raw = memo
        else:
            raw = {
                "claims": [
                    {
                        "claim_id": item["claim_id"],
                        "supported": item["claim_id"] != "summary",
                        "reason": "The summary claims facts not supplied by its references.",
                    }
                    for item in kwargs["proposals"]["claims"]
                ],
                "priorities_supported": True,
                "priority_reason": "The proposed tie uses only the requested criteria.",
            }
        parsed = kwargs["schema"].model_validate(raw)
        kwargs["validate"](parsed)
        return {
            "raw": parsed.model_dump(),
            "usage": {"total_tokens": 30},
            "model_calls": 1,
            "cache_hit": False,
        }

    saved = []
    reviewer = SimpleNamespace(
        model="fixture",
        apply_recommendations=lambda *args: deepcopy(decisions),
        _generate_source=generate,
        _with_completed=lambda p, e, c: saved.extend(c),
    )
    before = deepcopy(decisions)
    with pytest.raises(ValueError, match="did not pass"):
        recommend_cohort(reviewer, reviews, decisions, Requirements(must_have=[["Python"]]))
    assert decisions == before and calls == ["decision_memo", "decision_audit"]
    assert [name for name, _ in saved] == calls


def test_memo_output_contract_excludes_invented_ledger_references(memo_case):
    from pydantic import ValidationError

    from screening_agent.recommendation import memo_output_schema

    memo, facts, decisions = deepcopy(memo_case)
    schema = memo_output_schema(facts, decisions)
    schema.model_validate(memo)
    memo["candidates"][0]["what_would_change"]["fact_ids"] = ["A:observation:3"]
    with pytest.raises(ValidationError, match="literal_error"):
        schema.model_validate(memo)
    contract = schema.model_json_schema()
    enums = []

    def collect(value):
        if isinstance(value, dict):
            if "enum" in value:
                enums.append(value["enum"])
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(contract)
    assert sorted(facts) in enums
    assert ["A", "B"] in enums


def test_memo_reference_feedback_identifies_every_bad_claim(memo_case):
    memo, facts, decisions = deepcopy(memo_case)
    memo["candidates"][0]["what_would_change"]["fact_ids"] = ["A:observation:3"]
    memo["candidates"][1]["rationale"]["fact_ids"] = ["B:criterion:0"] * 2
    with pytest.raises(ValueError) as error:
        validate_memo(memo, facts, decisions)
    message = str(error.value)
    assert "A:what_would_change" in message and "A:observation:3" in message
    assert "B:rationale" in message and "duplicate fact IDs ['B:criterion:0']" in message
    assert "Available fact IDs: ['A:criterion:0']" in message


def test_memo_audit_contract_uses_only_current_claims():
    from pydantic import ValidationError

    from screening_agent.recommendation import memo_audit_schema

    schema = memo_audit_schema([{"claim_id": "summary"}, {"claim_id": "A:rationale"}])
    with pytest.raises(ValidationError, match="literal_error"):
        schema.model_validate(
            {
                "claims": [
                    {
                        "claim_id": "old:rationale",
                        "supported": True,
                        "reason": "This claim belongs to another review.",
                    }
                ],
                "priorities_supported": True,
                "priority_reason": "The current criteria are preserved.",
            }
        )
