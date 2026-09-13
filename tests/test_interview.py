"""Interview preparation preserves frozen criterion scope and source attribution."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from screening_agent.contracts import Candidate, Requirements
from screening_agent.criteria import RequirementCompiler, StandardSet, build_criteria, default_scope
from screening_agent.interview import INSTRUCTIONS, generate
from screening_agent.model_config import DEFAULT_MODEL


def compiled_case(tmp_path):
    """Compile controlled standards through the public compiler, without a provider."""
    calls = []

    def parse(**kwargs):
        payload = json.loads(kwargs["input"])
        calls.append(payload)
        standards = []
        for spec in payload["criteria"]:
            duration = spec["kind"] in {"total_duration", "skill_duration"}
            standards.append(
                dict(
                    criterion_id=spec["criterion_id"],
                    capability=spec["criterion"],
                    requested_depth="employment"
                    if duration
                    else "role"
                    if spec["kind"] == "role"
                    else "presence",
                    experience_basis="total_employment"
                    if spec["kind"] == "total_duration"
                    else "skill_related_employment"
                    if duration
                    else "not_applicable",
                    sufficient_evidence="Employment across occupations contributes to total employment; assess other criteria separately.",
                    equivalence_boundary="Use the requested capability and depth, without extra eligibility conditions.",
                    uncertainty_boundary="Seek clarification where the available source does not establish the requested facts.",
                )
            )
            row = standards[-1]
            row["evidence_scope"] = spec.get("meaning", {}).get(
                "evidence_scope",
                default_scope(spec["kind"], row["requested_depth"], row["experience_basis"]),
            )
        return SimpleNamespace(
            output_parsed=StandardSet.model_validate({"standards": standards}),
            usage=None,
            model=DEFAULT_MODEL,
        )

    requirements = Requirements(
        role="software_developer",
        must_have=[["python", "java"]],
        minimum_years=3,
        skill_years={"python": 2},
        nice_to_have=["sql"],
    )
    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    compiled = RequirementCompiler(tmp_path, client=client, model=DEFAULT_MODEL).compile(
        requirements
    )
    specs = build_criteria(compiled)
    candidate = Candidate(
        candidate_id="X7", name="Anonymous Candidate", source_path="data/resumes/X7.txt", skills=[]
    )
    checks = []
    index = SimpleNamespace(
        candidates={"X7": candidate}, _check_sources=lambda ids: checks.append(ids)
    )
    rows = [
        dict(
            criterion_id=spec["criterion_id"],
            criterion="Display label is not a join key",
            status="supported" if i % 2 else "uncertain",
            finding="The source records relevant work; ask about the remaining scope.",
            evidence=[
                dict(
                    candidate_id="X7",
                    source_path="data/resumes/X7.txt",
                    quote=f"Source evidence for criterion number {i}.",
                )
            ],
        )
        for i, spec in enumerate(specs)
    ]
    review = {"packet": {"semantic_criteria": specs}, "criteria": list(reversed(rows))}
    return index, compiled, review, calls, checks


def guide_client(calls, *, mutate=None):
    def parse(**kwargs):
        calls.append(kwargs)
        payload = json.loads(kwargs["input"])
        raw = {
            row["criterion_id"]: {
                "question": "Could you clarify the scope described in this cited example?",
                "purpose": "Clarify the existing evidence against the supplied frozen requirement.",
                "follow_up": "What evidence would help establish the remaining facts?",
                "strong_answer": "A concrete example that addresses the frozen criterion without adding conditions.",
                "evidence_ids": [item["evidence_id"] for item in row["evidence"]],
            }
            for row in payload["criteria"]
        }
        if mutate:
            mutate(raw, payload)
            parsed = SimpleNamespace(model_dump=lambda **_: raw)
        else:
            parsed = kwargs["text_format"].model_validate(raw)
        return SimpleNamespace(output_parsed=parsed, usage=None, model=DEFAULT_MODEL)

    return SimpleNamespace(responses=SimpleNamespace(parse=parse))


def test_interview_uses_exact_compiled_standards_and_stable_ids_for_every_scope(tmp_path):
    index, requirements, review, compiler_calls, checks = compiled_case(tmp_path)
    calls = []
    output, _ = generate(
        index, "X7", requirements, client=guide_client(calls), model=DEFAULT_MODEL, review=review
    )
    payload = json.loads(calls[0]["input"])
    actual = {row["criterion_id"]: row for row in payload["criteria"]}
    expected = {row["criterion_id"]: row for row in build_criteria(requirements)}
    assert len(compiler_calls) == 1 and set(actual) == set(expected)
    for cid, spec in expected.items():
        assert {key: actual[cid][key] for key in spec} == spec
    total = next(row for row in actual.values() if row["kind"] == "total_duration")
    skill = next(row for row in actual.values() if row["kind"] == "skill_duration")
    assert (
        total["minimum_years"] == 3 and total["standard"]["experience_basis"] == "total_employment"
    )
    assert (
        skill["minimum_years"] == 2
        and skill["standard"]["experience_basis"] == "skill_related_employment"
    )
    assert (
        next(row for row in actual.values() if row["kind"] == "role")["standard"][
            "experience_basis"
        ]
        == "not_applicable"
    )
    assert (
        next(row for row in actual.values() if not row["mandatory"])["standard"]["requested_depth"]
        == "presence"
    )
    assert {item["criterion_id"] for item in output} == set(expected)
    for question in output:
        criterion = question["criterion_id"]
        original = next(row for row in review["criteria"] if row["criterion_id"] == criterion)
        assert question["evidence"] == original["evidence"]
        assert question["criterion"] == expected[criterion]["criterion"]
    assert checks == [["X7"], ["X7"]]
    assert (
        "across occupations" in INSTRUCTIONS
        and "never an added screening condition" in INSTRUCTIONS
    )
    assert "does not require\ncontinuous" in INSTRUCTIONS


@pytest.mark.parametrize("damage", ["missing", "duplicate", "stale_threshold", "uncompiled"])
def test_interview_rejects_mismatched_or_uncompiled_criteria_before_provider(tmp_path, damage):
    index, requirements, review, _, _ = compiled_case(tmp_path)
    if damage == "missing":
        review["criteria"].pop()
    elif damage == "duplicate":
        review["criteria"].append(deepcopy(review["criteria"][0]))
    elif damage == "stale_threshold":
        requirements = requirements.model_copy(update={"minimum_years": 5})
    else:
        requirements = requirements.model_copy(
            update={"evidence_standards": [], "standards_signature": ""}
        )
    with pytest.raises(ValueError, match="current.*criteria|compiled evidence standards"):
        generate(index, "X7", requirements, client=object(), model=DEFAULT_MODEL, review=review)


def test_interview_rejects_cross_criterion_citations_even_when_both_exist(tmp_path):
    index, requirements, review, _, _ = compiled_case(tmp_path)
    calls = []

    def cross_citation(raw, payload):
        first, second = payload["criteria"][:2]
        raw[first["criterion_id"]]["evidence_ids"] = [second["evidence"][0]["evidence_id"]]

    with pytest.raises(ValueError):
        generate(
            index,
            "X7",
            requirements,
            client=guide_client(calls, mutate=cross_citation),
            model=DEFAULT_MODEL,
            review=review,
        )
    assert len(calls) == 1


def test_interview_allows_uncited_question_only_for_criterion_without_source_evidence(tmp_path):
    index, requirements, review, _, _ = compiled_case(tmp_path)
    review["criteria"][0]["evidence"] = []
    output, _ = generate(
        index, "X7", requirements, client=guide_client([]), model=DEFAULT_MODEL, review=review
    )
    question = next(
        row for row in output if row["criterion_id"] == review["criteria"][0]["criterion_id"]
    )
    assert question["evidence"] == []
