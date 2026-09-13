"""Validated model responses and live conversation semantics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from screening_agent.contracts import Candidate, Requirements
from screening_agent.criteria import build_criteria, default_scope
from screening_agent.planner import (
    OpenAIPlanner,
    PlanningError,
    _MeaningUpdate,
    _SkillAlternatives,
    _WirePlan,
    validate_requirements,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def bind_legacy_wire_fixture_provenance(monkeypatch, tmp_path, request):
    """Complete old operation fixtures with explicit controlled source clauses.

    Only the reserved fixture placeholder is replaced. Missing or invented provenance
    in the dedicated boundary tests is left unchanged and must fail in production.
    """
    original = OpenAIPlanner.__init__

    def initialize(self, root, *, client, model):
        if client is not None and hasattr(client, "responses"):
            parse = client.responses.parse

            def complete(**kwargs):
                response = parse(**kwargs)
                wire = response.output_parsed
                if (
                    wire is not None
                    and wire.requirements is not None
                    and any(
                        clause.quote == "FIXTURE_CURRENT_REQUEST"
                        for update in wire.requirements.criterion_updates
                        for clause in update.conditions
                    )
                ):
                    text = next(
                        message["content"].removeprefix("Current request:\n")
                        for message in kwargs["input"]
                        if message["content"].startswith("Current request:\n")
                    )
                    values = wire.requirements
                    req = validate_requirements(
                        Requirements(
                            role=values.role,
                            must_have=[[x] for x in values.required_skills]
                            + [g.any_of for g in values.alternative_skill_groups],
                            nice_to_have=values.nice_to_have,
                            minimum_years=values.minimum_years,
                            skill_years={row.skill: row.years for row in values.skill_years},
                        ),
                        self.vocabulary,
                    )
                    updates = []
                    for spec in build_criteria(req):
                        kind = spec["kind"]
                        depth = (
                            "role"
                            if kind == "role"
                            else "employment"
                            if kind.endswith("duration")
                            else "presence"
                        )
                        basis = (
                            "total_employment"
                            if kind == "total_duration"
                            else "skill_related_employment"
                            if kind == "skill_duration"
                            else "not_applicable"
                        )
                        updates.append(
                            _MeaningUpdate(
                                kind=kind,
                                alternatives=spec["alternatives"],
                                change="meaning",
                                conditions=[{"source_id": "current_request", "quote": text}],
                                examples=[],
                                assumptions=[],
                                evidence_scope=default_scope(kind, depth, basis),
                            )
                        )
                    wire.requirements.criterion_updates = updates
                return response

            client = SimpleNamespace(responses=SimpleNamespace(parse=complete))
        original(self, root, client=client, model=model)
        if request.node.get_closest_marker("live") is None:
            self.root = tmp_path

    monkeypatch.setattr(OpenAIPlanner, "__init__", initialize)


@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY")
    or not (os.getenv("OPENAI_MODEL") or os.getenv("MATCHING_MODEL")),
    reason="An authorized OpenAI key and model are required",
)
def test_live_exact_experience_then_role_followup_and_removal():
    planner = OpenAIPlanner.from_env(ROOT)
    first_text = "find someone with 3+ years of experience"
    first = planner.plan(first_text, None, candidates={}, shortlist=[], history=[])
    assert first.action == "search"
    assert first.requirements.minimum_years == 3
    assert first.requirements.role is None
    second_text = "and the condition that software developer a candidate must be"
    second = planner.plan(
        second_text,
        first.requirements,
        candidates={},
        shortlist=[],
        history=[{"role": "user", "content": first_text}],
    )
    assert second.action == "refine"
    assert second.requirements.role == "software_developer"
    assert second.requirements.minimum_years == 3
    assert second.requirements.must_have == []
    assert second.requirements.nice_to_have == []
    assert second.requirements.skill_years == {}
    changed = planner.plan(
        "Change the role to backend developer",
        second.requirements,
        candidates={},
        shortlist=[],
        history=[],
    )
    assert changed.action == "refine"
    assert changed.requirements.role == "backend_developer"
    assert changed.requirements.minimum_years == 3
    assert changed.requirements.must_have == []
    removed = planner.plan(
        "Remove the role requirement but keep the experience requirement",
        changed.requirements,
        candidates={},
        shortlist=[],
        history=[],
    )
    assert removed.action == "refine"
    assert removed.requirements.role is None
    assert removed.requirements.minimum_years == 3
    assert removed.requirements.must_have == []


def test_missing_live_configuration_is_explicit(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "MATCHING_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(PlanningError, match="requires"):
        OpenAIPlanner.from_env(ROOT)


def _wire(**changes):
    values = {
        "action": "search",
        "requirements": {
            "title": "React developer",
            "role": None,
            "required_skills": ["react"],
            "alternative_skill_groups": [],
            "nice_to_have": [],
            "minimum_years": 3,
            "skill_years": [{"skill": "react", "years": 2}],
            "source_text": "React developer",
            "criterion_updates": [
                {
                    "kind": "skill",
                    "alternatives": ["react"],
                    "change": "meaning",
                    "conditions": [
                        {"source_id": "current_request", "quote": "FIXTURE_CURRENT_REQUEST"}
                    ],
                    "examples": [],
                    "assumptions": [],
                    "evidence_scope": "source_claim",
                }
            ],
            "unresolved": [],
        },
        "candidate_ids": [],
        "top_n": 3,
        "explanation": "Search with explicit skill and duration thresholds.",
        "file_tool_name": None,
        "file_arguments": None,
    }
    values.update(changes)
    return _WirePlan.model_validate(values)


def test_model_plan_converts_strict_wire_schema_to_public_contract():
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_parsed=_wire())

    planner = OpenAIPlanner(
        ROOT, client=SimpleNamespace(responses=SimpleNamespace(parse=parse)), model="test-model"
    )
    result = planner.plan("Find React developers", None, candidates={}, shortlist=[], history=[])
    assert result.requirements.skill_years == {"react": 2}
    assert calls[0]["store"] is False
    assert calls[0]["text_format"] is planner.response_format
    assert "hidden reasoning" in calls[0]["instructions"]


def test_structured_schema_keeps_actions_constrained_and_skill_names_open():
    planner = OpenAIPlanner(ROOT, client=None, model="test-model")
    schema = planner.response_format.model_json_schema()
    assert "CanonicalSkill" not in schema["$defs"]
    properties = schema["$defs"]["NewSearchRequirements"]["properties"]
    assert properties["required_skills"]["items"]["type"] == "string"
    assert "AND" in properties["required_skills"]["description"]
    assert properties["nice_to_have"]["items"]["type"] == "string"
    values = _wire().model_dump(mode="json")
    values["requirements"]["required_skills"] = ["elixir", "root cause analysis"]
    parsed = planner.response_format.model_validate(values)
    assert parsed.requirements.required_skills == ["elixir", "root cause analysis"]
    values["action"] = "delete_candidate"
    with pytest.raises(ValidationError):
        planner.response_format.model_validate(values)


def test_initial_schema_cannot_generate_refinement_or_screening_without_requirements():
    planner = OpenAIPlanner(ROOT, client=None, model="test-model")
    initial_actions = planner.response_format.model_json_schema()["$defs"]["WorkflowAction"]["enum"]
    assert set(initial_actions) == {"search", "help", "file_tool", "approve"}
    existing_actions = planner.refinement_response_format.model_json_schema()["$defs"][
        "WorkflowAction"
    ]["enum"]
    assert {"refine", "compare", "explain", "questions", "deep_screen", "finalize"} <= set(
        existing_actions
    )


def test_dynamic_model_response_normalizes_enums_to_canonical_strings():
    planner = OpenAIPlanner(ROOT, client=None, model="test-model")
    parsed = planner.response_format.model_validate(_wire().model_dump(mode="json"))
    planner.client = SimpleNamespace(
        responses=SimpleNamespace(parse=lambda **_: SimpleNamespace(output_parsed=parsed))
    )
    planner = OpenAIPlanner(ROOT, client=planner.client, model="test-model")
    result = planner.plan("Find React candidates", None, candidates={}, shortlist=[], history=[])
    assert result.requirements.must_have == [["react"]]
    assert result.requirements.skill_years == {"react": 2}


def test_explicit_and_skills_and_or_groups_normalize_without_changing_logic():
    planner = OpenAIPlanner(ROOT, client=None, model="test-model")
    values = _wire().model_dump(mode="json")
    values["requirements"].update(
        required_skills=["react", "typescript"],
        alternative_skill_groups=[{"any_of": ["python", "javascript"]}],
    )
    parsed = planner.response_format.model_validate(values)
    planner.client = SimpleNamespace(
        responses=SimpleNamespace(parse=lambda **_: SimpleNamespace(output_parsed=parsed))
    )
    planner = OpenAIPlanner(ROOT, client=planner.client, model="test-model")
    result = planner.plan(
        "React and TypeScript and either Python or JavaScript",
        None,
        candidates={},
        shortlist=[],
        history=[],
    )
    assert result.requirements.must_have == [["react"], ["typescript"], ["python", "javascript"]]


def test_planning_context_uses_current_state_without_replaying_old_reports():
    calls = []
    response = SimpleNamespace(
        output_parsed=_wire(action="questions", requirements=None, candidate_ids=["A"], top_n=1)
    )

    def parse(**kwargs):
        calls.append(kwargs)
        return response

    planner = OpenAIPlanner(
        ROOT, client=SimpleNamespace(responses=SimpleNamespace(parse=parse)), model="test-model"
    )
    current = Requirements(
        must_have=[["react"]], minimum_years=2, source_text="Old criteria: require seven years"
    )
    candidate = Candidate(
        candidate_id="A",
        name="Jane Example",
        source_path="data/resumes/A.txt",
        skills=["react"],
        source_id="syn2_private_cohort_original",
    )
    query = "Prepare interview questions for the first person on this shortlist."
    result = planner.plan(
        query,
        current,
        candidates={"A": candidate},
        shortlist=[{"candidate_id": "A", "name": candidate.name}],
        history=[
            {"role": "user", "content": "Compare the top three candidates"},
            {"role": "assistant", "content": "OBSOLETE REPORT AND COMPARISON" * 1000},
            {
                "role": "document",
                "document": {
                    "path": "job.txt",
                    "sha256": "a" * 64,
                    "content": "Backend engineer; Python required.",
                },
            },
            {"role": "user", "content": query},
        ],
    )
    assert result.action == "questions"
    assert result.candidate_ids == ["A"]
    messages = calls[0]["input"]
    assert messages[-1]["content"] == "Current request:\n" + query
    context = json.loads(messages[0]["content"].split("\n", 1)[1])
    assert context["current_requirements"]["minimum_years"] == 2
    assert context["current_requirements"]["required_skills"] == ["react"]
    assert context["current_requirements"]["alternative_skill_groups"] == []
    assert "source_text" not in context["current_requirements"]
    assert context["previous_user_requests"] == ["Compare the top three candidates"]
    assert "OBSOLETE REPORT" not in str(messages)
    assert candidate.source_id not in str(messages)
    assert "source_id" not in context["candidate_directory"][0]
    assert context["last_read_document"]["content"] == "Backend engineer; Python required."
    assert (
        "and the condition that software developer a candidate must be"
        not in calls[0]["instructions"]
    )


def test_single_candidate_is_valid_for_questions_but_invalid_for_comparison():
    candidate = Candidate(
        candidate_id="A", name="Jane Example", source_path="A.txt", skills=["react"]
    )
    response = SimpleNamespace(
        output_parsed=_wire(action="compare", requirements=None, candidate_ids=["A"])
    )
    planner = OpenAIPlanner(
        ROOT,
        client=SimpleNamespace(responses=SimpleNamespace(parse=lambda **_: response)),
        model="test-model",
    )
    with pytest.raises(PlanningError, match="two distinct candidates"):
        planner.plan(
            "Compare Jane", Requirements(), candidates={"A": candidate}, shortlist=[], history=[]
        )
    response.output_parsed = _wire(
        action="questions", requirements=None, candidate_ids=["A"], top_n=1
    )
    result = planner.plan(
        "Prepare questions for Jane",
        Requirements(),
        candidates={"A": candidate},
        shortlist=[],
        history=[],
    )
    assert result.action == "questions"


@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY")
    or not (os.getenv("OPENAI_MODEL") or os.getenv("MATCHING_MODEL")),
    reason="An authorized OpenAI key and model are required",
)
def test_live_interview_request_after_comparison_and_refinements():
    first = Candidate(
        candidate_id="CAND_A",
        name="Jane Example",
        source_path="data/resumes/CAND_A.txt",
        skills=["react"],
    )
    second = Candidate(
        candidate_id="CAND_B",
        name="John Example",
        source_path="data/resumes/CAND_B.txt",
        skills=["react"],
    )
    history = [
        {"role": "user", "content": text}
        for text in [
            "Compare the top three people side by side",
            "Why did Jane rank above John?",
            "Raise minimum experience to seven years and make TypeScript mandatory",
            "Make TypeScript optional and lower minimum experience to two years",
        ]
    ]
    history.insert(2, {"role": "assistant", "content": "Previous comparison report. " * 1000})
    result = OpenAIPlanner.from_env(ROOT).plan(
        "Prepare interview questions for the first person on this shortlist.",
        Requirements(must_have=[["react"]], nice_to_have=["typescript"], minimum_years=2),
        candidates={first.candidate_id: first, second.candidate_id: second},
        shortlist=[
            {"candidate_id": candidate.candidate_id, "name": candidate.name}
            for candidate in (first, second)
        ],
        history=history,
    )
    assert result.action == "questions"
    assert result.candidate_ids == [first.candidate_id]
    assert result.top_n == 1


def test_model_provider_failure_has_no_automatic_offline_fallback():
    def parse(**_kwargs):
        raise TimeoutError("provider detail must not appear in the user-visible result")

    planner = OpenAIPlanner(
        ROOT, client=SimpleNamespace(responses=SimpleNamespace(parse=parse)), model="test-model"
    )
    with pytest.raises(PlanningError, match="TimeoutError") as caught:
        planner.plan("Find React", None, candidates={}, shortlist=[], history=[])
    assert "provider detail" not in str(caught.value)


def test_model_cannot_introduce_unknown_candidate_id():
    response = SimpleNamespace(
        output_parsed=_wire(action="compare", requirements=None, candidate_ids=["MADE_UP", "OTHER"])
    )
    planner = OpenAIPlanner(
        ROOT,
        client=SimpleNamespace(responses=SimpleNamespace(parse=lambda **_: response)),
        model="test-model",
    )
    with pytest.raises(PlanningError, match="do not exist"):
        planner.plan("Compare candidates", Requirements(), candidates={}, shortlist=[], history=[])


def test_model_can_select_a_filesystem_tool_from_broad_language():
    response = SimpleNamespace(
        output_parsed=_wire(
            action="file_tool",
            requirements=None,
            file_tool_name="list_files",
            file_arguments={
                "directory": "data/resumes",
                "extension": ".pdf",
                "filepath": None,
                "keyword": None,
                "content": None,
            },
        )
    )
    planner = OpenAIPlanner(
        ROOT,
        client=SimpleNamespace(responses=SimpleNamespace(parse=lambda **_: response)),
        model="test-model",
    )
    result = planner.plan(
        "Could you show me which PDF resumes are available?",
        None,
        candidates={},
        shortlist=[],
        history=[],
    )
    assert result.action == "file_tool"
    assert result.file_tool_name == "list_files"
    assert result.file_arguments.directory == "data/resumes"
    assert result.file_arguments.extension == ".pdf"


@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY")
    or not (os.getenv("OPENAI_MODEL") or os.getenv("MATCHING_MODEL")),
    reason="An authorized OpenAI key and model are required",
)
def test_live_model_selects_filesystem_tool():
    result = OpenAIPlanner.from_env(ROOT).plan(
        "Could you show me which PDF resumes are available?",
        None,
        candidates={},
        shortlist=[],
        history=[],
    )
    assert result.action == "file_tool"
    assert result.file_tool_name == "list_files"
    assert result.file_arguments.directory == "data/resumes"
    assert result.file_arguments.extension in {"pdf", ".pdf"}


@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY")
    or not (os.getenv("OPENAI_MODEL") or os.getenv("MATCHING_MODEL")),
    reason="An authorized OpenAI key and model are required",
)
def test_live_structured_planner_preserves_or_and_skill_duration():
    planner = OpenAIPlanner.from_env(ROOT)
    result = planner.plan(
        "Find candidates who must have React or JavaScript, with at least three years of total experience. TypeScript would be nice to have.",
        None,
        candidates={},
        shortlist=[],
        history=[],
    )
    assert result.action == "search"
    assert result.requirements.minimum_years == 3
    assert any(set(group) == {"react", "javascript"} for group in result.requirements.must_have)
    assert "typescript" in result.requirements.nice_to_have


@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY")
    or not (os.getenv("OPENAI_MODEL") or os.getenv("MATCHING_MODEL")),
    reason="An authorized OpenAI key and model are required",
)
def test_live_model_keeps_overall_professional_duration_out_of_skill_groups():
    result = OpenAIPlanner.from_env(ROOT).plan(
        "Find candidates with React and at least three years of overall professional experience. TypeScript is preferred.",
        None,
        candidates={},
        shortlist=[],
        history=[],
    )
    assert result.action == "search"
    assert result.requirements.must_have == [["react"]]
    assert result.requirements.minimum_years == 3
    assert result.requirements.nice_to_have == ["typescript"]


@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY")
    or not (os.getenv("OPENAI_MODEL") or os.getenv("MATCHING_MODEL")),
    reason="An authorized OpenAI key and model are required",
)
def test_live_refinement_requires_both_skills_and_retains_other_criteria():
    planner = OpenAIPlanner.from_env(ROOT)
    initial = planner.plan(
        "Find React candidates with at least three years of overall employment. TypeScript is preferred.",
        None,
        candidates={},
        shortlist=[],
        history=[],
    )
    assert initial.action == "search"
    current = initial.requirements
    assert current.must_have == [["react"]] and current.nice_to_have == ["typescript"]
    assert current.minimum_years == 3
    narrowed = planner.plan(
        "Raise the minimum overall experience to seven years and make TypeScript mandatory.",
        current,
        candidates={},
        shortlist=[],
        history=[],
    )
    assert narrowed.action == "refine"
    assert {tuple(group) for group in narrowed.requirements.must_have} == {
        ("react",),
        ("typescript",),
    }
    assert narrowed.requirements.nice_to_have == []
    assert narrowed.requirements.minimum_years == 7
    relaxed = planner.plan(
        "Actually, make TypeScript optional again and lower the minimum overall experience to two years.",
        narrowed.requirements,
        candidates={},
        shortlist=[],
        history=[],
    )
    assert relaxed.action == "refine"
    assert relaxed.requirements.must_have == [["react"]]
    assert relaxed.requirements.nice_to_have == ["typescript"]
    assert relaxed.requirements.minimum_years == 2


def test_model_repairs_absorbed_alternative_without_committing_invalid_criteria():
    current = Requirements(must_have=[["python"], ["sql"]])
    bad = _wire(action="refine")
    bad.requirements.required_skills = ["python"]
    bad.requirements.alternative_skill_groups = [_SkillAlternatives(any_of=["sql", "python"])]
    bad.requirements.skill_years = []
    bad = _WirePlan.model_validate(bad.model_dump())
    good = bad.model_copy(deep=True)
    good.requirements.alternative_skill_groups = []
    good.requirements.nice_to_have = ["sql"]
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_parsed=bad if len(calls) == 1 else good)

    planner = OpenAIPlanner(
        ROOT, client=SimpleNamespace(responses=SimpleNamespace(parse=parse)), model="test"
    )
    result = planner.plan(
        "SQL can be optional; Python must remain required.",
        current,
        candidates={},
        shortlist=[],
        history=[],
    )
    assert result.requirements.must_have == [["python"]]
    assert result.requirements.nice_to_have == ["sql"]
    assert current.must_have == [["python"], ["sql"]]
    assert len(calls) == 2
    assert "consistency check" in calls[-1]["input"][-1]["content"]


def test_model_consistency_correction_is_bounded_and_preserves_existing_state():
    current = Requirements(must_have=[["python"]])
    bad = _wire()
    bad.requirements.nice_to_have = ["react"]
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_parsed=bad)

    planner = OpenAIPlanner(
        ROOT, client=SimpleNamespace(responses=SimpleNamespace(parse=parse)), model="test"
    )
    with pytest.raises(PlanningError, match="after one correction"):
        planner.plan("Change preferences", current, candidates={}, shortlist=[], history=[])
    assert len(calls) == 2
    assert current.must_have == [["python"]]


def meaning_update(
    kind,
    alternatives,
    quote,
    *,
    change="meaning",
    examples=None,
    depth=None,
    basis=None,
    scope=None,
):
    depth = depth or (
        "role" if kind == "role" else "employment" if kind.endswith("duration") else "presence"
    )
    basis = basis or (
        "total_employment"
        if kind == "total_duration"
        else "skill_related_employment"
        if kind == "skill_duration"
        else "not_applicable"
    )
    return dict(
        kind=kind,
        alternatives=alternatives,
        change=change,
        conditions=[{"source_id": "current_request", "quote": quote}],
        examples=examples or [],
        assumptions=[],
        evidence_scope=None
        if change == "constraint"
        else scope or default_scope(kind, depth, basis),
    )


def scoped_wire(requirements, action, updates):
    return _wire(
        action=action,
        requirements={
            "title": requirements.title,
            "role": requirements.role,
            "required_skills": [group[0] for group in requirements.must_have if len(group) == 1],
            "alternative_skill_groups": [
                {"any_of": group} for group in requirements.must_have if len(group) > 1
            ],
            "nice_to_have": requirements.nice_to_have,
            "minimum_years": requirements.minimum_years,
            "skill_years": [
                {"skill": skill, "years": years}
                for skill, years in requirements.skill_years.items()
            ],
            "source_text": "",
            "semantic_brief": requirements.semantic_brief,
            "criterion_updates": updates,
            "unresolved": [],
        },
    )


def bound_planner(response):
    return OpenAIPlanner(
        ROOT,
        client=SimpleNamespace(
            responses=SimpleNamespace(
                parse=lambda **kwargs: SimpleNamespace(output_parsed=response)
            )
        ),
        model="test-model",
    )


def first_bound_request():
    req = Requirements(must_have=[["python"], ["sql"]], minimum_years=3)
    query = "Find Python and SQL candidates with three years overall employment."
    updates = [
        meaning_update(spec["kind"], spec["alternatives"], query) for spec in build_criteria(req)
    ]
    return (
        bound_planner(scoped_wire(req, "search", updates))
        .plan(query, None, candidates={}, shortlist=[], history=[])
        .requirements
    )


def test_refinement_changes_threshold_without_redrafting_other_criterion_meanings():
    current = first_bound_request()
    original = {
        meaning.criterion_id: meaning.model_dump_json() for meaning in current.criterion_meanings
    }
    query = "Raise the overall employment minimum to five years."
    proposed = current.model_copy(
        update={
            "minimum_years": 5,
            "semantic_brief": "Rephrased summary is not authoritative over frozen meanings.",
        }
    )
    response = scoped_wire(
        proposed, "refine", [meaning_update("total_duration", [], query, change="constraint")]
    )
    result = bound_planner(response).plan(query, current, candidates={}, shortlist=[], history=[])
    assert result.requirements.minimum_years == 5 and result.changed_criterion_ids == []
    assert {
        meaning.criterion_id: meaning.model_dump_json()
        for meaning in result.requirements.criterion_meanings
    } == original
    assert result.requirements.requirement_sources[-1].text == query
    assert current.minimum_years == 3


def test_illustrative_technologies_are_provenance_examples_not_new_and_or_requirements():
    current = first_bound_request()
    query = "For SQL accept practical relational database work, for example PostgreSQL or SQLAlchemy; keep the other requirements."
    condition, example = (
        "accept practical relational database work",
        "for example PostgreSQL or SQLAlchemy",
    )
    update = meaning_update(
        "skill",
        ["sql"],
        condition,
        examples=[{"source_id": "current_request", "quote": example}],
        depth="application",
        scope="personal_application",
    )
    response = scoped_wire(current, "refine", [update])
    result = bound_planner(response).plan(query, current, candidates={}, shortlist=[], history=[])
    sql_id = next(
        spec["criterion_id"] for spec in build_criteria(current) if spec["alternatives"] == ["sql"]
    )
    assert result.changed_criterion_ids == [sql_id]
    assert result.requirements.must_have == [["python"], ["sql"]]
    revised = next(
        meaning
        for meaning in result.requirements.criterion_meanings
        if meaning.criterion_id == sql_id
    )
    assert revised.conditions[0].quote == condition and revised.examples[0].quote == example
    assert revised.evidence_scope == "personal_application"
    for old in current.criterion_meanings:
        if old.criterion_id != sql_id:
            assert old == next(
                m
                for m in result.requirements.criterion_meanings
                if m.criterion_id == old.criterion_id
            )


@pytest.mark.parametrize(
    "damage",
    [
        "missing_new",
        "invented_quote",
        "wrong_scope",
        "constraint_new",
        "changed_without_new_clause",
        "duplicate",
    ],
)
def test_plan_provenance_rejects_missing_invented_or_incompatible_meaning(damage):
    req = Requirements(must_have=[["python"]])
    query = "Find Python candidates."
    updates = [meaning_update("skill", ["python"], query)]
    if damage == "missing_new":
        updates = []
    elif damage == "invented_quote":
        updates[0]["conditions"][0]["quote"] = "Only full-time employment may count."
    elif damage == "wrong_scope":
        updates[0]["evidence_scope"] = "all_employment"
    elif damage == "constraint_new":
        updates = [meaning_update("skill", ["python"], query, change="constraint")]
    elif damage == "changed_without_new_clause":
        updates[0]["conditions"][0]["source_id"] = "unknown_old_source"
    else:
        updates.append(dict(updates[0]))
    with pytest.raises(PlanningError, match="meaning|provenance|scope|constraint-only"):
        bound_planner(scoped_wire(req, "search", updates)).plan(
            query, None, candidates={}, shortlist=[], history=[]
        )


@pytest.mark.parametrize("error_kind", ["new_role_without_scope", "constraint_redrafts_scope"])
def test_one_correction_includes_full_meaning_binding_and_preserves_frozen_criteria(error_kind):
    current = first_bound_request()
    original = current.model_dump_json()
    if error_kind == "new_role_without_scope":
        query = "Also require a backend developer."
        proposed = current.model_copy(update={"role": "backend_developer"})
        good_update = meaning_update("role", ["backend_developer"], query)
        bad_update = {
            **good_update,
            "evidence_scope": None,
        }
    else:
        query = "Raise overall employment to five years."
        proposed = current.model_copy(update={"minimum_years": 5})
        good_update = meaning_update("total_duration", [], query, change="constraint")
        bad_update = {**good_update, "evidence_scope": "all_employment"}
    invalid = scoped_wire(proposed, "refine", [bad_update])
    corrected = scoped_wire(proposed, "refine", [good_update])
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            output_parsed=invalid if len(calls) == 1 else corrected,
            usage=SimpleNamespace(input_tokens=5, output_tokens=15, total_tokens=20),
        )

    planner = OpenAIPlanner(
        ROOT, client=SimpleNamespace(responses=SimpleNamespace(parse=parse)), model="test-model"
    )
    result = planner.plan(query, current, candidates={}, shortlist=[], history=[])
    assert len(calls) == 2 and result.action == "refine"
    assert current.model_dump_json() == original
    for meaning in current.criterion_meanings:
        assert meaning == next(
            row
            for row in result.requirements.criterion_meanings
            if row.criterion_id == meaning.criterion_id
        )
    assert json.loads(calls[1]["input"][-2]["content"]) == invalid.model_dump(mode="json")
    assert planner.last_trace["attempts"][0]["validation_error"]
    assert planner.last_trace["attempts"][1]["validation_error"] is None
    assert planner.last_trace["usage"]["total_tokens"] == 40
    assert planner.last_trace["model_calls"] == 2 and planner.last_trace["usage_unknown_calls"] == 0
    if error_kind == "new_role_without_scope":
        assert result.requirements.role == "backend_developer"
        assert len(result.changed_criterion_ids) == 1
    else:
        assert result.requirements.minimum_years == 5 and result.changed_criterion_ids == []


def test_wire_and_meaning_validation_share_one_correction_budget():
    current = first_bound_request()
    original = current.model_dump_json()
    query = "Raise overall employment to five years."
    proposed = current.model_copy(update={"minimum_years": 5})
    update = meaning_update("total_duration", [], query, change="constraint")
    invalid_wire = scoped_wire(proposed, "refine", [update])
    invalid_wire.requirements.nice_to_have = ["python"]
    invalid_meaning = scoped_wire(
        proposed, "refine", [{**update, "evidence_scope": "all_employment"}]
    )
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_parsed=invalid_wire if len(calls) == 1 else invalid_meaning)

    planner = OpenAIPlanner(
        ROOT, client=SimpleNamespace(responses=SimpleNamespace(parse=parse)), model="test-model"
    )
    with pytest.raises(PlanningError, match="after one correction") as caught:
        planner.plan(query, current, candidates={}, shortlist=[], history=[])
    assert len(calls) == caught.value.model_calls == 2
    assert len(caught.value.repair_trace) == 2
    assert "optional" in caught.value.repair_trace[0]["validation_error"]
    assert "constraint-only" in caught.value.repair_trace[1]["validation_error"]
    assert current.model_dump_json() == original


def test_provider_failure_after_meaning_error_preserves_known_usage_and_failure_trace():
    query = "Find Python candidates."
    invalid = scoped_wire(Requirements(must_have=[["python"]]), "search", [])
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise TimeoutError("private provider diagnostic")
        return SimpleNamespace(
            output_parsed=invalid,
            usage=SimpleNamespace(input_tokens=4, output_tokens=6, total_tokens=10),
        )

    planner = OpenAIPlanner(
        ROOT, client=SimpleNamespace(responses=SimpleNamespace(parse=parse)), model="test-model"
    )
    with pytest.raises(PlanningError, match="TimeoutError") as caught:
        planner.plan(query, None, candidates={}, shortlist=[], history=[])
    error = caught.value
    assert error.usage["total_tokens"] == 10 and error.model_calls == 2
    assert error.usage_unknown_calls == 1 and len(error.repair_trace) == 1
    assert error.repair_trace[0]["response"]["requirements"]["criterion_updates"] == []
    assert "private provider diagnostic" not in str(error)
    assert error.__suppress_context__


def test_explicit_removal_is_bound_to_a_previous_criterion_and_current_source():
    current = first_bound_request()
    query = "Remove SQL, preserving Python and the overall experience threshold."
    proposed = current.model_copy(update={"must_have": [["python"]]})
    update = meaning_update("skill", ["sql"], query, change="constraint")
    update["change"] = "remove"
    planner = bound_planner(scoped_wire(proposed, "refine", [update]))
    result = planner.plan(query, current, candidates={}, shortlist=[], history=[])
    assert result.requirements.must_have == [["python"]] and result.requirements.minimum_years == 3
    assert result.changed_criterion_ids == []
    assert len(result.requirements.criterion_meanings) == 2
    assert result.requirements.requirement_sources[-1].text == query
    assert (
        planner.last_trace["attempts"][0]["response"]["requirements"]["criterion_updates"][0][
            "change"
        ]
        == "remove"
    )
    assert current.must_have == [["python"], ["sql"]]


@pytest.mark.parametrize(
    "damage", ["still_present", "never_existed", "new_scope", "invented_quote"]
)
def test_removal_cannot_target_an_active_or_unrelated_criterion_or_invent_provenance(
    damage, tmp_path
):
    current = first_bound_request()
    query = "Remove SQL."
    proposed = current.model_copy(update={"must_have": [["python"]]})
    update = meaning_update("skill", ["sql"], query, change="constraint")
    update["change"] = "remove"
    if damage == "still_present":
        proposed = current
    elif damage == "never_existed":
        update["alternatives"] = ["rust"]
    elif damage == "new_scope":
        update["evidence_scope"] = "source_claim"
    else:
        update["conditions"][0]["quote"] = "Remove Python."
    planner = bound_planner(scoped_wire(proposed, "refine", [update]))
    planner.root = tmp_path
    with pytest.raises(PlanningError) as caught:
        planner.plan(query, current, candidates={}, shortlist=[], history=[])
    saved = json.loads((tmp_path / caught.value.failure_record).read_text())
    assert saved["model_calls"] == 2 and len(saved["attempts"]) == 2
    from screening_agent.policy import fingerprint

    assert saved["integrity"] == fingerprint({k: v for k, v in saved.items() if k != "integrity"})
    assert current.must_have == [["python"], ["sql"]]


def test_planning_repair_identifies_the_missing_selector():
    req = Requirements(must_have=[["python"]], skill_years={"python": 3})
    query = "Find candidates with three years of Python."
    updates = [meaning_update("skill_duration", ["python"], query)]
    with pytest.raises(PlanningError, match='Missing criterion selectors:.*"kind": "skill"'):
        bound_planner(scoped_wire(req, "search", updates)).plan(
            query, None, candidates={}, shortlist=[], history=[]
        )


def test_scope_repair_identifies_actual_and_allowed_dimensions():
    req = Requirements(must_have=[["python"]])
    query = "Find Python candidates."
    updates = [
        meaning_update("skill", ["python"], query, basis="explicit_usage", scope="explicit_usage")
    ]
    with pytest.raises(PlanningError, match="Criterion skill.*evidence scopes"):
        bound_planner(scoped_wire(req, "search", updates)).plan(
            query, None, candidates={}, shortlist=[], history=[]
        )


def test_grouped_independent_preferences_return_actionable_selector_repair():
    req = Requirements(nice_to_have=["cad", "solidworks"])
    query = "CAD and SolidWorks are separate optional preferences."
    updates = [meaning_update("skill", ["cad", "solidworks"], query)]
    with pytest.raises(PlanningError) as caught:
        bound_planner(scoped_wire(req, "search", updates)).plan(
            query, None, candidates={}, shortlist=[], history=[]
        )
    message = str(caught.value)
    assert '"alternatives": ["cad", "solidworks"]' in message
    assert '"alternatives": ["cad"]' in message
    assert '"alternatives": ["solidworks"]' in message
    assert "repair the selector, not the criteria" in message
    assert req.nice_to_have == ["cad", "solidworks"]


def test_new_search_cannot_patch_a_nonexistent_criterion():
    planner = OpenAIPlanner(ROOT, client=None, model="test-model")
    values = _wire().model_dump(mode="json")
    values["requirements"]["criterion_updates"][0]["change"] = "constraint"
    with pytest.raises(ValidationError):
        planner.response_format.model_validate(values)
    # The same structural operation remains available when there is prior state.
    planner.refinement_response_format.model_validate(values)


def test_model_wire_selects_scope_once_without_redundant_dimensions():
    schema = _MeaningUpdate.model_json_schema()
    assert "evidence_scope" in schema["properties"]
    assert "requested_depth" not in schema["properties"]
    assert "experience_basis" not in schema["properties"]
    query = "Find people who have applied an unfamiliar technology in their own work."
    req = Requirements(must_have=[["unfamiliar technology"]])
    result = bound_planner(
        scoped_wire(
            req,
            "search",
            [
                meaning_update(
                    "skill", ["unfamiliar technology"], query, scope="personal_application"
                )
            ],
        )
    ).plan(query, None, candidates={}, shortlist=[], history=[])
    meaning = result.requirements.criterion_meanings[0]
    assert (meaning.requested_depth, meaning.experience_basis, meaning.evidence_scope) == (
        "application",
        "not_applicable",
        "personal_application",
    )


def test_source_repair_reports_every_noncontiguous_list_quote():
    query = "Prefer CAD, SolidWorks, Ansys."
    req = Requirements(nice_to_have=["cad", "solidworks", "ansys"])
    updates = [
        meaning_update("skill", [skill], quote)
        for skill, quote in [
            ("cad", "Prefer CAD"),
            ("solidworks", "Prefer SolidWorks"),
            ("ansys", "Prefer Ansys"),
        ]
    ]
    with pytest.raises(PlanningError) as error:
        bound_planner(scoped_wire(req, "search", updates)).plan(
            query, None, candidates={}, shortlist=[], history=[]
        )
    assert '"quote": "Prefer SolidWorks"' in str(error.value)
    assert '"quote": "Prefer Ansys"' in str(error.value)
    assert '"quote": "Prefer CAD"' not in str(error.value)
    assert "contiguous source text" in str(error.value)
