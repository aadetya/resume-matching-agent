"""Source identity, temporal arithmetic, contextual routing and independent review."""

from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import pytest

from screening_agent.contextual import (
    AssessmentResponse,
    ContextualResponse,
    ContextualReviewer,
    GroundingChecks,
    compile_response,
    source_packet,
)
from screening_agent.contracts import Candidate, Requirements
from screening_agent.criteria import StandardSet
from screening_agent.inventory import InventoryCheck, SourceInventory
from screening_agent.model_config import DEFAULT_MODEL


def bind_fixture_ids(response, criteria):
    """Bind ordinal fixture slots before sending them to the real ID contract.

    Only declared fixture placeholders are replaced. Invented IDs such as c99,
    duplicates and omitted findings remain unchanged for rejection tests.
    """
    for unit in response["work_units"]:
        unit.setdefault(
            "end_kind",
            "unknown"
            if unit["end_date"] is None
            else "ongoing"
            if unit["end_date"].casefold() in {"present", "current", "now"}
            else "calendar",
        )
    for finding in response["findings"]:
        for association in finding.get("duration_associations", []):
            if association.get("temporal_scope") == "dated_usage":
                association.setdefault(
                    "usage_end_kind", "calendar" if association.get("usage_end_date") else "unknown"
                )
    for i, finding in enumerate(response["findings"]):
        if i < len(criteria) and finding["criterion_id"] == f"c{i}":
            finding["criterion_id"] = criteria[i]["criterion_id"]
    for finding, spec in zip(response["findings"], criteria, strict=False):
        if spec["kind"] in {"total_duration", "skill_duration"}:
            selected = {a["unit_id"] for a in finding.get("duration_associations", [])}
            finding["unit_decisions"] = [
                {
                    "unit_id": u["unit_id"],
                    "decision": "include" if u["unit_id"] in selected else "exclude",
                    "reason": "The controlled fixture explicitly selects this employment."
                    if u["unit_id"] in selected
                    else "The controlled fixture does not credit this employment.",
                    "source_lines": u["source_lines"],
                }
                for u in response["work_units"]
                if u["scope"] == "employment"
            ]
        else:
            finding["unit_decisions"] = []
    return response


def compile_fixture(response, packet):
    bind_fixture_ids(response, packet["semantic_criteria"])
    packet["source_inventory"] = {
        "work_units": deepcopy(response["work_units"]),
        "complete": True,
        "version": "controlled-test-inventory",
    }
    return compile_response(ContextualResponse.model_validate(response), packet)


@pytest.fixture
def case():
    text = (
        "Anonymous profile\nPython Developer | 2020-01 to 2022-12\n"
        "Spearheaded services using Python.\nPython Developer | 2021-01 to 2023-12\n"
        "Maintained Python job processing and released service updates.\n"
        "PROJECT\nBuilt an ETL pipeline.\nTechnologies used: Python, PostgreSQL.\n"
    )
    candidate = Candidate(
        candidate_id="A", source_path="data/resumes/A.txt", name="Anonymous profile", skills=[]
    )
    index = SimpleNamespace(
        candidates={"A": candidate},
        documents={
            "A": {
                "text": text,
                "sha256": sha256(text.encode()).hexdigest(),
                "source_path": candidate.source_path,
            }
        },
        manifest={"snapshot_date": "2026-09-01"},
        engine_fingerprint="index-version",
        _check_sources=lambda *_: None,
    )
    req = Requirements(skill_years={"python": 3})
    packet = source_packet(index, "A", req)
    response = {
        "work_units": [
            {
                "unit_id": "first",
                "scope": "employment",
                "title": "Python Developer",
                "source_lines": ["L2", "L3"],
                "start_date": "2020-01",
                "end_date": "2022-12",
            },
            {
                "unit_id": "second",
                "scope": "employment",
                "title": "Python Developer",
                "source_lines": ["L4", "L5"],
                "start_date": "2021-01",
                "end_date": "2023-12",
            },
        ],
        "findings": [
            {
                "criterion_id": "c0",
                "status": "supported",
                "evidence_depth": "dated_association",
                "relationship": "direct",
                "finding": "Dated developer roles describe personal Python responsibilities.",
                "unit_ids": ["first", "second"],
                "source_lines": ["L2", "L3", "L4", "L5"],
                "usage_claims": [],
                "duration_basis": "skill_related_employment",
                "duration_associations": [
                    {
                        "unit_id": "first",
                        "basis": "skill_related_employment",
                        "assertion": "The first Python developer role describes personal Python service work.",
                        "source_lines": ["L2", "L3"],
                    },
                    {
                        "unit_id": "second",
                        "basis": "skill_related_employment",
                        "assertion": "The second Python developer role describes maintaining Python job processing.",
                        "source_lines": ["L4", "L5"],
                    },
                ],
            }
        ],
        "observations": [],
    }
    return index, packet, response


def compile_case(case):
    return compile_fixture(case[2], case[1])


def cap_second_role(case):
    """Keep the uncapped union fixture separate from the explicit-shorter-use source."""
    index, packet, response = case
    document = index.documents["A"]
    document["text"] = document["text"].replace(
        "Maintained Python job processing and released service updates.",
        "Used Python for two months; other duties used Java.",
    )
    document["sha256"] = sha256(document["text"].encode()).hexdigest()
    refreshed = source_packet(index, "A", Requirements(skill_years={"python": 3}))
    packet.clear()
    packet.update(refreshed)
    response["findings"][0]["duration_associations"][1].update(
        basis="explicit_usage",
        assertion="Python usage in the second role is explicitly limited to two months.",
    )


def test_union_is_computed_from_semantically_selected_source_dates(case):
    result = compile_case(case)
    assert result["criteria"][0]["duration"]["months"] == 48
    assert result["criteria"][0]["duration"]["basis"] == "skill_related_employment"
    assert "not proof of uninterrupted" in result["criteria"][0]["finding"]
    for unit in result["evidence_graph"]["work_units"]:
        for ev in unit["evidence"]:
            assert case[0].documents["A"]["text"][ev["start"] : ev["end"]] == ev["quote"]


def test_explicit_short_use_caps_the_role_without_erasing_other_work(case):
    cap_second_role(case)
    case[2]["findings"][0]["usage_claims"] = [
        {"unit_id": "second", "amount_text": "two", "unit_text": "months", "source_lines": ["L5"]}
    ]
    row = compile_case(case)["criteria"][0]
    assert row["duration"]["months"] == 36  # The shorter overlapping claim is not added again.
    assert row["duration"]["explicit_caps_months"] == {"second": 2}


def test_short_claim_alone_does_not_pass_a_long_role_as_skill_years(case):
    cap_second_role(case)
    case[2]["findings"][0].update(
        unit_ids=["second"],
        source_lines=["L4", "L5"],
        duration_basis="explicit_usage",
        duration_associations=[case[2]["findings"][0]["duration_associations"][1]],
        usage_claims=[
            {
                "unit_id": "second",
                "amount_text": "two",
                "unit_text": "months",
                "source_lines": ["L5"],
            }
        ],
    )
    row = compile_case(case)["criteria"][0]
    assert row["duration"]["months"] == 2 and row["status"] == "not_demonstrated"


def test_undated_project_never_inherits_nearby_job_dates(case):
    case[2]["work_units"][0].update(
        scope="project", source_lines=["L6", "L7", "L8"], start_date=None, end_date=None
    )
    case[2]["findings"][0].update(
        unit_ids=["first"], source_lines=["L7", "L8"], duration_associations=[]
    )
    row = compile_case(case)["criteria"][0]
    assert row["duration"]["months"] is None and row["status"] == "uncertain"


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda r: r["findings"][0].update(source_lines=["OTHER:L1"]), "invalid IDs"),
        (lambda r: r["findings"][0].update(unit_ids=["another-candidate"]), "unknown work unit"),
        (lambda r: r["findings"][0].update(source_lines=["L7"]), "outside its linked"),
        (lambda r: r["work_units"][0].update(start_date="1990-01"), "dates are not present"),
        (lambda r: r["findings"][0].update(criterion_id="c99"), "invented a criterion"),
        (lambda r: r["findings"][0].update(evidence_depth="contradictory"), "contradicts"),
        (lambda r: r["findings"][0].update(relationship="related"), "contradicts"),
        (
            lambda r: r["findings"][0].update(
                usage_claims=[
                    {
                        "unit_id": "second",
                        "amount_text": "thirty",
                        "unit_text": "months",
                        "source_lines": ["L5"],
                    }
                ]
            ),
            "quantity is absent",
        ),
    ],
)
def test_structural_and_source_validation_rejects_invented_support(case, mutation, match):
    mutation(case[2])
    with pytest.raises(ValueError, match=match):
        compile_case(case)


def test_missing_duration_does_not_become_zero_or_exclusion(case):
    case[2]["findings"][0].update(
        status="uncertain",
        unit_ids=[],
        source_lines=[],
        relationship="none",
        evidence_depth="not_found",
        duration_associations=[],
    )
    row = compile_case(case)["criteria"][0]
    assert row["duration"]["months"] is None
    assert row["status"] == "uncertain"


def test_contextual_prompt_does_not_depend_on_verb_or_skill_catalogue():
    from screening_agent.contextual_prompts import ASSESSMENT_PROMPT

    assert "No verb whitelist" in ASSESSMENT_PROMPT
    assert "first complete candidate assessment" in ASSESSMENT_PROMPT
    assert "No interview questions" in ASSESSMENT_PROMPT


def test_complete_source_audit_does_not_invoke_duration_repair(case, tmp_path):
    calls = []
    reviewer = ContextualReviewer(
        root=tmp_path, client=fake_client(case[2], calls), model=DEFAULT_MODEL
    )
    result = reviewer.review_graph.invoke({"packet": case[1]})
    assert issubclass(calls[0]["text_format"], SourceInventory)
    assert issubclass(calls[1]["text_format"], InventoryCheck)
    assert issubclass(calls[2]["text_format"], AssessmentResponse)
    assert [c["text_format"] for c in calls[3:]] == [GroundingChecks]
    assert result["steps"] == [
        "extract_and_audit_work_history",
        "assess_requirements",
        "audit_evidence",
    ]
    assert "repair_duration_inventory" not in result["steps"]


def test_unknown_or_unrelated_skill_context_cannot_accrue_role_months(case):
    for relationship, basis in [("direct", "unknown"), ("none", "skill_related_employment")]:
        case[2]["findings"][0].update(
            status="uncertain",
            relationship=relationship,
            duration_basis=basis,
            duration_associations=[],
        )
        row = compile_case(case)["criteria"][0]
        assert row["duration"]["months"] is None
        assert row["duration"]["maximum_months"] is None
        assert row["status"] == "uncertain"


def test_complete_calendar_date_is_capped_at_snapshot(case):
    from screening_agent.contextual import WorkUnit, _interval

    index = case[0]
    index.documents["A"]["text"] = (
        "Anonymous profile\nStarted January 2024; still employed at September 1, 2026.\n"
    )
    packet = source_packet(index, "A", Requirements(minimum_years=2))
    unit = WorkUnit(
        unit_id="role",
        scope="employment",
        title="Service role",
        source_lines=["L2"],
        start_date="January 2024",
        end_date="September 1, 2026",
        end_kind="calendar",
    )
    start, end = _interval(unit, packet)
    assert end - start == 32


def test_number_word_and_singular_duration_are_arithmetic_equivalents(case):
    cap_second_role(case)
    case[2]["findings"][0].update(
        unit_ids=["second"],
        source_lines=["L4", "L5"],
        duration_basis="explicit_usage",
        duration_associations=[case[2]["findings"][0]["duration_associations"][1]],
        usage_claims=[
            dict(unit_id="second", amount_text="2", unit_text="month", source_lines=["L5"])
        ],
    )
    assert compile_case(case)["criteria"][0]["duration"]["months"] == 2


def test_changed_semantic_qualifier_changes_requirement_identity():
    from screening_agent.policy import requirements_fingerprint

    req = Requirements(
        must_have=[["unlisted capability"]],
        semantic_brief="Has personally built an independent project",
    )
    changed = req.model_copy(
        update={"semantic_brief": "Has completed coursework; project work is not required"}
    )
    assert requirements_fingerprint(req) != requirements_fingerprint(changed)


def fake_client(response, calls, *, grounding_transform=None):
    def parse(**kwargs):
        import json

        calls.append(kwargs)
        payload = json.loads(kwargs["input"])
        if issubclass(kwargs["text_format"], StandardSet):
            parsed = StandardSet.model_validate(
                {
                    "standards": [
                        {
                            "criterion_id": spec["criterion_id"],
                            "capability": spec["criterion"],
                            "requested_depth": "employment"
                            if spec["kind"] in {"total_duration", "skill_duration"}
                            else "role"
                            if spec["kind"] == "role"
                            else "presence",
                            "experience_basis": "total_employment"
                            if spec["kind"] == "total_duration"
                            else "skill_related_employment"
                            if spec["kind"] == "skill_duration"
                            else "not_applicable",
                            "evidence_scope": spec.get("meaning", {}).get(
                                "evidence_scope",
                                "all_employment"
                                if spec["kind"] == "total_duration"
                                else "skill_related_employment"
                                if spec["kind"] == "skill_duration"
                                else "role_or_equivalent_work"
                                if spec["kind"] == "role"
                                else "source_claim",
                            ),
                            "sufficient_evidence": "The controlled fixture's dated employment supports this requested criterion.",
                            "equivalence_boundary": "Only the requested meaning is assessed; adjacent technology alone is insufficient.",
                            "uncertainty_boundary": "Missing dated source evidence remains unresolved.",
                        }
                        for spec in payload["criteria"]
                    ]
                }
            )
        elif issubclass(kwargs["text_format"], SourceInventory):
            bind_fixture_ids(response, [])
            units = deepcopy(response["work_units"])
            used = {line for u in units for line in u["source_lines"]}
            extra = [line["id"] for line in payload["source_lines"] if line["id"] not in used]
            if extra:
                units.append(
                    dict(
                        unit_id="remaining",
                        scope="other",
                        title="Remaining source context",
                        source_lines=extra,
                        start_date=None,
                        end_date=None,
                        end_kind="unknown",
                    )
                )
            parsed = SourceInventory(work_units=units)
        elif issubclass(kwargs["text_format"], InventoryCheck):
            parsed = InventoryCheck(
                complete=True,
                employment_complete=True,
                reason="The controlled test inventory retains all employment entries.",
                source_lines=[],
            )
        elif kwargs["text_format"] is GroundingChecks:
            bind_fixture_ids(response, payload["criteria"])
            targets = payload["targets"]
            associations = {target["target_id"]: target for target in targets["associations"]}

            def citation_lines(target):
                if "source_lines" in target:
                    return target["source_lines"]
                return list(
                    dict.fromkeys(
                        line
                        for association_id in target.get("association_target_ids", [])
                        for line in associations[association_id]["source_lines"]
                    )
                )

            verdicts = {
                kind: [
                    {
                        "target_id": target["target_id"],
                        "entailed": True,
                        "reason": "The controlled fixture source supports this assertion.",
                        "source_lines": citation_lines(target),
                    }
                    for target in targets[kind]
                ]
                for kind in ("criteria", "associations")
            }
            originals = {finding["criterion_id"]: finding for finding in response["findings"]}
            for target, check in zip(targets["criteria"], verdicts["criteria"], strict=True):
                if target["check_scope"] == "duration_evidence_coverage":
                    check["verdict"] = None
                else:
                    finding = originals[target["target_id"]]
                    check.update(
                        entailed=None,
                        claim_supported=True,
                        standard_clause=next(
                            (
                                c.get("standard", {}).get("capability", "")
                                for c in payload["criteria"]
                                if c["criterion_id"] == target["target_id"]
                            ),
                            "",
                        ),
                        verdict={
                            "assessed_status": finding["status"],
                            "relationship": finding["relationship"],
                            "evidence_depth": finding["evidence_depth"],
                        },
                    )
            verdicts["endpoints"] = [
                {
                    "target_id": target["target_id"],
                    "extraction_complete": True,
                    "reason": "The controlled fixture faithfully retains the available endpoint information.",
                    "source_lines": target["source_lines"],
                }
                for target in targets["endpoints"]
            ]
            verdicts["observations"] = [
                {
                    "observation_id": target["observation_id"],
                    "supported": True,
                    "reason": "The controlled fixture source supports this observation.",
                    "source_lines": target["source_lines"],
                }
                for target in targets["observations"]
            ]
            if grounding_transform:
                grounding_transform(verdicts, payload)
            parsed = GroundingChecks.model_validate(verdicts)
        elif kwargs["text_format"].__name__ in {"DecisionMemo", "MemoAudit"}:
            from test_workload import memo_response

            parsed = memo_response(kwargs["text_format"], payload)
        else:
            assert issubclass(kwargs["text_format"], AssessmentResponse)
            chosen = deepcopy(response)
            if payload.get("work_units"):
                mapping = {
                    u["unit_id"]: next(
                        v["unit_id"]
                        for v in payload["work_units"]
                        if v["source_lines"] == u["source_lines"]
                    )
                    for u in chosen["work_units"]
                }
                for finding in chosen["findings"]:
                    finding["unit_ids"] = [mapping[x] for x in finding["unit_ids"]]
                    for assoc in finding.get("duration_associations", []) + finding.get(
                        "usage_claims", []
                    ):
                        assoc["unit_id"] = mapping[assoc["unit_id"]]
                chosen["work_units"] = deepcopy(payload["work_units"])
            bind_fixture_ids(chosen, payload["criteria"])
            parsed = AssessmentResponse.model_validate(
                {k: chosen[k] for k in ("findings", "observations")}
            )
        return SimpleNamespace(
            output_parsed=parsed,
            usage=SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30),
            model=DEFAULT_MODEL,
        )

    return SimpleNamespace(responses=SimpleNamespace(parse=parse))


def test_cache_checks_signature_source_context_and_integrity(case, tmp_path):
    import json

    calls = []
    reviewer = ContextualReviewer(
        root=tmp_path, client=fake_client(case[2], calls), model=DEFAULT_MODEL
    )
    first = reviewer.assess(case[1])
    first_cache = next(
        p
        for p in reviewer.cache_dir.glob("*.json")
        if "findings" in json.loads(p.read_text())["response"]
    )
    second = reviewer.assess(case[1])
    assert first["usage"]["total_tokens"] == 30 and second["usage"] == {}
    assert len(calls) == 3 and second["cache_hit"]
    altered = deepcopy(case[1])
    altered["semantic_brief"] = "An amended depth requirement"
    reviewer.assess(altered)
    assert len(calls) == 4
    cache = first_cache
    saved = json.loads(cache.read_text())
    saved["response"]["findings"][0]["finding"] = "Corrupted cached source assertion"
    cache.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="integrity"):
        reviewer.assess(case[1])


def test_compile_allows_requested_familiarity_without_forcing_project_depth(case):
    packet = source_packet(
        case[0],
        "A",
        Requirements(
            must_have=[["python"]],
            semantic_brief="Require Python familiarity as listed in the resume; no project evidence needed.",
        ),
    )
    response = deepcopy(case[2])
    response["findings"][0].update(
        evidence_depth="skills_list", duration_basis="none", duration_associations=[]
    )
    assert compile_fixture(response, packet)["criteria"][0]["status"] == "supported"


def test_connected_graph_never_calls_legacy_qualification(case, tmp_path, monkeypatch):
    from pathlib import Path

    from matching_agent import MatchingSession
    from screening_agent.contracts import Match, Plan, RetrievalBatch, SearchResult
    from screening_agent.planner import OpenAIPlanner
    from screening_agent.policy import requirements_fingerprint

    root = Path(__file__).resolve().parents[1]
    index = case[0]
    index.candidates["B"] = index.candidates["A"].model_copy(
        update={"candidate_id": "B", "name": "Second example"}
    )
    index.documents["B"] = deepcopy(index.documents["A"])
    req = Requirements(minimum_years=3, semantic_brief="At least three years of total employment")
    response = deepcopy(case[2])
    response["findings"][0]["duration_basis"] = "total_employment"
    for association in response["findings"][0]["duration_associations"]:
        association["basis"] = "total_employment"
        association["assertion"] = (
            "This source describes a dated employment interval for total tenure."
        )
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("Connected workflow called legacy qualification")

    for name in ["assess", "deep_screen", "finalize", "compare", "search", "rank_strict"]:
        setattr(index, name, forbidden)
    index.retrieve = lambda requirements: RetrievalBatch(
        requirements_fingerprint=requirements_fingerprint(requirements),
        engine_fingerprint=index.engine_fingerprint,
        candidate_scores={"A": (1, 1), "B": (0.8, 0.9)},
        evidence_scores=[],
        corpus_size=2,
        retrieved_count=2,
        backend="fixture",
    )

    def expand(batch, requirements, *, progress=None):
        result = RetrievalBatch.model_validate(batch)
        result.retrieval_audit = {
            "local_scored_candidate_count": 2,
            "criteria": ["Total employment"],
            "query_count": 1,
            "context_scores": {},
            "expanded": True,
            "recovered_ids": [],
        }
        return result

    index.expand_retrieval = expand

    def rank(batch, requirements, *, top_k):
        return SearchResult(
            matches=[
                Match(
                    candidate_id=cid,
                    name=index.candidates[cid].name,
                    score=70,
                    eligible=False,
                    requirements_fingerprint=requirements_fingerprint(requirements),
                    engine_fingerprint=index.engine_fingerprint,
                )
                for cid in ["A", "B"]
            ],
            corpus_size=2,
            retrieved_count=2,
            backend="fixture",
        )

    index.rank = rank

    class Planner(OpenAIPlanner):
        def plan(self, text, current, **kwargs):
            return Plan(
                action=text,
                requirements=req.model_copy(update={"minimum_years": 4})
                if text == "refine"
                else req
                if text == "search"
                else None,
                candidate_ids=["A", "B"] if text == "compare" else [],
            )

    planner = Planner(root, client=fake_client(response, calls), model=DEFAULT_MODEL)
    with MatchingSession(
        root, planner=planner, index=index, db_path=tmp_path / "sessions.sqlite"
    ) as session:
        session.registry.requirement_compiler.cache_dir = tmp_path / "standards-cache"
        session.registry.deep_reviewer.cache_dir = tmp_path / "cache"
        session.registry.deep_reviewer.cache_dir.mkdir()
        initial = session.send("search")
        assert not initial.get("error"), initial.get("error")
        assert "initial_reviews" not in initial and all(
            not m["eligible"] and not m["assessments"] for m in initial["shortlist"]
        )
        deep = session.send("deep_screen")
        assert not deep.get("error"), deep.get("error")
        assert all(m["recommendation"] == "pending" for m in deep["shortlist"])
        calls_before_final = len(calls)
        final = session.send("finalize")
        assert not final.get("error"), final.get("error")
        assert len(calls) == calls_before_final + 2
        assert all(m["recommendation"] == "hire" for m in final["shortlist"])
        comparison = session.send("compare")
        assert not comparison.get("error"), comparison.get("error")
        assert all(c["contextual"] for c in comparison["comparison"]["candidates"])
        direct = session.registry.invoke(
            "rag_search", {"requirements": req.model_dump(), "top_k": 2}
        )
        assert direct["ok"] and len(direct["result"]["matches"]) == 2

        def interview(index, candidate_id, requirements, **kwargs):
            assert kwargs["review"]["contextual"]
            return [], {}

        monkeypatch.setattr("screening_agent.interview.generate", interview)
        questions = session.registry.invoke(
            "generate_interview_questions",
            {"candidate_id": "A"},
            requirements=req,
            review=deep["detailed_reviews"][0],
        )
        assert questions["ok"]
        previous = session.snapshot()

        def unavailable(*args, **kwargs):
            raise RuntimeError("Simulated provider outage")

        session.registry.deep_reviewer.initial = unavailable
        failed = session.send("refine")
        assert failed["error"]
        assert failed["shortlist"] == previous["shortlist"]
        assert failed["requirements_version"] == previous["requirements_version"]
        assert failed["requirements"] == previous["requirements"]


def test_explicit_zero_is_not_downgraded_to_unknown_by_threshold_arithmetic(case):
    case[2]["findings"][0].update(
        status="not_demonstrated",
        relationship="direct",
        evidence_depth="contradictory",
        duration_basis="unknown",
        duration_associations=[],
    )
    row = compile_case(case)["criteria"][0]
    assert row["duration"]["months"] == 0 and row["status"] == "not_demonstrated"


def test_source_valid_usage_fact_can_accompany_a_skill_without_a_years_threshold(case):
    cap_second_role(case)
    packet = source_packet(case[0], "A", Requirements(must_have=[["python"]]))
    response = deepcopy(case[2])
    response["findings"][0].update(
        unit_ids=["second"],
        source_lines=["L4", "L5"],
        duration_associations=[],
        usage_claims=[
            dict(unit_id="second", amount_text="two-month", unit_text="month", source_lines=["L5"])
        ],
    )
    result = compile_fixture(response, packet)
    assert result["criteria"][0]["duration"] is None
    assert result["criteria"][0]["status"] == "supported"


def mixed_employment_case():
    """Two jobs are useful context, but only one supplies positive Python tenure."""
    text = (
        "Anonymous profile\n"
        "Cashier | Sample Shop | 2017-01 to 2020-12\n"
        "Processed payments. I did not use Python in this job.\n"
        "Python Developer | Sample Service | 2021-01 to 2022-12\n"
        "I maintained Python services throughout this developer role.\n"
    )
    candidate = Candidate(
        candidate_id="A", source_path="data/resumes/A.txt", name="Anonymous profile", skills=[]
    )
    index = SimpleNamespace(
        candidates={"A": candidate},
        documents={
            "A": {
                "text": text,
                "sha256": sha256(text.encode()).hexdigest(),
                "source_path": candidate.source_path,
            }
        },
        manifest={"snapshot_date": "2026-09-01"},
        engine_fingerprint="controlled-mixed-employment",
        _check_sources=lambda *_: None,
    )
    packet = source_packet(
        index,
        "A",
        Requirements(
            skill_years={"python": 2},
            semantic_brief="At least two years of Python-related employment; unrelated jobs do not count.",
        ),
    )
    response = {
        "work_units": [
            {
                "unit_id": "cashier",
                "scope": "employment",
                "title": "Cashier",
                "source_lines": ["L2", "L3"],
                "start_date": "2017-01",
                "end_date": "2020-12",
            },
            {
                "unit_id": "developer",
                "scope": "employment",
                "title": "Python Developer",
                "source_lines": ["L4", "L5"],
                "start_date": "2021-01",
                "end_date": "2022-12",
            },
        ],
        "findings": [
            {
                "criterion_id": "c0",
                "status": "supported",
                "evidence_depth": "personal_work",
                "relationship": "direct",
                "finding": "The dated Python developer role supplies the requested two years of Python-related employment; the cashier job explicitly excludes Python.",
                "unit_ids": ["cashier", "developer"],
                "source_lines": ["L2", "L3", "L4", "L5"],
                "usage_claims": [],
                "duration_basis": "skill_related_employment",
                "duration_associations": [
                    {
                        "unit_id": "developer",
                        "basis": "skill_related_employment",
                        "assertion": "The developer role describes maintaining Python services throughout its dated employment.",
                        "source_lines": ["L4", "L5"],
                    }
                ],
            }
        ],
        "observations": [],
    }
    return index, packet, response


def test_cashier_contrary_context_does_not_supply_python_duration():
    index, _, response = mixed_employment_case()
    text = (
        "Anonymous profile\nCashier | Sample Shop | 2017-01 to 2020-12\n"
        "Processed payments. I have never used Python in employment; my only Python activity was coursework.\n"
    )
    index.documents["A"].update(text=text, sha256=sha256(text.encode()).hexdigest())
    packet = source_packet(index, "A", Requirements(skill_years={"python": 1}))
    response["work_units"] = response["work_units"][:1]
    response["findings"][0].update(
        status="not_demonstrated",
        evidence_depth="contradictory",
        relationship="direct",
        finding="The only supplied employment is cashier work, and Python is explicitly restricted to coursework rather than employment.",
        unit_ids=["cashier"],
        source_lines=["L2", "L3"],
        duration_associations=[],
    )
    row = compile_fixture(response, packet)["criteria"][0]
    assert row["unit_ids"] == ["cashier"]  # The contrary job remains inspectable context.
    assert row["duration"]["unit_ids"] == []
    assert row["duration"]["months"] == 0
    assert row["status"] == "not_demonstrated"


def test_mixed_jobs_credit_only_the_positive_duration_association():
    case = mixed_employment_case()
    row = compile_case(case)["criteria"][0]
    assert set(row["unit_ids"]) == {"cashier", "developer"}
    assert row["duration"]["unit_ids"] == ["developer"]
    assert row["duration"]["months"] == row["duration"]["maximum_months"] == 24
    assert row["status"] == "supported"


def test_duration_credit_is_stable_under_unit_renaming_and_citation_order():
    original = mixed_employment_case()
    baseline = compile_case(original)["criteria"][0]
    changed = deepcopy(original[2])
    names = {"cashier": "context_z", "developer": "positive_a"}
    changed["work_units"].reverse()
    for unit in changed["work_units"]:
        unit["unit_id"] = names[unit["unit_id"]]
        unit["source_lines"].reverse()
    finding = changed["findings"][0]
    finding["unit_ids"] = [names[uid] for uid in reversed(finding["unit_ids"])]
    finding["source_lines"].reverse()
    for association in finding["duration_associations"]:
        association["unit_id"] = names[association["unit_id"]]
        association["source_lines"].reverse()
    revised = compile_fixture(changed, original[1])["criteria"][0]
    for field in ("months", "minimum_months", "maximum_months", "basis"):
        assert revised["duration"][field] == baseline["duration"][field]
    assert revised["duration"]["unit_ids"] == ["positive_a"]
    assert revised["status"] == baseline["status"]


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda finding: finding["duration_associations"].append(
                deepcopy(finding["duration_associations"][0])
            ),
            "distinct",
        ),
        (
            lambda finding: finding["duration_associations"][0].update(unit_id="foreign"),
            "cited work units",
        ),
        (
            lambda finding: finding["duration_associations"][0].update(source_lines=["L2", "L3"]),
            "different work context",
        ),
    ],
)
def test_positive_duration_links_require_distinct_scoped_source_references(mutation, message):
    case = mixed_employment_case()
    mutation(case[2]["findings"][0])
    with pytest.raises(ValueError, match=message):
        compile_case(case)


def test_final_grounding_rejects_an_unsupported_independent_duration_link(tmp_path):
    import json

    case = mixed_employment_case()
    calls = []

    def reject_cashier(verdicts, payload):
        assert len(payload["targets"]["criteria"]) == 1
        assert len(payload["targets"]["associations"]) == 2
        for target, verdict in zip(
            payload["targets"]["associations"], verdicts["associations"], strict=True
        ):
            if target["unit_id"] == "cashier":
                verdict.update(
                    entailed=False,
                    reason="The cited cashier source explicitly says Python was not used in that job.",
                    source_lines=["L2", "L3"],
                )

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(case[2], calls, grounding_transform=reject_cashier),
        model=DEFAULT_MODEL,
    )
    raw = deepcopy(case[2])
    raw["findings"][0]["duration_associations"].append(
        {
            "unit_id": "cashier",
            "basis": "skill_related_employment",
            "assertion": "The cashier interval also represents sustained Python employment.",
            "source_lines": ["L2", "L3"],
        }
    )
    compiled = compile_fixture(raw, case[1])
    assert compiled["criteria"][0]["duration"]["months"] == 72
    audit = reviewer.audit_grounding(case[1], {**compiled, "raw": raw})
    final = audit["assessment"]
    assert final["criteria"][0]["duration"]["months"] == 24
    assert final["criteria"][0]["duration"]["unit_ids"] == ["developer"]
    assert final["criteria"][0]["status"] == "supported"
    assert len(final["raw"]["findings"][0]["duration_associations"]) == 1
    assert audit["rejected_associations"][0]["association"]["unit_id"] == "cashier"
    assert audit["model_calls"] == 1  # A semantic rejection is not a structural retry.
    assert len(calls) == 1 and calls[0]["text_format"] is GroundingChecks
    assert len(json.loads(calls[0]["input"])["targets"]["associations"]) == 2


@pytest.mark.parametrize("claimed_absence", [False, True])
def test_failed_coverage_without_accepted_links_remains_unknown(case, tmp_path, claimed_absence):
    if claimed_absence:
        case[2]["findings"][0].update(
            status="not_demonstrated",
            evidence_depth="contradictory",
            finding="The proposed assessment claims that no Python employment exists.",
        )

    def reject_criterion(verdicts, payload):
        verdicts["criteria"][0].update(
            entailed=False,
            reason="The proposed interpretation was not confirmed in this controlled audit.",
            source_lines=["L3"],
        )
        for verdict in verdicts["associations"]:
            verdict.update(
                entailed=False,
                reason="The controlled final source audit rejects this duration association.",
            )

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(case[2], [], grounding_transform=reject_criterion),
        model=DEFAULT_MODEL,
    )
    assessment = {**compile_case(case), "raw": deepcopy(case[2])}
    audit = reviewer.audit_grounding(case[1], assessment)
    row = audit["assessment"]["criteria"][0]
    assert row["status"] == "uncertain"
    assert row["duration"]["months"] is None
    assert row["duration"]["unit_ids"] == []
    assert row["duration_associations"] == []
    assert audit["audited_input_hash"] != audit["audited_result_hash"]


@pytest.mark.parametrize("minimum_years, expected_status", [(3, "supported"), (4, "uncertain")])
def test_incomplete_coverage_preserves_dated_role_credit_despite_undated_project_context(
    case, tmp_path, minimum_years, expected_status
):
    import json

    index, _, response = case
    text = (
        "Anonymous profile\nPython Developer | 2020-01 to 2022-12\n"
        "Spearheaded services using Python.\n"
        "PROJECT\nBuilt an ETL pipeline.\nTechnologies used: Python, PostgreSQL.\n"
    )
    index.documents["A"].update(text=text, sha256=sha256(text.encode()).hexdigest())
    packet = source_packet(
        index,
        "A",
        Requirements(
            skill_years={"python": minimum_years},
            semantic_brief="Python-related employment, assessed from a dated role and its work responsibilities.",
        ),
    )
    response["work_units"] = [
        response["work_units"][0],
        {
            "unit_id": "personal_project",
            "scope": "project",
            "title": "ETL pipeline",
            "source_lines": ["L4", "L5", "L6"],
            "start_date": None,
            "end_date": None,
        },
    ]
    finding = response["findings"][0]
    finding.update(
        finding="The dated Python developer role describes Python service work. An undated personal ETL project also lists Python, without establishing additional employment time.",
        unit_ids=["first", "personal_project"],
        source_lines=["L2", "L3", "L4", "L5", "L6"],
        duration_associations=finding["duration_associations"][:1],
    )
    calls = []

    def reject_coverage_only(verdicts, payload):
        assert len(verdicts["associations"]) == 1
        assert payload["targets"]["associations"][0]["unit_id"] == "first"
        verdicts["criteria"][0].update(
            entailed=False,
            reason="The undated Python project does not establish an employment interval, so this coverage assessment remains incomplete.",
            source_lines=["L4", "L5", "L6"],
        )

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(response, calls, grounding_transform=reject_coverage_only),
        model=DEFAULT_MODEL,
    )
    assessment = {
        **compile_fixture(response, packet),
        "raw": deepcopy(response),
    }
    audit = reviewer.audit_grounding(packet, assessment)
    row = audit["assessment"]["criteria"][0]
    assert row["duration"]["months"] == row["duration"]["minimum_months"] == 36
    assert row["duration"]["maximum_months"] is None
    assert row["duration"]["coverage_complete"] is False
    assert row["duration"]["basis"] == "skill_related_employment"
    assert row["duration"]["unit_ids"] == ["first"]
    assert row["status"] == expected_status
    assert row["duration_associations"]
    assert audit["criteria"][0]["entailed"] is False
    assert audit["associations"][0]["entailed"] is True
    assert audit["rejected_associations"] == []
    assert audit["criteria"][0]["reason"] in row["finding"]

    request = json.loads(calls[-1]["input"])
    target = request["targets"]["criteria"][0]
    assert target["association_target_ids"] == [f"{target['target_id']}:a0"]
    assert not {"context_unit_ids", "unit_ids", "source_lines", "finding", "status"} & target.keys()
    assert {unit["unit_id"] for unit in request["work_units"]} == {"first", "personal_project"}
    assert request["targets"]["associations"][0]["basis"] == "skill_related_employment"


def test_incomplete_coverage_does_not_make_an_accepted_short_claim_a_global_maximum(case, tmp_path):
    cap_second_role(case)
    response = case[2]
    finding = response["findings"][0]
    finding.update(
        duration_basis="explicit_usage",
        duration_associations=finding["duration_associations"][1:],
        usage_claims=[
            {
                "unit_id": "second",
                "amount_text": "two",
                "unit_text": "months",
                "source_lines": ["L5"],
            }
        ],
        finding="The second role explicitly describes two months of Python usage; the first role also describes Python work without an explicit usage duration.",
    )

    def incomplete_inventory(verdicts, payload):
        verdicts["criteria"][0].update(
            entailed=False,
            reason="The first dated Python role remains relevant but is absent from the proposed positive association inventory.",
            source_lines=["L2", "L3"],
        )

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(response, [], grounding_transform=incomplete_inventory),
        model=DEFAULT_MODEL,
    )
    audit = reviewer.audit_grounding(case[1], {**compile_case(case), "raw": deepcopy(response)})
    row = audit["assessment"]["criteria"][0]
    assert row["duration"]["months"] == 2
    assert row["duration"]["maximum_months"] is None
    assert row["duration"]["coverage_complete"] is False
    assert row["duration"]["unit_ids"] == ["second"]
    assert row["duration"]["explicit_caps_months"] == {"second": 2}
    assert row["status"] == "uncertain"
    assert audit["criteria"][0]["reason"] in row["finding"]
    assert audit["associations"][0]["entailed"] is True


def test_failed_non_duration_interpretation_still_becomes_uncertain(case, tmp_path):
    packet = source_packet(case[0], "A", Requirements(must_have=[["python"]]))
    response = deepcopy(case[2])
    response["findings"][0].update(duration_associations=[], duration_basis="none")

    def reject_interpretation(verdicts, payload):
        assert payload["targets"]["criteria"][0]["check_scope"] == "criterion_evidence_support"
        verdicts["criteria"][0].update(
            verdict={
                "assessed_status": "uncertain",
                "relationship": "direct",
                "evidence_depth": "personal_work",
            },
            reason="This controlled semantic audit did not confirm the requested interpretation.",
            source_lines=["L3"],
        )

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(response, [], grounding_transform=reject_interpretation),
        model=DEFAULT_MODEL,
    )
    audit = reviewer.audit_grounding(
        packet,
        {**compile_fixture(response, packet), "raw": response},
    )
    row = audit["assessment"]["criteria"][0]
    assert row["status"] == "uncertain"
    assert row["duration"] is None


@pytest.mark.parametrize(
    "audited_lines, expected_unit_ids, audited_reason",
    [
        (
            ["L2", "L3"],
            ["first"],
            "The dated Python developer entry describes service work using Python, supporting the requested applied skill.",
        ),
        (
            ["L7", "L8"],
            [],
            "The ETL pipeline project identifies Python in its technologies used, supporting application of the requested skill.",
        ),
        (
            ["L3", "L7", "L8"],
            [],
            "The developer responsibilities describe services using Python, and the ETL pipeline also names Python in its technologies used.",
        ),
    ],
)
def test_supported_criterion_publishes_audited_source_reason_instead_of_incidental_draft_claim(
    case, tmp_path, audited_lines, expected_unit_ids, audited_reason
):
    import json

    packet = source_packet(case[0], "A", Requirements(must_have=[["python"]]))
    response = deepcopy(case[2])
    incidental = "a sole-author personal project"
    response["findings"][0].update(
        finding=f"The candidate used Python in {incidental} and in developer work.",
        evidence_depth="personal_work",
        duration_associations=[],
        duration_basis="none",
    )
    calls = []

    def assess_requested_criterion(verdicts, payload):
        target = payload["targets"]["criteria"][0]
        assert target["check_scope"] == "criterion_evidence_support"
        assert "finding" in target
        assert incidental in json.dumps(payload)
        assert target["status"] == "supported"
        verdicts["criteria"][0].update(
            reason=audited_reason,
            source_lines=audited_lines,
        )

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(response, calls, grounding_transform=assess_requested_criterion),
        model=DEFAULT_MODEL,
    )
    audit = reviewer.audit_grounding(
        packet,
        {**compile_fixture(response, packet), "raw": response},
    )
    row = audit["assessment"]["criteria"][0]
    final_finding = audit["assessment"]["raw"]["findings"][0]
    assert row["status"] == "supported"
    assert row["finding"] == final_finding["finding"] == audited_reason
    assert incidental not in json.dumps(row)
    assert final_finding["source_lines"] == audited_lines
    assert row["citations"] == [{"passage_id": line} for line in audited_lines]
    assert row["unit_ids"] == expected_unit_ids
    assert row["duration"] is None
    assert row["duration_associations"] == []
    assert audit["criteria"][0]["entailed"] is None
    assert audit["criteria"][0]["revised"] is False
    for evidence in row["evidence"]:
        assert (
            case[0].documents["A"]["text"][evidence["start"] : evidence["end"]] == evidence["quote"]
        )


@pytest.mark.parametrize("earlier_status", ["uncertain", "not_demonstrated"])
def test_final_typed_verdict_can_correct_missed_positive_evidence(case, tmp_path, earlier_status):
    from screening_agent.policy import fingerprint

    packet = source_packet(case[0], "A", Requirements(must_have=[["python"]]))
    response = deepcopy(case[2])
    response["findings"][0].update(
        status=earlier_status,
        relationship="none",
        evidence_depth="not_found",
        finding="The earlier assessment did not identify Python application evidence.",
        duration_associations=[],
        duration_basis="none",
    )
    reason = "The dated developer role explicitly describes services using Python, which establishes the requested applied skill."

    def correct_missed_evidence(verdicts, payload):
        verdicts["criteria"][0].update(
            entailed=None,
            verdict={
                "assessed_status": "supported",
                "relationship": "direct",
                "evidence_depth": "personal_work",
            },
            reason=reason,
            source_lines=["L2", "L3"],
        )

    calls = []
    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(response, calls, grounding_transform=correct_missed_evidence),
        model=DEFAULT_MODEL,
    )
    assessment = {
        **compile_fixture(response, packet),
        "raw": response,
        "usage": {},
        "cache_hit": False,
        "elapsed_seconds": 0,
        "provider_model": "fixture",
    }
    audit = reviewer.audit_grounding(packet, assessment)
    row = audit["assessment"]["criteria"][0]
    assert row["status"] == "uncertain"
    assert audit["disputes"][0]["resolution"] == "needs_review"
    assert row["relationship"] == "direct"
    assert row["evidence_depth"] == "personal_work"
    assert reason in row["finding"]
    assert row["unit_ids"] == ["first"]
    assert row["citations"] == [{"passage_id": "L2"}, {"passage_id": "L3"}]
    assert audit["criteria"][0]["entailed"] is None
    assert audit["criteria"][0]["revised"] is True
    assert audit["criteria"][0]["verdict"]["assessed_status"] == "supported"
    assert audit["audited_result_hash"] == fingerprint(audit["assessment"]["raw"])
    assert audit["audited_criteria_hash"] == fingerprint(audit["assessment"]["criteria"])
    assert len(calls) == 1  # Correction is part of the existing final source review.
    record = reviewer._record(packet, assessment, audit["assessment"], stage="deep", audit=audit)
    check = record["claim_checks"][0]
    assert check["check_scope"] == "criterion_evidence_support"
    assert check["entailed"] is None
    assert check["verdict"]["assessed_status"] == "supported"
    assert check["revised"] is True
    assert record["changes"][0]["changed"] is (earlier_status != "uncertain")


def test_final_typed_verdict_preserves_contrary_source_in_a_downgrade(case, tmp_path):
    index = case[0]
    text = (
        "Anonymous profile\nPython Developer | 2020-01 to 2022-12\n"
        "Built services using Python.\nSummary\n"
        "I have never used Python in any employment or project.\n"
    )
    index.documents["A"].update(text=text, sha256=sha256(text.encode()).hexdigest())
    packet = source_packet(index, "A", Requirements(must_have=[["python"]]))
    response = deepcopy(case[2])
    response["work_units"] = [
        response["work_units"][0],
        {
            "unit_id": "contrary_summary",
            "scope": "other",
            "title": "Summary",
            "source_lines": ["L4", "L5"],
            "start_date": None,
            "end_date": None,
        },
    ]
    response["findings"][0].update(
        unit_ids=["first"],
        source_lines=["L2", "L3"],
        duration_associations=[],
        duration_basis="none",
    )
    reason = "The role claims Python service work, while the summary denies all Python use; the source does not resolve this contradiction."

    def flag_contradiction(verdicts, payload):
        verdicts["criteria"][0].update(
            verdict={
                "assessed_status": "uncertain",
                "relationship": "direct",
                "evidence_depth": "contradictory",
            },
            reason=reason,
            source_lines=["L3", "L5"],
        )

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(response, [], grounding_transform=flag_contradiction),
        model=DEFAULT_MODEL,
    )
    audit = reviewer.audit_grounding(
        packet,
        {**compile_fixture(response, packet), "raw": response},
    )
    row = audit["assessment"]["criteria"][0]
    assert row["status"] == "uncertain"
    assert row["evidence_depth"] == "contradictory"
    assert reason in row["finding"]
    assert row["unit_ids"] == ["first", "contrary_summary"]
    assert "Built services using Python." in row["evidence"][0]["quote"]
    assert "never used Python" in row["evidence"][1]["quote"]


@pytest.mark.parametrize(
    "mutation, error",
    [
        (lambda check: check.pop("verdict"), "Final grounding failed"),
        (lambda check: check.update(verdict=None), "requires a final verdict"),
        (lambda check: check.update(entailed=False), "requires a final verdict"),
        (
            lambda check: check["verdict"].update(assessed_status="hire"),
            "Final grounding failed",
        ),
        (lambda check: check.update(source_lines=[]), "must cite its original source"),
        (
            lambda check: check["verdict"].update(evidence_depth="contradictory"),
            "contradicts the response's own",
        ),
    ],
)
def test_final_criterion_verdict_fails_closed_when_malformed_or_ungrounded(
    case, tmp_path, mutation, error
):
    packet = source_packet(case[0], "A", Requirements(must_have=[["python"]]))
    response = deepcopy(case[2])
    response["findings"][0].update(duration_associations=[], duration_basis="none")

    def malformed_verdict(verdicts, payload):
        mutation(verdicts["criteria"][0])

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(response, [], grounding_transform=malformed_verdict),
        model=DEFAULT_MODEL,
    )
    assessment = {
        **compile_fixture(response, packet),
        "raw": response,
    }
    before = deepcopy(assessment)
    with pytest.raises((ValueError, RuntimeError), match=error):
        reviewer.audit_grounding(packet, assessment)
    assert assessment == before
    assert list(reviewer.cache_dir.glob("*.json")) == []


@pytest.mark.parametrize("wrong_scope", ["missing_boolean", "criterion_verdict"])
def test_duration_coverage_cannot_receive_a_non_duration_verdict(case, tmp_path, wrong_scope):
    def wrong_response(verdicts, payload):
        check = verdicts["criteria"][0]
        if wrong_scope == "missing_boolean":
            check["entailed"] = None
        else:
            check["verdict"] = {
                "assessed_status": "supported",
                "relationship": "direct",
                "evidence_depth": "dated_association",
            }

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(case[2], [], grounding_transform=wrong_response),
        model=DEFAULT_MODEL,
    )
    with pytest.raises(ValueError, match="Duration coverage requires a Boolean check"):
        reviewer.audit_grounding(case[1], {**compile_case(case), "raw": deepcopy(case[2])})


@pytest.mark.parametrize("reject_all", [False, True])
def test_duration_prose_uses_only_final_accepted_credit_after_pruning(case, tmp_path, reject_all):
    index = case[0]
    first_work = (
        "Maintained reporting services; programming language is not specified."
        if reject_all
        else "Maintained Python reporting services."
    )
    text = (
        "Anonymous profile\nDeveloper | Example Analytics | 2019-11 to 2021-03\n"
        + first_work
        + "\nDeveloper | Example Tools | 2017-01 to 2018-06\n"
        "Supported internal reporting workflow; programming language is not specified.\n"
    )
    index.documents["A"].update(text=text, sha256=sha256(text.encode()).hexdigest())
    packet = source_packet(index, "A", Requirements(skill_years={"python": 3}))
    response = deepcopy(case[2])
    response["work_units"][0].update(
        title="Developer at Example Analytics", start_date="2019-11", end_date="2021-03"
    )
    response["work_units"][1].update(
        title="Developer at Example Tools", start_date="2017-01", end_date="2018-06"
    )
    stale_aggregate = (
        "The two roles establish 35 calendar months (2.92 years) of Python employment."
    )
    response["findings"][0]["finding"] = stale_aggregate
    response["findings"][0]["duration_associations"][0]["assertion"] = (
        "The Example Analytics role describes Python reporting services."
    )
    response["findings"][0]["duration_associations"][1]["assertion"] = (
        "The Example Tools role also describes Python reporting work."
    )

    def prune_unfounded_associations(verdicts, payload):
        verdicts["criteria"][0].update(
            entailed=False,
            reason="The proposed inventory contains work whose programming language is not specified.",
            source_lines=["L3", "L5"],
        )
        for target, check in zip(
            payload["targets"]["associations"], verdicts["associations"], strict=True
        ):
            if reject_all or target["unit_id"] == "second":
                check.update(
                    entailed=False,
                    reason="The role describes reporting work but supplies no Python association.",
                )

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(response, [], grounding_transform=prune_unfounded_associations),
        model=DEFAULT_MODEL,
    )
    assessment = {
        **compile_fixture(response, packet),
        "raw": response,
        "usage": {},
        "cache_hit": False,
        "elapsed_seconds": 0,
        "provider_model": "fixture",
    }
    assert assessment["criteria"][0]["duration"]["months"] == 35
    audit = reviewer.audit_grounding(packet, assessment)
    row = audit["assessment"]["criteria"][0]
    authoritative, notes = row["finding"].split("\n\n", 1)
    assert "35" not in authoritative and "2.92" not in authoritative
    assert row["duration"]["months"] == (None if reject_all else 17)
    assert row["status"] == "uncertain"
    if reject_all:
        assert "Duration remains unknown" in authoritative
        assert "Accepted source associations:" not in authoritative
    else:
        assert "At least 17 months (1.42 years)" in authoritative
        assert "2019-11 to 2021-03" in authoritative
        assert "Example Analytics" in authoritative
        assert "Example Tools" not in authoritative
    assert "Unresolved coverage:" in notes
    assert "Rejected duration link (second):" in notes
    assert audit["proposed_assessment"]["findings"][0]["finding"] == stale_aggregate
    record = reviewer._record(packet, assessment, audit["assessment"], stage="deep", audit=audit)
    assert record["pre_audit_assessment"]["findings"][0]["finding"] == stale_aggregate


def test_unknown_duration_retains_separately_labelled_source_assessment(case, tmp_path):
    response = deepcopy(case[2])
    response["findings"][0].update(
        status="uncertain",
        duration_associations=[],
        duration_basis="unknown",
        finding="An earlier draft proposed 35 months without an accepted dated association.",
    )

    def explain_missing_association(verdicts, payload):
        verdicts["criteria"][0].update(
            reason="The undated project supplies skill context but no dated employment association.",
            source_lines=["L7", "L8"],
        )

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(response, [], grounding_transform=explain_missing_association),
        model=DEFAULT_MODEL,
    )
    audit = reviewer.audit_grounding(
        case[1],
        {**compile_fixture(response, case[1]), "raw": response},
    )
    row = audit["assessment"]["criteria"][0]
    assert row["duration"]["months"] is None
    assert "35 months" not in row["finding"]
    assert "Source assessment: The undated project" in row["finding"]


@pytest.mark.parametrize("kind", ["criteria", "associations"])
def test_final_grounding_cannot_omit_a_criterion_or_positive_link(case, tmp_path, kind):
    def omit_target(verdicts, payload):
        verdicts[kind].pop()

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(case[2], [], grounding_transform=omit_target),
        model=DEFAULT_MODEL,
    )
    with pytest.raises(ValueError, match="omitted, reordered or invented"):
        reviewer.audit_grounding(case[1], {**compile_case(case), "raw": deepcopy(case[2])})


@pytest.mark.parametrize(
    "change",
    [
        "remove_audit_binding",
        "change_final_association",
        "remove_criteria_binding",
        "change_derived_criteria",
    ],
)
def test_deep_review_cannot_reuse_an_audit_for_a_changed_final_assessment(case, tmp_path, change):
    from screening_agent.contracts import CriterionAssessment, Match
    from screening_agent.policy import fingerprint, requirements_fingerprint

    index, _, response = case
    req = Requirements(skill_years={"python": 3})
    reviewer = ContextualReviewer(
        root=tmp_path, client=fake_client(response, []), model=DEFAULT_MODEL
    )
    match = Match(
        candidate_id="A",
        name="Anonymous profile",
        score=75,
        eligible=True,
        requirements_fingerprint=requirements_fingerprint(req),
        engine_fingerprint=index.engine_fingerprint,
        assessments=[
            CriterionAssessment.model_validate(row) for row in reviewer.ledger(compile_case(case))
        ],
    )
    review = reviewer.review(index, [match], req)[0]
    assert review["audited_result_hash"] == fingerprint(review["final_assessment"])
    assert review["audited_criteria_hash"] == fingerprint(review["criteria"])
    altered = deepcopy(review)
    if change == "remove_audit_binding":
        altered.pop("audited_result_hash")
    elif change == "change_final_association":
        altered["final_assessment"]["findings"][0]["duration_associations"][0]["assertion"] = (
            "A different interval interpretation was introduced after the source audit."
        )
    elif change == "remove_criteria_binding":
        altered.pop("audited_criteria_hash")
    else:
        altered["criteria"][0]["status"] = "uncertain"
        altered["criteria"][0]["duration"]["months"] = 12
        assert altered["final_assessment"] == review["final_assessment"]
    # Repacking the outer record must not make the preceding semantic audit applicable.
    altered["integrity"] = fingerprint(
        {key: value for key, value in altered.items() if key != "integrity"}
    )
    with pytest.raises(ValueError, match="changed after (its|their) grounding audit"):
        reviewer.verify([altered], index, [match], req)


def test_calendar_format_normalization_requires_same_source_month(case):
    from screening_agent.contextual import WorkUnit, _interval

    index = case[0]
    index.documents["A"]["text"] = (
        "Profile\nStarted January 2024, employed through September 1, 2026.\n"
    )
    packet = source_packet(index, "A", Requirements(minimum_years=2))
    unit = WorkUnit(
        unit_id="role",
        scope="employment",
        title="Dated role",
        source_lines=["L2"],
        start_date="2024-01",
        end_date="2026-09-01",
        end_kind="calendar",
    )
    start, end = _interval(unit, packet)
    assert end - start == 32
    with pytest.raises(ValueError, match="dates are not present"):
        _interval(unit.model_copy(update={"start_date": "2020-01"}), packet)


@pytest.mark.parametrize("shorter_second_role", [False, True])
def test_final_duration_audit_receives_source_facts_without_calculated_totals(
    case, tmp_path, shorter_second_role
):
    import json

    if shorter_second_role:
        cap_second_role(case)
        case[2]["findings"][0]["usage_claims"] = [
            {
                "unit_id": "second",
                "amount_text": "two",
                "unit_text": "months",
                "source_lines": ["L5"],
            }
        ]
    calls = []
    reviewer = ContextualReviewer(
        root=tmp_path, client=fake_client(case[2], calls), model=DEFAULT_MODEL
    )
    assessment = {**compile_case(case), "raw": deepcopy(case[2])}
    audit = reviewer.audit_grounding(case[1], assessment)
    request = json.loads(calls[-1]["input"])
    criterion = request["targets"]["criteria"][0]
    associations = {target["unit_id"]: target for target in request["targets"]["associations"]}

    # The model verifies source association, while code calculates 48 union months.
    # A shorter claim remains attached only to its own role; its unknown placement
    # changes the calculated union to 36–38 without asking the model to recompute it.
    expected_union = 36 if shorter_second_role else 48
    expected_upper = 38 if shorter_second_role else 48
    assert criterion["check_scope"] == "duration_evidence_coverage"
    assert "finding" not in criterion and "status" not in criterion
    assert criterion["association_target_ids"] == [
        f"{criterion['target_id']}:a0",
        f"{criterion['target_id']}:a1",
    ]
    assert not {"context_unit_ids", "unit_ids", "source_lines"} & criterion.keys()
    forbidden_fields = {
        "computed_duration",
        "proposed_duration",
        "unit_credit",
        "calendar_span_months",
        "credited_months_before_overlap",
        "minimum_years",
        "months",
        "years",
        "minimum_months",
        "maximum_months",
    }

    def assert_source_only(value):
        if isinstance(value, dict):
            assert not forbidden_fields.intersection(value)
            for child in value.values():
                assert_source_only(child)
        elif isinstance(value, list):
            for child in value:
                assert_source_only(child)

    assert_source_only(request)
    source_dates = {
        "first": {"start": "2020-01", "end": "2022-12", "end_kind": "calendar"},
        "second": {"start": "2021-01", "end": "2023-12", "end_kind": "calendar"},
    }
    for unit_id, target in associations.items():
        assert target["criterion_id"] == criterion["target_id"]
        assert target["source_dates"] == source_dates[unit_id]
        assert target["assertion"]
        assert all(claim["unit_id"] == unit_id for claim in target["usage_claims"])
    assert len(associations["second"]["usage_claims"]) == int(shorter_second_role)
    if shorter_second_role:
        assert associations["second"]["usage_claims"] == case[2]["findings"][0]["usage_claims"]
    assert associations["first"]["usage_claims"] == []
    assert audit["assessment"]["criteria"][0]["duration"]["months"] == expected_union
    assert audit["assessment"]["criteria"][0]["duration"]["maximum_months"] == expected_upper
    assert audit["assessment"]["criteria"][0]["duration"]["explicit_caps_months"] == (
        {"second": 2} if shorter_second_role else {}
    )
    explanation = audit["assessment"]["criteria"][0]["finding"]
    if shorter_second_role:
        assert "36\u201338 months (3.00\u20133.17 years)" in explanation
        assert "Explicit usage is limited to 2 months." in explanation
        assert "placement of explicit usage within overlapping roles is unresolved" in explanation
    else:
        assert "48 months (4.00 years)" in explanation
    assert "after merging overlaps" in explanation


def test_audit_structural_repair_uses_invalid_response_and_accounts_for_both_calls(case, tmp_path):
    import json

    calls = []
    invalid = []

    def bad_first_citation(verdicts, payload):
        if len(calls) == 1:
            verdicts["associations"][0]["source_lines"] = ["L999"]
            invalid.append(deepcopy(verdicts))

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(case[2], calls, grounding_transform=bad_first_citation),
        model=DEFAULT_MODEL,
    )
    proposed = {**compile_case(case), "raw": deepcopy(case[2])}
    audit = reviewer.audit_grounding(case[1], proposed)
    assert len(calls) == audit["model_calls"] == 2
    first, second = [json.loads(call["input"]) for call in calls]
    assert "structural_repair" not in first
    assert {key: value for key, value in second.items() if key != "structural_repair"} == first
    assert second["structural_repair"]["invalid_response"] == GroundingChecks.model_validate(
        invalid[0]
    ).model_dump(mode="json")
    assert (
        second["structural_repair"]["validation_error"]
        == audit["repair_trace"][0]["validation_error"]
    )
    assert (
        "L999" in second["structural_repair"]["invalid_response"]["associations"][0]["source_lines"]
    )
    assert "structural validation failure" in calls[1]["instructions"]
    assert "Preserve valid semantic judgments" in calls[1]["instructions"]
    assert audit["usage"] == {"input_tokens": 20, "output_tokens": 40, "total_tokens": 60}
    assert [step["attempt"] for step in audit["repair_trace"]] == [1, 2]
    assert audit["repair_trace"][1]["validation_error"] is None
    assert audit["assessment"]["criteria"][0]["duration"]["months"] == 48
    assert audit["assessment"]["criteria"][0]["status"] == "supported"
    assert audit["proposed_assessment"] == proposed["raw"]

    # Restarting may reuse the valid repaired result, but must not count prior calls as new cost.
    restarted = ContextualReviewer(root=tmp_path, client=object(), model=DEFAULT_MODEL)
    cached = restarted.audit_grounding(case[1], proposed)
    assert cached["cache_hit"] and cached["model_calls"] == 0 and cached["usage"] == {}
    assert cached["cached_model_calls"] == 2 and cached["cached_usage"] == audit["usage"]
    assert cached["repair_trace"] == audit["repair_trace"]
    assert cached["assessment"] == audit["assessment"]


def test_audit_repeated_invalid_citations_stop_after_one_repair_with_cost_trace(case, tmp_path):
    calls = []

    def bad_citations(verdicts, payload):
        verdicts["associations"][0]["source_lines"] = ["L999"]

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(case[2], calls, grounding_transform=bad_citations),
        model=DEFAULT_MODEL,
    )
    with pytest.raises(ValueError, match="after one repair") as caught:
        reviewer.audit_grounding(case[1], {**compile_case(case), "raw": deepcopy(case[2])})
    assert len(calls) == caught.value.model_calls == 2
    assert caught.value.usage["total_tokens"] == 60
    assert len(caught.value.repair_trace) == 2
    assert all(step["validation_error"] for step in caught.value.repair_trace)
    assert not list(reviewer.cache_dir.glob("*.json"))


def test_audit_repairs_a_missing_typed_verdict_without_coercing_schema(case, tmp_path):
    import json

    packet = source_packet(case[0], "A", Requirements(must_have=[["python"]]))
    response = deepcopy(case[2])
    response["findings"][0].update(
        status="uncertain",
        relationship="none",
        evidence_depth="not_found",
        finding="The earlier assessment left the requested skill unresolved.",
        duration_basis="none",
        duration_associations=[],
    )
    calls = []
    base = fake_client(response, calls)

    def parse(**kwargs):
        result = base.responses.parse(**kwargs)
        if len(calls) == 1:
            invalid = result.output_parsed.model_dump(mode="json")
            invalid["criteria"][0].pop("verdict")
            result.output_parsed = SimpleNamespace(model_dump=lambda **_: invalid)
        return result

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),
        model=DEFAULT_MODEL,
    )
    audit = reviewer.audit_grounding(packet, {**compile_fixture(response, packet), "raw": response})
    assert len(calls) == 2
    sent = json.loads(calls[1]["input"])["structural_repair"]
    assert "verdict" not in sent["invalid_response"]["criteria"][0]
    assert "verdict" in sent["validation_error"]
    assert audit["criteria"][0]["verdict"]["assessed_status"] == "uncertain"
    assert audit["assessment"]["criteria"][0]["status"] == "uncertain"


@pytest.mark.parametrize(
    "corruption", ["response_integrity", "invalid_cached_source", "trace_integrity"]
)
def test_audit_cache_corruption_never_invokes_structural_repair(case, tmp_path, corruption):
    import json

    from screening_agent.policy import fingerprint

    calls = []

    def bad_once(verdicts, payload):
        if len(calls) == 1:
            verdicts["associations"][0]["source_lines"] = ["L999"]

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(case[2], calls, grounding_transform=bad_once),
        model=DEFAULT_MODEL,
    )
    assessment = {**compile_case(case), "raw": deepcopy(case[2])}
    reviewer.audit_grounding(case[1], assessment)
    path = next(reviewer.cache_dir.glob("*.json"))
    saved = json.loads(path.read_text())
    if corruption == "trace_integrity":
        saved["repair_trace"][0]["validation_error"] = "An altered historic failure."
    else:
        saved["response"]["associations"][0]["source_lines"] = ["L999"]
        if corruption == "invalid_cached_source":
            saved["integrity"] = fingerprint(saved["response"])
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        reviewer.audit_grounding(case[1], assessment)
    assert len(calls) == 2  # No new calls even if the cached checksum is recomputed.


def test_audit_provider_failure_after_structural_retry_retains_consumed_usage(case, tmp_path):
    calls = []

    def bad_citations(verdicts, payload):
        verdicts["associations"][0]["source_lines"] = ["L999"]

    base = fake_client(case[2], calls, grounding_transform=bad_citations)

    def parse(**kwargs):
        if calls:
            calls.append(kwargs)
            raise ConnectionError("controlled provider outage")
        return base.responses.parse(**kwargs)

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),
        model=DEFAULT_MODEL,
    )
    with pytest.raises(RuntimeError, match="ConnectionError") as caught:
        reviewer.audit_grounding(case[1], {**compile_case(case), "raw": deepcopy(case[2])})
    assert caught.value.model_calls == len(calls) == 2
    assert caught.value.usage["total_tokens"] == 30  # No fabricated usage for the failed call.
    assert len(caught.value.repair_trace) == 1
    assert not list(reviewer.cache_dir.glob("*.json"))


def dated_usage_case(case, *, start="2021-02", end="2021-03"):
    index = case[0]
    document = index.documents["A"]
    document["text"] = (
        "Anonymous profile\nDeveloper | 2020-01 to 2023-12\n"
        f"Used Python from {start} to {end} for a migration; other duties used Java.\n"
    )
    document["sha256"] = sha256(document["text"].encode()).hexdigest()
    packet = source_packet(index, "A", Requirements(skill_years={"python": 1}))
    raw = deepcopy(case[2])
    raw["work_units"] = [
        {
            "unit_id": "role",
            "scope": "employment",
            "title": "Developer",
            "start_date": "2020-01",
            "end_date": "2023-12",
            "source_lines": ["L2", "L3"],
        }
    ]
    raw["findings"][0].update(
        unit_ids=["role"],
        source_lines=["L2", "L3"],
        duration_basis="explicit_usage",
        duration_associations=[
            {
                "unit_id": "role",
                "basis": "explicit_usage",
                "source_lines": ["L2", "L3"],
                "assertion": f"The role explicitly limits Python use to {start} through {end}.",
                "temporal_scope": "dated_usage",
                "usage_start_date": start,
                "usage_end_date": end,
            }
        ],
    )
    return packet, raw


def test_dated_usage_counts_only_the_source_subinterval_and_projects_both_date_scopes(
    case, tmp_path
):
    import json

    packet, raw = dated_usage_case(case)
    result = compile_fixture(raw, packet)
    row = result["criteria"][0]
    assert row["duration"]["months"] == row["duration"]["maximum_months"] == 2
    assert row["status"] == "not_demonstrated"
    assert "48" not in row["finding"]
    calls = []
    reviewer = ContextualReviewer(
        root=tmp_path, client=fake_client(raw, calls), model=DEFAULT_MODEL
    )
    audit = reviewer.audit_grounding(packet, {**result, "raw": raw})
    target = json.loads(calls[0]["input"])["targets"]["associations"][0]
    assert target["source_dates"] == {"start": "2021-02", "end": "2021-03", "end_kind": "calendar"}
    assert target["employment_dates"] == {
        "start": "2020-01",
        "end": "2023-12",
        "end_kind": "calendar",
    }
    assert target["temporal_scope"] == "dated_usage"
    assert audit["assessment"]["criteria"][0]["duration"]["months"] == 2


def test_dated_usage_must_be_contained_by_its_source_employment_interval(case):
    packet, raw = dated_usage_case(case, start="2019-02", end="2019-03")
    with pytest.raises(ValueError, match="outside its employment context"):
        compile_fixture(raw, packet)


def test_quantity_scope_cannot_credit_the_whole_role_without_a_usage_claim(case):
    packet, raw = dated_usage_case(case)
    raw["findings"][0]["duration_associations"][0].update(
        temporal_scope="quantity_within_role", usage_start_date=None, usage_end_date=None
    )
    with pytest.raises(ValueError, match="requires its source usage claim"):
        compile_fixture(raw, packet)


def endpoint_case(jobs, *, minimum_years=3):
    text = "Anonymous profile\n" + "".join(
        job["heading"] + "\nMaintained documented services.\n" for job in jobs
    )
    candidate = Candidate(
        candidate_id="A", name="Anonymous profile", source_path="A.txt", skills=[]
    )
    index = SimpleNamespace(
        candidates={"A": candidate},
        documents={
            "A": {"text": text, "source_path": "A.txt", "sha256": sha256(text.encode()).hexdigest()}
        },
        manifest={"snapshot_date": "2026-09-01"},
        engine_fingerprint="test",
        _check_sources=lambda *_: None,
    )
    packet = source_packet(index, "A", Requirements(minimum_years=minimum_years))
    units = [
        {
            "unit_id": f"job{i}",
            "scope": "employment",
            "title": f"Service role {i}",
            "source_lines": [f"L{2 + 2 * i}", f"L{3 + 2 * i}"],
            "start_date": job["start"],
            "end_date": job["end"],
            "end_kind": job["kind"],
        }
        for i, job in enumerate(jobs)
    ]
    raw = {
        "work_units": units,
        "findings": [
            {
                "criterion_id": packet["semantic_criteria"][0]["criterion_id"],
                "status": "supported",
                "evidence_depth": "dated_association",
                "relationship": "direct",
                "finding": "The source describes these employment positions and their available date context.",
                "unit_ids": [unit["unit_id"] for unit in units],
                "source_lines": [line for unit in units for line in unit["source_lines"]],
                "usage_claims": [],
                "duration_basis": "total_employment",
                "duration_associations": [
                    {
                        "unit_id": unit["unit_id"],
                        "basis": "total_employment",
                        "temporal_scope": "whole_role",
                        "assertion": "The source presents this role as the candidate's employment.",
                        "source_lines": unit["source_lines"],
                    }
                    for unit in units
                ],
            }
        ],
        "observations": [],
    }
    return index, packet, raw


@pytest.mark.parametrize("token", ["Present", "to date", "continuing in this role"])
def test_explicit_ongoing_kind_uses_source_token_and_snapshot_without_phrase_whitelist(token):
    item = endpoint_case(
        [
            {
                "heading": f"Service role | 2023-04 to {token}",
                "start": "2023-04",
                "end": token,
                "kind": "ongoing",
            }
        ]
    )
    row = compile_case(item)["criteria"][0]
    assert row["duration"]["months"] == row["duration"]["maximum_months"] == 41
    assert row["duration"]["measurable_dates_complete"] is True
    assert compile_case(item)["evidence_graph"]["work_units"][0]["end_date"] == token


@pytest.mark.parametrize(
    "kind,end", [("ongoing", None), ("calendar", None), ("unknown", "Present")]
)
def test_endpoint_kind_and_verbatim_value_must_be_structurally_consistent(kind, end):
    item = endpoint_case(
        [
            {
                "heading": "Service role | 2020-01 to Present",
                "start": "2020-01",
                "end": end,
                "kind": kind,
            }
        ]
    )
    with pytest.raises(ValueError):
        compile_case(item)


def test_endpoint_kind_is_required_and_ongoing_anchor_must_belong_to_unit():
    from screening_agent.contextual import WorkUnit

    with pytest.raises(ValueError, match="end_kind"):
        WorkUnit(
            unit_id="role",
            scope="employment",
            title="Service role",
            source_lines=["L2"],
            start_date="2020-01",
            end_date=None,
        )
    item = endpoint_case(
        [
            {
                "heading": "Service role | 2023-04 to 2024-03",
                "start": "2023-04",
                "end": "Present",
                "kind": "ongoing",
            }
        ]
    )
    with pytest.raises(ValueError):
        compile_case(item)


@pytest.mark.parametrize(
    "start,end,kind,heading",
    [
        ("2022-01", None, "unknown", "Service role | started 2022-01; end not recorded"),
        (None, None, "unknown", "Service role | dates unavailable"),
        (None, "Present", "ongoing", "Service role | employed through Present; start unavailable"),
    ],
)
@pytest.mark.parametrize("minimum_years,expected_status", [(1, "supported"), (3, "uncertain")])
def test_unknown_dates_preserve_known_lower_bound_without_a_closed_total_or_pointless_repair(
    tmp_path, start, end, kind, heading, minimum_years, expected_status
):
    item = endpoint_case(
        [
            {
                "heading": "Service role | 2020-01 to 2021-12",
                "start": "2020-01",
                "end": "2021-12",
                "kind": "calendar",
            },
            {"heading": heading, "start": start, "end": end, "kind": kind},
        ],
        minimum_years=minimum_years,
    )
    index, packet, raw = item
    calls = []
    reviewer = ContextualReviewer(
        root=tmp_path, client=fake_client(raw, calls), model=DEFAULT_MODEL
    )
    graph = reviewer.review_graph.invoke({"packet": packet})
    row = graph["final"]["criteria"][0]
    assert row["duration"]["minimum_months"] == 24 and row["duration"]["maximum_months"] is None
    assert row["duration"]["measurable_dates_complete"] is False
    assert row["duration"]["extraction_complete"] is True
    assert len(row["duration"]["unmeasurable_unit_ids"]) == 1
    assert row["duration"]["unmeasurable_unit_ids"][0] == next(
        u["unit_id"]
        for u in graph["final"]["raw"]["work_units"]
        if u["source_lines"] == ["L4", "L5"]
    )
    assert row["status"] == expected_status and "At least 24 months" in row["finding"]
    assert len(graph["final"]["raw"]["findings"][0]["duration_associations"]) == 2
    assert graph["steps"] == [
        "extract_and_audit_work_history",
        "assess_requirements",
        "audit_evidence",
    ]
    assert len(calls) == 4  # Source inventory, inventory audit, assessment and evidence audit.


def test_only_unmeasurable_positive_employment_stays_unknown_instead_of_zero():
    item = endpoint_case(
        [
            {
                "heading": "Service role | dates unavailable",
                "start": None,
                "end": None,
                "kind": "unknown",
            }
        ]
    )
    row = compile_case(item)["criteria"][0]
    assert row["duration"]["months"] is None and row["duration"]["maximum_months"] is None
    assert row["duration"]["measurable_dates_complete"] is False and row["status"] == "uncertain"
    assert row["duration"]["unmeasurable_unit_ids"] == ["job0"]


def test_calendar_endpoint_remains_inclusive_while_ongoing_uses_snapshot_boundary():
    item = endpoint_case(
        [
            {
                "heading": "Service role | 2024-01 to 2024-03",
                "start": "2024-01",
                "end": "2024-03",
                "kind": "calendar",
            }
        ]
    )
    assert compile_case(item)["criteria"][0]["duration"]["months"] == 3


@pytest.mark.parametrize(
    "start,end,expected_lower",
    [("2020", "2023", 24), ("2020", "2023-12", 36), ("2020-01", "2023", 36)],
)
def test_year_only_precision_is_a_lower_bound_even_when_extraction_is_complete(
    tmp_path, start, end, expected_lower
):
    item = endpoint_case(
        [
            {
                "heading": f"Service role | {start} to {end}",
                "start": start,
                "end": end,
                "kind": "calendar",
            }
        ],
        minimum_years=1,
    )
    _, packet, raw = item
    calls = []
    reviewer = ContextualReviewer(
        root=tmp_path, client=fake_client(raw, calls), model=DEFAULT_MODEL
    )
    graph = reviewer.review_graph.invoke({"packet": packet})
    duration = graph["final"]["criteria"][0]["duration"]
    assert duration["minimum_months"] == expected_lower and duration["maximum_months"] is None
    assert (
        duration["measurable_dates_complete"] is False and duration["extraction_complete"] is True
    )
    assert len(duration["partial_date_unit_ids"]) == 1
    assert duration["partial_date_unit_ids"][0] in duration["unit_ids"]
    assert graph["final"]["criteria"][0]["status"] == "supported"
    assert len(calls) == 4 and graph["steps"][-1] == "audit_evidence"


def test_rejected_temporal_extraction_cannot_credit_an_otherwise_true_employment_link(tmp_path):
    item = endpoint_case(
        [
            {
                "heading": "Service role | 2020-01 to 2021-12",
                "start": "2020-01",
                "end": "2021-12",
                "kind": "calendar",
            },
            {
                "heading": "Service role | 2022-01 to 2022-12",
                "start": "2022-01",
                "end": "2022-12",
                "kind": "calendar",
            },
        ]
    )
    _, packet, raw = item

    def reject_endpoint(verdicts, payload):
        verdicts["endpoints"][1].update(
            extraction_complete=False,
            reason="The proposed endpoint does not faithfully represent the role's source timing.",
        )

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=fake_client(raw, [], grounding_transform=reject_endpoint),
        model=DEFAULT_MODEL,
    )
    proposed = {**compile_case(item), "raw": raw}
    assert proposed["criteria"][0]["duration"]["months"] == 36
    audited = reviewer.audit_grounding(packet, proposed)
    duration = audited["assessment"]["criteria"][0]["duration"]
    assert duration["months"] == 24 and duration["maximum_months"] is None
    assert (
        duration["extraction_complete"] is False and duration["measurable_dates_complete"] is False
    )
    assert len(audited["assessment"]["raw"]["findings"][0]["duration_associations"]) == 2
    assert audited["associations"][1]["entailed"] is True
    assert audited["criteria"][0]["inventory_entailed"] is True
    assert audited["criteria"][0]["entailed"] is False
    assert (
        audited["proposed_assessment"]["findings"][0]["duration_associations"][1]["temporal_scope"]
        == "whole_role"
    )


def test_endpoint_audit_cannot_omit_a_positive_employment_target(tmp_path):
    item = endpoint_case(
        [
            {
                "heading": "Service role | 2020-01 to 2021-12",
                "start": "2020-01",
                "end": "2021-12",
                "kind": "calendar",
            }
        ]
    )
    _, packet, raw = item
    calls = []

    def omit(verdicts, payload):
        verdicts["endpoints"] = []

    reviewer = ContextualReviewer(
        root=tmp_path, client=fake_client(raw, calls, grounding_transform=omit), model=DEFAULT_MODEL
    )
    with pytest.raises(ValueError, match="endpoints target"):
        reviewer.audit_grounding(packet, {**compile_case(item), "raw": raw})
    assert len(calls) == 2  # One structural repair, never silently accept missing endpoint checks.


def test_inconsistent_ongoing_endpoint_uses_the_existing_structural_repair(tmp_path):
    import json

    item = endpoint_case(
        [
            {
                "heading": "Service role | 2023-04 to Present",
                "start": "2023-04",
                "end": None,
                "kind": "ongoing",
            }
        ]
    )
    _, packet, raw = item
    calls = []
    original = fake_client(raw, calls)

    def parse(**kwargs):
        if calls:
            raw["work_units"][0]["end_date"] = "Present"
        return original.responses.parse(**kwargs)

    reviewer = ContextualReviewer(
        root=tmp_path,
        client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),
        model=DEFAULT_MODEL,
    )
    result = reviewer.assess(packet)
    repair = json.loads(calls[1]["input"])
    assert len(calls) == 4 and "verbatim source token" in repair["validation_error"]
    assert repair["previous_response"]["work_units"][0]["end_date"] is None
    assert repair["previous_response"]["work_units"][0]["end_kind"] == "ongoing"
    assert result["criteria"][0]["duration"]["months"] == 41
    assert result["usage"]["total_tokens"] == 30


def explicit_usage_source(
    parent_start, parent_end, parent_kind, usage_start, usage_end, usage_kind
):
    display_parent = parent_end or "endpoint unavailable"
    display_usage = usage_end or "endpoint unavailable"
    item = endpoint_case(
        [
            {
                "heading": f"Developer | {parent_start} to {display_parent}",
                "start": parent_start,
                "end": parent_end,
                "kind": parent_kind,
            }
        ]
    )
    index, _, raw = item
    doc = index.documents["A"]
    doc["text"] = doc["text"].replace(
        "Maintained documented services.",
        f"Used Python from {usage_start} to {display_usage} for the migration.",
    )
    doc["sha256"] = sha256(doc["text"].encode()).hexdigest()
    packet = source_packet(index, "A", Requirements(skill_years={"python": 1}))
    finding = raw["findings"][0]
    finding.update(
        criterion_id=packet["semantic_criteria"][0]["criterion_id"], duration_basis="explicit_usage"
    )
    finding["duration_associations"][0].update(
        basis="explicit_usage",
        temporal_scope="dated_usage",
        usage_start_date=usage_start,
        usage_end_date=usage_end,
        usage_end_kind=usage_kind,
        assertion="This employment role explicitly states the source-scoped Python usage period.",
    )
    return index, packet, raw


@pytest.mark.parametrize("parent_start,parent_end", [("2020", "2023"), ("2020", "2020")])
def test_precise_usage_can_narrow_a_year_only_employment_range(parent_start, parent_end):
    item = explicit_usage_source(
        parent_start, parent_end, "calendar", "2020-06", "2020-07", "calendar"
    )
    row = compile_case(item)["criteria"][0]
    assert row["duration"]["months"] == row["duration"]["maximum_months"] == 2
    assert row["duration"]["measurable_dates_complete"] is True
    assert row["duration"]["partial_date_unit_ids"] == []
    assert row["status"] == "not_demonstrated"


@pytest.mark.parametrize("usage_token", ["to date", "still continuing", "Present"])
def test_usage_ongoing_kind_is_independent_of_the_parent_token(usage_token, tmp_path):
    import json

    item = explicit_usage_source("2020-01", "Present", "ongoing", "2022-01", usage_token, "ongoing")
    _, packet, raw = item
    compiled = compile_case(item)
    assert compiled["criteria"][0]["duration"]["months"] == 56
    calls = []
    reviewer = ContextualReviewer(
        root=tmp_path, client=fake_client(raw, calls), model=DEFAULT_MODEL
    )
    audit = reviewer.audit_grounding(packet, {**compiled, "raw": raw})
    targets = json.loads(calls[0]["input"])["targets"]
    assert targets["associations"][0]["source_dates"] == {
        "start": "2022-01",
        "end": usage_token,
        "end_kind": "ongoing",
    }
    assert targets["associations"][0]["employment_dates"]["end"] == "Present"
    assert [target["check_scope"] for target in targets["endpoints"]] == [
        "employment_endpoints",
        "usage_endpoints",
    ]
    assert audit["assessment"]["criteria"][0]["duration"]["months"] == 56


@pytest.mark.parametrize("start,end", [("2020-06-01", "2020-06-30"), ("2020-06-20", "2020-07-31")])
def test_precise_usage_outside_known_employment_days_is_not_hidden_by_month_rounding(start, end):
    item = explicit_usage_source("2020-06-20", "2020-07-10", "calendar", start, end, "calendar")
    with pytest.raises(ValueError, match="outside its employment context"):
        compile_case(item)


def test_unknown_employment_days_do_not_impose_fabricated_day_precision():
    item = explicit_usage_source(
        "2020-06", "2020-07", "calendar", "2020-06-20", "2020-07-10", "calendar"
    )
    assert compile_case(item)["criteria"][0]["duration"]["months"] == 2


def test_known_usage_days_inside_known_employment_remain_calendar_coverage():
    item = explicit_usage_source(
        "2020-06-15", "2020-07-20", "calendar", "2020-06-20", "2020-07-10", "calendar"
    )
    assert compile_case(item)["criteria"][0]["duration"]["months"] == 2


def test_unknown_usage_endpoint_does_not_inherit_an_ongoing_job_endpoint(tmp_path):
    item = explicit_usage_source("2020-01", "Present", "ongoing", "2022-01", None, "unknown")
    _, packet, raw = item
    calls = []
    reviewer = ContextualReviewer(
        root=tmp_path, client=fake_client(raw, calls), model=DEFAULT_MODEL
    )
    graph = reviewer.review_graph.invoke({"packet": packet})
    row = graph["final"]["criteria"][0]
    assert row["duration"]["months"] is None and row["duration"]["maximum_months"] is None
    assert row["duration"]["extraction_complete"] is True
    assert graph["steps"][-1] == "audit_evidence" and len(calls) == 4


@pytest.mark.parametrize(
    "parent_start,parent_end,parent_kind",
    [(None, None, "unknown"), ("2020-01", None, "unknown"), (None, "2023-12", "calendar")],
)
def test_independently_dated_usage_can_be_credited_without_inventing_unknown_job_dates(
    parent_start, parent_end, parent_kind, tmp_path
):
    item = explicit_usage_source(
        parent_start, parent_end, parent_kind, "2021-06", "2021-07", "calendar"
    )
    _, packet, raw = item
    calls = []
    reviewer = ContextualReviewer(
        root=tmp_path, client=fake_client(raw, calls), model=DEFAULT_MODEL
    )
    audit = reviewer.audit_grounding(packet, {**compile_case(item), "raw": raw})
    row = audit["assessment"]["criteria"][0]
    assert row["duration"]["months"] == row["duration"]["maximum_months"] == 2
    assert row["duration"]["measurable_dates_complete"] is True
    unit = audit["assessment"]["evidence_graph"]["work_units"][0]
    assert unit["start_date"] == parent_start and unit["end_date"] == parent_end
    assert unit["end_kind"] == parent_kind
    assert unit["interval"] is None
    assert row["duration_associations"][0]["usage_start_date"] == "2021-06"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "start,end,kind", [("2022-01", None, "unknown"), (None, "2020-12", "calendar")]
)
def test_missing_one_job_endpoint_does_not_override_a_contradicted_known_endpoint(start, end, kind):
    item = explicit_usage_source(start, end, kind, "2021-06", "2021-07", "calendar")
    with pytest.raises(ValueError, match="outside its employment context"):
        compile_case(item)


def scripted_source_client(responses, calls):
    """Return provider-like results or errors without inferring unavailable usage."""
    sequence = iter(responses)

    def parse(**kwargs):
        calls.append(kwargs)
        response = next(sequence)
        if isinstance(response, Exception):
            raise response
        return response

    return SimpleNamespace(responses=SimpleNamespace(parse=parse))


def provider_result(parsed, usage=True):
    return SimpleNamespace(
        output_parsed=parsed,
        usage=SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30)
        if usage
        else None,
        model=DEFAULT_MODEL,
    )


def observation(claim="Built an ETL pipeline using Python and PostgreSQL.", lines=None):
    return {"kind": "contribution", "claim": claim, "source_lines": lines or ["L7", "L8"]}


def invalid_assessment(case):
    raw = deepcopy(case[2])
    bind_fixture_ids(raw, case[1]["semantic_criteria"])
    raw["findings"][0]["source_lines"] = ["L999"]
    return ContextualResponse.model_validate(raw)


def test_assessment_two_invalid_responses_preserve_both_usage_and_failed_trace(case, tmp_path):
    invalid = invalid_assessment(case)
    calls = []
    reviewer = ContextualReviewer(
        root=tmp_path,
        model=DEFAULT_MODEL,
        client=scripted_source_client([provider_result(invalid), provider_result(invalid)], calls),
    )
    with pytest.raises(ValueError, match="after one structural repair") as caught:
        reviewer.assess(case[1])
    error = caught.value
    assert error.model_calls == len(calls) == 2
    assert error.usage == {"input_tokens": 20, "output_tokens": 40, "total_tokens": 60}
    assert error.usage_unknown_calls == 0 and len(error.repair_trace) == 2
    assert all(entry["validation_error"] for entry in error.repair_trace)
    assert not list(reviewer.cache_dir.glob("*.json"))


def test_assessment_later_provider_failure_retains_known_cost_without_exposing_error_text(
    case, tmp_path
):
    calls = []
    reviewer = ContextualReviewer(
        root=tmp_path,
        model=DEFAULT_MODEL,
        client=scripted_source_client(
            [
                provider_result(invalid_assessment(case)),
                RuntimeError("provider secret token material"),
            ],
            calls,
        ),
    )
    with pytest.raises(RuntimeError, match="preceding review was preserved") as caught:
        reviewer.assess(case[1])
    error = caught.value
    assert error.model_calls == 2 and error.usage_unknown_calls == 1
    assert error.usage == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    assert len(error.repair_trace) == 1
    assert "secret" not in str(error) and error.__suppress_context__
    assert not list(reviewer.cache_dir.glob("*.json"))


def test_empty_parsed_response_retains_reported_usage(case, tmp_path):
    reviewer = ContextualReviewer(
        root=tmp_path,
        model=DEFAULT_MODEL,
        client=scripted_source_client([provider_result(None)], []),
    )
    with pytest.raises(ValueError, match="no complete assessment") as caught:
        reviewer.assess(case[1])
    assert caught.value.usage["total_tokens"] == 30 and caught.value.model_calls == 1
    assert caught.value.usage_unknown_calls == 0


def test_generated_references_are_bound_to_current_source(case, tmp_path):
    from pydantic import ValidationError

    calls = []
    reviewer = ContextualReviewer(
        root=tmp_path, client=fake_client(case[2], calls), model=DEFAULT_MODEL
    )
    result = reviewer.review_graph.invoke({"packet": case[1]})
    audit_schema = calls[1]["text_format"]
    with pytest.raises(ValidationError):
        audit_schema.model_validate(
            dict(
                complete=False,
                employment_complete=False,
                reason="The inventory requires another source check.",
                source_lines=["L1: explanation appended to an ID"],
            )
        )
    assessment_schema = calls[2]["text_format"]
    raw = {k: deepcopy(result["first"]["raw"][k]) for k in ("findings", "observations")}
    assessment_schema.model_validate(raw)
    raw["findings"][0]["unit_ids"] = ["invented-unit"]
    with pytest.raises(ValidationError):
        assessment_schema.model_validate(raw)
