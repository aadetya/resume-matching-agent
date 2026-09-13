"""Stable criterion identity and one request-specific, cached evidence contract."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace

import pytest

from screening_agent.contracts import Requirements
from screening_agent.criteria import RequirementCompiler, StandardSet, build_criteria, default_scope
from screening_agent.model_config import DEFAULT_MODEL
from screening_agent.policy import fingerprint, requirements_fingerprint


def request():
    return Requirements(
        role="software_developer",
        must_have=[["python", "java"], ["sql"]],
        minimum_years=3,
        skill_years={"sql": 2},
        nice_to_have=["react"],
        semantic_brief="Software development with Python or Java, SQL experience, and optional React.",
    )


def standard_payload(criteria):
    def standard(spec):
        kind = spec["kind"]
        row = {
            "criterion_id": spec["criterion_id"],
            "capability": " or ".join(spec["alternatives"]) or "Total employment",
            "requested_depth": "employment"
            if kind in {"total_duration", "skill_duration"}
            else "role"
            if kind == "role"
            else "presence",
            "experience_basis": "total_employment"
            if kind == "total_duration"
            else "skill_related_employment"
            if kind == "skill_duration"
            else "not_applicable",
            "sufficient_evidence": "Relevant dated employment supports duration; explicit source claims support requested presence.",
            "equivalence_boundary": "Equivalent work must demonstrate the requested capability; adjacent technologies alone do not suffice.",
            "uncertainty_boundary": "Missing or conflicting source facts remain uncertain and are shown for review.",
        }

        row["evidence_scope"] = default_scope(kind, row["requested_depth"], row["experience_basis"])
        if "meaning" in spec:
            row.update(
                {
                    k: spec["meaning"][k]
                    for k in ("requested_depth", "experience_basis", "evidence_scope")
                }
            )
        return row

    return {"standards": [standard(spec) for spec in criteria]}


def fake_client(calls, *, transform=None, no_output=False, error=None):
    def parse(**kwargs):
        calls.append(kwargs)
        assert issubclass(kwargs["text_format"], StandardSet)
        if error:
            raise error
        payload = json.loads(kwargs["input"])
        raw = standard_payload(payload["criteria"])
        if transform:
            transform(raw)
            # The application must validate a malformed parsed-provider object too.
            parsed = SimpleNamespace(model_dump=lambda **_: raw)
        else:
            parsed = StandardSet.model_validate(raw)
        return SimpleNamespace(
            output_parsed=None if no_output else parsed,
            usage=SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18),
            model=kwargs["model"],
        )

    return SimpleNamespace(responses=SimpleNamespace(parse=parse))


def keyed(specs):
    return {
        (spec["kind"], tuple(sorted(spec["alternatives"]))): spec["criterion_id"] for spec in specs
    }


def test_criterion_ids_are_stable_under_list_and_or_alternative_reordering():
    initial = request()
    reordered = initial.model_copy(
        update={
            "must_have": [list(reversed(group)) for group in reversed(initial.must_have)],
            "nice_to_have": list(reversed(initial.nice_to_have)),
        }
    )
    before, after = build_criteria(initial), build_criteria(reordered)
    assert keyed(before) == keyed(after)
    assert all(re.fullmatch(r"c_[0-9a-f]{16}", row["criterion_id"]) for row in before)
    assert len({row["criterion_id"] for row in before}) == len(before)
    assert requirements_fingerprint(initial) == requirements_fingerprint(reordered)


def test_changed_threshold_preserves_criterion_identity_but_invalidates_requirement_identity():
    original = request()
    amended = original.model_copy(update={"minimum_years": 5, "skill_years": {"sql": 4}})
    assert keyed(build_criteria(original)) == keyed(build_criteria(amended))
    assert requirements_fingerprint(original) != requirements_fingerprint(amended)
    durations = {
        row["kind"]: row["minimum_years"]
        for row in build_criteria(amended)
        if "minimum_years" in row
    }
    assert durations == {"total_duration": 5, "skill_duration": 4}


def test_different_meanings_have_distinct_ids_even_with_same_skill_name():
    specs = build_criteria(Requirements(must_have=[["sql"]], skill_years={"sql": 2}))
    assert len(specs) == 2
    assert specs[0]["criterion_id"] != specs[1]["criterion_id"]
    combined = build_criteria(Requirements(must_have=[["python", "java"]]))
    separate = build_criteria(Requirements(must_have=[["python"], ["java"]]))
    assert not {row["criterion_id"] for row in combined} & {row["criterion_id"] for row in separate}


def test_duplicate_semantic_criterion_is_rejected_instead_of_overwriting_scores():
    with pytest.raises(ValueError, match="more than once"):
        build_criteria(Requirements(must_have=[["python", "java"], ["java", "python"]]))


def test_compilation_is_cached_durably_with_honest_usage_metadata(tmp_path):
    calls = []
    compiler = RequirementCompiler(tmp_path, client=fake_client(calls), model=DEFAULT_MODEL)
    req = request()
    compiled, metadata = compiler.compile_with_metadata(req)
    again, reused = compiler.compile_with_metadata(req)
    assert len(calls) == 1
    assert compiled.model_dump() == again.model_dump()
    assert not metadata["cache_hit"] and metadata["usage"] == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }
    assert reused["cache_hit"] and reused["usage"] == {}
    assert reused["cached_usage"] == metadata["usage"]
    assert metadata["compiler_signature"] == reused["compiler_signature"] == compiler.signature
    assert compiled.standards_signature == metadata["standards_signature"]
    assert req.evidence_standards == [] and req.standards_signature == ""
    assert {standard.criterion_id for standard in compiled.evidence_standards} == {
        row["criterion_id"] for row in build_criteria(req)
    }
    assert all("standard" in row for row in build_criteria(compiled))
    restarted = RequirementCompiler(
        tmp_path,
        client=fake_client([], error=AssertionError("Cache hit called provider")),
        model=DEFAULT_MODEL,
    )
    assert restarted.compile(req).model_dump() == compiled.model_dump()


def test_compiler_receives_only_current_request_and_no_candidate_material(tmp_path):
    calls = []
    compiler = RequirementCompiler(tmp_path, client=fake_client(calls), model=DEFAULT_MODEL)
    req = request().model_copy(
        update={"source_text": "Untrusted historical source marker CANDIDATE_927"}
    )
    compiler.compile(req)
    payload = json.loads(calls[0]["input"])
    assert set(payload) == {"criteria", "semantic_brief"}
    assert payload["semantic_brief"] == req.semantic_brief
    assert "CANDIDATE_927" not in calls[0]["input"]
    assert calls[0]["model"] == DEFAULT_MODEL == "gpt-5.6-luna"
    assert calls[0]["store"] is False


@pytest.mark.parametrize("change", ["meaning", "threshold", "model"])
def test_changed_request_or_model_does_not_reuse_previous_standard_cache(tmp_path, change):
    calls = []
    compiler = RequirementCompiler(tmp_path, client=fake_client(calls), model=DEFAULT_MODEL)
    req = request()
    original = compiler.compile(req)
    if change == "meaning":
        req = req.model_copy(
            update={
                "semantic_brief": req.semantic_brief
                + " Demonstrated personal application is required."
            }
        )
    elif change == "threshold":
        req = req.model_copy(update={"minimum_years": 8})
    else:
        compiler = RequirementCompiler(
            tmp_path, client=fake_client(calls), model="different-test-model"
        )
    amended, metadata = compiler.compile_with_metadata(req)
    assert len(calls) == 2 and not metadata["cache_hit"]
    assert amended.standards_signature != original.standards_signature
    assert requirements_fingerprint(amended) != requirements_fingerprint(original)


def test_mutating_a_frozen_request_without_explicit_refinement_fails_closed(tmp_path):
    calls = []
    compiler = RequirementCompiler(tmp_path, client=fake_client(calls), model=DEFAULT_MODEL)
    compiled = compiler.compile(request())
    changed = compiled.model_copy(update={"minimum_years": 9})
    with pytest.raises(ValueError, match="Frozen evidence standards changed"):
        compiler.compile(changed)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "unknown",
        "wrong_basis",
        "invalid_depth",
        "missing_field",
        "extra_field",
    ],
)
def test_malformed_or_incomplete_standards_are_rejected_without_caching(tmp_path, mutation):
    def transform(raw):
        standards = raw["standards"]
        if mutation == "missing":
            standards.pop()
        elif mutation == "duplicate":
            standards[1] = deepcopy(standards[0])
        elif mutation == "unknown":
            standards[0]["criterion_id"] = "c_0000000000000000"
        elif mutation == "reordered":
            standards.reverse()
        elif mutation == "wrong_basis":
            standards[0]["experience_basis"] = "explicit_usage"
        elif mutation == "invalid_depth":
            standards[0]["requested_depth"] = "invented_depth"
        elif mutation == "missing_field":
            del standards[0]["capability"]
        else:
            standards[0]["hidden_requirement"] = "Unrequested certification"

    calls = []
    compiler = RequirementCompiler(
        tmp_path, client=fake_client(calls, transform=transform), model=DEFAULT_MODEL
    )
    with pytest.raises(ValueError):
        compiler.compile(request())
    assert len(calls) == 1
    assert not list(compiler.cache_dir.glob("*.json"))


@pytest.mark.parametrize("no_output", [True, False])
def test_absent_output_and_provider_failure_do_not_create_partial_standards(tmp_path, no_output):
    calls = []
    compiler = RequirementCompiler(
        tmp_path,
        client=fake_client(
            calls,
            no_output=no_output,
            error=None if no_output else TimeoutError("Provider did not complete"),
        ),
        model=DEFAULT_MODEL,
    )
    with pytest.raises((ValueError, RuntimeError), match="no complete|compilation failed"):
        compiler.compile(request())
    assert not list(compiler.cache_dir.glob("*.json"))


@pytest.mark.parametrize("rehash", [False, True])
def test_cache_tampering_cannot_bypass_integrity_or_dimension_validation(tmp_path, rehash):
    calls = []
    compiler = RequirementCompiler(tmp_path, client=fake_client(calls), model=DEFAULT_MODEL)
    compiler.compile(request())
    path = next(compiler.cache_dir.glob("*.json"))
    saved = json.loads(path.read_text())
    saved["response"]["standards"][0]["experience_basis"] = "total_employment"
    if rehash:
        saved["integrity"] = fingerprint(saved["response"])
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="integrity|experience dimension"):
        compiler.compile(request())
    assert len(calls) == 1


def test_concurrent_compilation_on_one_session_reuses_one_complete_result(tmp_path):
    calls = []
    compiler = RequirementCompiler(tmp_path, client=fake_client(calls), model=DEFAULT_MODEL)
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(lambda _: compiler.compile_with_metadata(request()), range(8)))
    assert len(calls) == 1
    assert sum(not metadata["cache_hit"] for _, metadata in results) == 1
    assert len({compiled.standards_signature for compiled, _ in results}) == 1
    assert len(list(compiler.cache_dir.glob("*.json"))) == 1
    assert not list(compiler.cache_dir.glob("*.tmp"))


def clear_compilation(requirements, **changes):
    return requirements.model_copy(
        update={"evidence_standards": [], "standards_signature": "", **changes}, deep=True
    )


def test_threshold_and_unrelated_summary_refinement_preserve_every_standard_byte_for_byte(tmp_path):
    calls = []
    compiler = RequirementCompiler(tmp_path, client=fake_client(calls), model=DEFAULT_MODEL)
    previous = compiler.compile(request())
    original = [s.model_dump_json() for s in previous.evidence_standards]
    proposal = clear_compilation(
        previous,
        minimum_years=7,
        semantic_brief="A rephrased summary must not redefine unchanged criterion meaning.",
    )
    revised, metadata = compiler.compile_with_metadata(
        proposal, previous=previous, changed_criterion_ids=[]
    )
    assert len(calls) == 1 and metadata["compiled_criterion_ids"] == []
    assert len(metadata["preserved_criterion_ids"]) == len(original)
    assert [s.model_dump_json() for s in revised.evidence_standards] == original
    assert (
        revised.minimum_years == 7 and revised.standards_signature != previous.standards_signature
    )
    assert metadata["usage"] == {} and metadata["cache_hit"]
    compiler.compile(revised)
    assert len(calls) == 1


def test_only_explicit_meaning_change_is_compiled_and_examples_stay_separate(tmp_path):
    from screening_agent.contracts import CriterionMeaning, RequirementClause, RequirementSource

    calls = []
    compiler = RequirementCompiler(tmp_path, client=fake_client(calls), model=DEFAULT_MODEL)
    previous = compiler.compile(request())
    selected = next(
        spec
        for spec in build_criteria(previous)
        if spec["kind"] == "skill" and spec["alternatives"] == ["sql"]
    )
    cid = selected["criterion_id"]
    source = RequirementSource(
        source_id="clarification",
        kind="user_request",
        text="Accept practical relational database work; PostgreSQL and SQLAlchemy are examples, not an exclusive list.",
    )
    meaning = CriterionMeaning(
        criterion_id=cid,
        conditions=[
            RequirementClause(
                source_id=source.source_id, quote="Accept practical relational database work"
            )
        ],
        examples=[
            RequirementClause(
                source_id=source.source_id,
                quote="PostgreSQL and SQLAlchemy are examples, not an exclusive list",
            )
        ],
        assumptions=[],
        requested_depth="application",
        experience_basis="not_applicable",
        evidence_scope="personal_application",
    )
    proposal = clear_compilation(
        previous,
        requirement_sources=[*previous.requirement_sources, source],
        criterion_meanings=[
            meaning if m.criterion_id == cid else m for m in previous.criterion_meanings
        ],
        semantic_brief="An unrelated rewrite must not override the scoped clauses.",
    )
    revised, metadata = compiler.compile_with_metadata(
        proposal, previous=previous, changed_criterion_ids=[cid]
    )
    payload = json.loads(calls[-1]["input"])
    assert len(calls) == 2 and metadata["compiled_criterion_ids"] == [cid]
    assert len(payload["criteria"]) == 1 and payload["semantic_brief"] == ""
    assert payload["criteria"][0]["meaning"] == meaning.model_dump(exclude={"criterion_id"})
    assert "exclusive list" in payload["criteria"][0]["meaning"]["examples"][0]["quote"]
    before = {s.criterion_id: s.model_dump_json() for s in previous.evidence_standards}
    assert all(
        s.model_dump_json() == before[s.criterion_id]
        for s in revised.evidence_standards
        if s.criterion_id != cid
    )
    assert (
        next(s for s in revised.evidence_standards if s.criterion_id == cid).evidence_scope
        == "personal_application"
    )
    with pytest.raises(ValueError, match="without an explicit change declaration"):
        compiler.compile(proposal, previous=previous, changed_criterion_ids=[])


def test_threshold_change_cannot_smuggle_different_meaning_under_stable_identity(tmp_path):
    compiler = RequirementCompiler(tmp_path, client=fake_client([]), model=DEFAULT_MODEL)
    old = compiler.compile(request())
    altered = [
        m.model_copy(update={"assumptions": ["Only full-time work may count."]})
        if m.experience_basis == "total_employment"
        else m
        for m in old.criterion_meanings
    ]
    with pytest.raises(ValueError, match="unchanged criterion"):
        compiler.compile(
            clear_compilation(old, minimum_years=7, criterion_meanings=altered),
            previous=old,
            changed_criterion_ids=[],
        )


@pytest.mark.parametrize(
    "damage", ["invented_quote", "unknown_source", "condition_is_example", "wrong_scope"]
)
def test_provenance_and_scope_fail_before_any_candidate_or_compiler_call(tmp_path, damage):
    calls = []
    compiler = RequirementCompiler(tmp_path, client=fake_client(calls), model=DEFAULT_MODEL)
    original = compiler.compile(request())
    proposal = clear_compilation(original)
    meaning = proposal.criterion_meanings[0]
    if damage == "invented_quote":
        meaning.conditions[0].quote = "The user never supplied this exact instruction."
    elif damage == "unknown_source":
        meaning.conditions[0].source_id = "unknown"
    elif damage == "condition_is_example":
        meaning.examples = deepcopy(meaning.conditions)
    else:
        meaning.evidence_scope = "all_employment"
    with pytest.raises(ValueError, match="exact quotation|both.*condition|evidence scope"):
        compiler.compile(proposal, previous=original, changed_criterion_ids=[meaning.criterion_id])
    assert len(calls) == 1


def test_compiler_cannot_change_typed_depth_even_with_well_formed_response(tmp_path):
    calls = []
    first = RequirementCompiler(tmp_path, client=fake_client(calls), model=DEFAULT_MODEL).compile(
        request()
    )

    def alter(raw):
        row = next(s for s in raw["standards"] if s["requested_depth"] == "presence")
        row.update(requested_depth="application", evidence_scope="personal_application")

    compiler = RequirementCompiler(
        tmp_path / "new", client=fake_client([], transform=alter), model=DEFAULT_MODEL
    )
    with pytest.raises(ValueError, match="user-bound depth"):
        compiler.compile(clear_compilation(first))


def test_preference_status_change_reuses_same_semantic_standard(tmp_path):
    calls = []
    compiler = RequirementCompiler(tmp_path, client=fake_client(calls), model=DEFAULT_MODEL)
    old = compiler.compile(Requirements(must_have=[["python"]], nice_to_have=["rust"]))
    proposal = clear_compilation(old, must_have=[["python"], ["rust"]], nice_to_have=[])
    updated = compiler.compile(proposal, previous=old, changed_criterion_ids=[])
    assert len(calls) == 1
    assert [s.model_dump_json() for s in old.evidence_standards] == [
        s.model_dump_json() for s in updated.evidence_standards
    ]
    assert all(spec["mandatory"] for spec in build_criteria(updated))


def test_tool_refinement_passes_previous_standards_and_direct_search_does_not_recompile(tmp_path):
    from screening_agent.agent_tools import ToolRegistry
    from screening_agent.planner import SkillVocabulary

    calls = []
    planner = SimpleNamespace(
        client=fake_client(calls), model=DEFAULT_MODEL, vocabulary=SkillVocabulary(tmp_path)
    )
    index = SimpleNamespace(candidates={})
    registry = ToolRegistry(tmp_path, index, planner)
    created = registry.invoke(
        "extract_requirements",
        {"jd": "Find Python candidates with three years overall experience."},
        proposal=Requirements(must_have=[["python"]], minimum_years=3),
    )
    assert created["ok"] and len(calls) == 1
    previous = Requirements.model_validate(created["result"])
    revised = registry.invoke(
        "extract_requirements",
        {"jd": "Raise overall experience to five years."},
        proposal=clear_compilation(previous, minimum_years=5),
        requirements=previous,
        changed_criterion_ids=[],
    )
    assert revised["ok"] and len(calls) == 1
    assert revised["event"]["compiled_criterion_ids"] == []
    assert revised["result"]["evidence_standards"] == created["result"]["evidence_standards"]
    assert revised["result"]["minimum_years"] == 5

    def retrieve(current):
        assert current.standards_signature == revised["result"]["standards_signature"]
        return SimpleNamespace(
            model_dump=lambda **_: {"corpus_size": 0, "retrieved_count": 0, "backend": "controlled"}
        )

    index.retrieve = retrieve
    search = registry.invoke("rag_search", {"requirements": revised["result"]}, retrieval_only=True)
    assert search["ok"] and len(calls) == 1


def test_separate_compiler_instances_share_a_single_cache_writer(tmp_path):
    import threading

    calls = []
    start = threading.Barrier(4)
    delegate = fake_client(calls).responses.parse

    def parse(**kwargs):
        # Give the other independent instances time to reach the same cache miss.
        threading.Event().wait(0.06)
        return delegate(**kwargs)

    def compile_one(_):
        compiler = RequirementCompiler(
            tmp_path,
            client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),
            model=DEFAULT_MODEL,
        )
        start.wait(timeout=2)
        return compiler.compile_with_metadata(request())

    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(compile_one, range(4)))
    assert len(calls) == 1
    assert sum(not metadata["cache_hit"] for _, metadata in results) == 1
    assert len({compiled.standards_signature for compiled, _ in results}) == 1
    assert len(list((tmp_path / "artifacts/requirement_standards").glob("*.json"))) == 1


def test_compilation_lock_timeout_preserves_request_without_provider_call(tmp_path, monkeypatch):
    from filelock import Timeout

    from screening_agent import criteria as module

    class ContendedLock:
        def __init__(self, path, *, timeout):
            self.path = path
            assert timeout == 300

        def __enter__(self):
            raise Timeout(str(self.path))

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(module, "FileLock", ContendedLock)
    calls = []
    req = request()
    before = req.model_dump_json()
    compiler = RequirementCompiler(tmp_path, client=fake_client(calls), model=DEFAULT_MODEL)
    with pytest.raises(RuntimeError, match="preceding review was preserved"):
        compiler.compile(req)
    assert not calls and req.model_dump_json() == before
    assert not list(compiler.cache_dir.glob("*.json"))


def test_standard_list_order_is_canonicalized_by_exact_ids(tmp_path):
    calls = []
    compiler = RequirementCompiler(
        tmp_path,
        client=fake_client(calls, transform=lambda raw: raw["standards"].reverse()),
        model=DEFAULT_MODEL,
    )
    result = compiler.compile(request())
    assert [s.criterion_id for s in result.evidence_standards] == [
        c["criterion_id"] for c in build_criteria(request())
    ]
    schema = calls[0]["text_format"].model_json_schema()
    assert set(schema["$defs"]["CurrentCriterionID"]["enum"]) == {
        c["criterion_id"] for c in build_criteria(request())
    }
