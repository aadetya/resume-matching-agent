"""Criterion IDs must connect expansion, neural evidence and final relevance scores."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from screening_agent.contracts import EvidenceStandard, Requirements
from screening_agent.criteria import build_criteria
from screening_agent.retrieval import ResumeIndex, build_index


@pytest.fixture
def index(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "data/resumes").mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    shutil.copy(root / "config/skills.yaml", tmp_path / "config/skills.yaml")
    for cid in ("A", "B", "C", "D"):
        (tmp_path / f"data/resumes/{cid}.txt").write_text(
            f"Candidate {cid}\nSoftware Developer | Example | Jan 2020 - Jan 2025\n"
            "Built Python and SQL services. Maintained browser interfaces using React."
        )
    build_index(tmp_path, ner_backend="rules", semantic=False)
    return ResumeIndex(tmp_path, backend="bm25")


def full_request(index):
    return index._validate_requirements(
        Requirements(
            role="software_developer",
            must_have=[["python"], ["sql"]],
            minimum_years=3,
            skill_years={"python": 3, "sql": 2},
            nice_to_have=["react"],
            semantic_brief="Software work with Python and SQL experience; React is preferred.",
        )
    )


def test_context_score_can_recover_a_candidate_without_asserting_eligibility(index):
    req = full_request(index)
    batch = index.expand_retrieval(index.retrieve(req), req)
    batch.candidate_scores = {cid: (1, 1) for cid in index.candidates}
    batch.retrieval_audit["context_scores"] = {"A": -9, "B": 9, "C": 0, "D": 0}
    result = index.rank(batch, req)
    assert result.matches[0].candidate_id == "B"
    assert all(not row.assessments and not row.eligible for row in result.matches)
    assert all(len(row.retrieval_leads) == 7 for row in result.matches)
    assert all(row.score == sum(row.components.values()) for row in result.matches)


def test_retrieval_points_are_not_rounded_before_selection(index):
    req = full_request(index)
    batch = index.expand_retrieval(index.retrieve(req), req)
    # A tiny lexical margin must survive sorting even when the UI rounds it.
    batch.candidate_scores = {"A": (0, 1), "B": (0, 1.000001), "C": (0, 0.5), "D": (0, 0.2)}
    result = index.rank(batch, req)
    assert result.matches[0].candidate_id == "B"


def test_context_query_uses_frozen_capabilities_and_excludes_history(index):
    from screening_agent.candidate_retrieval import _context_query

    req = full_request(index)
    req.title = "Legacy Angular developer"
    req.source_text = "Angular first, then switch to Python and SQL."
    query = _context_query(req)
    assert "Angular" not in query and "angular" not in query
    assert "Optional preference: react" in query and "minimum 3 years" in query
    assert "python" in query and "sql" in query and "software developer" in query
    reversed_req = req.model_copy(update={"must_have": list(reversed(req.must_have))})
    assert _context_query(reversed_req) == query


def test_lexical_overlap_and_display_saturation_cannot_override_raw_learned_scores(index):
    req = full_request(index)
    batch = index.expand_retrieval(index.retrieve(req), req)
    batch.candidate_scores = {"A": (1, 1000), "B": (0, 0), "C": (0, 0), "D": (0, 0)}
    batch.retrieval_audit["context_scores"] = {"A": 49.0, "B": 50.0, "C": -2.0, "D": -3.0}
    matches = index.rank(batch, req).matches
    assert matches[0].candidate_id == "B"
    assert matches[0].score == matches[1].score == 100
    assert set(matches[0].components) == {"learned_relevance"}


def test_ranking_query_preserves_explicit_coursework_scope(index):
    from screening_agent.candidate_retrieval import _context_query

    req = Requirements(
        must_have=[["python"]],
        semantic_brief="Python study is sufficient; a completed course qualifies.",
    )
    assert req.semantic_brief in _context_query(req)
    assert "professional" not in index._query(req)


def test_display_labels_cannot_change_score_lookup(index, monkeypatch):
    req = full_request(index)
    batch = index.expand_retrieval(index.retrieve(req), req)
    expected = [row.model_dump() for row in index.rank(batch, req).matches]
    monkeypatch.setattr(index.skills, "display", lambda skill: f"Changed display: {skill.upper()}")
    batch.retrieval_audit["criteria"] = ["Changed presentation"]
    batch.retrieval_audit["criterion_labels"] = {
        key: "Changed label" for key in batch.criterion_scores
    }
    assert [row.model_dump() for row in index.rank(batch, req).matches] == expected


def test_reordered_requirements_reuse_same_id_scores(index):
    req = full_request(index)
    batch = index.expand_retrieval(index.retrieve(req), req)
    reordered = req.model_copy(update={"must_have": list(reversed(req.must_have))})
    assert {row["criterion_id"] for row in build_criteria(req)} == {
        row["criterion_id"] for row in build_criteria(reordered)
    }
    assert [row.model_dump() for row in index.rank(batch, req).matches] == [
        row.model_dump() for row in index.rank(batch, reordered).matches
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_id",
        "display_key",
        "extra_id",
        "missing_candidate",
        "nonfinite",
        "missing_context_candidate",
        "extra_context_candidate",
        "invalid_context",
        "empty_context",
    ],
)
def test_incomplete_expansion_cannot_silently_fall_back_to_base_scores(index, mutation):
    req = full_request(index)
    batch = index.expand_retrieval(index.retrieve(req), req)
    first = next(iter(batch.criterion_scores))
    if mutation == "missing_id":
        del batch.criterion_scores[first]
    elif mutation == "display_key":
        batch.criterion_scores["Required skill: Python"] = batch.criterion_scores.pop(first)
    elif mutation == "extra_id":
        batch.criterion_scores["unexpected"] = dict(batch.criterion_scores[first])
    elif mutation == "missing_candidate":
        del batch.criterion_scores[first]["A"]
    elif mutation == "nonfinite":
        batch.criterion_scores[first]["A"] = float("nan")
    else:
        batch.retrieval_audit["context_scores"] = {cid: 1.0 for cid in index.candidates}
        index.vectors = np.ones((len(index.chunks), 1))
        context = batch.retrieval_audit["context_scores"]
        if mutation == "missing_context_candidate":
            del context["A"]
        elif mutation == "extra_context_candidate":
            context["unexpected"] = 0.0
        elif mutation == "invalid_context":
            context["A"] = float("inf")
        else:
            context.clear()
    with pytest.raises(ValueError, match="criterion ID|incomplete|invalid|expanded candidate pool"):
        index.rank(batch, req)


def test_all_public_boundaries_normalize_aliases_and_numeric_defaults(index):
    raw = Requirements(must_have=[["ReactJS"]])
    normalized = index._validate_requirements(raw)
    assert isinstance(raw.minimum_years, int) and isinstance(normalized.minimum_years, float)
    batch = index.retrieve(raw)
    expanded = index.expand_retrieval(batch, raw)
    assert set(expanded.criterion_scores) == {
        row["criterion_id"] for row in build_criteria(normalized)
    }
    assert [row.model_dump() for row in index.rank(expanded, raw).matches] == [
        row.model_dump() for row in index.rank(expanded, normalized).matches
    ]


def test_whole_request_and_individual_criteria_have_separate_queries(index, monkeypatch):
    brief = (
        "Python experience in diagnosing production failures; basic SQL familiarity is sufficient."
    )
    req = Requirements(
        must_have=[["python"], ["sql"]],
        semantic_brief=brief,
        source_text="Earlier request: React browser developer.",
    )
    batch = index.retrieve(req)
    assert brief in index._query(req)
    assert "React" not in index._query(req)
    seen = []
    original = index.retrieve

    def capture(requirements, *, query_text=None):
        seen.append(query_text)
        return original(requirements, query_text=query_text)

    monkeypatch.setattr(index, "retrieve", capture)
    expanded = index.expand_retrieval(batch, req)
    assert len(seen) == len(expanded.criterion_scores)
    assert all("React" not in query and brief not in query for query in seen)
    assert any("python" in query for query in seen)
    assert any("sql" in query for query in seen)
    assert not any("python" in query and "sql" in query for query in seen)
    assert {row["criterion_id"] for row in build_criteria(req)} == set(expanded.criterion_scores)


def test_expansion_falls_back_to_current_structured_criteria_not_conversation_history(
    index, monkeypatch
):
    req = Requirements(
        must_have=[["python"]], source_text="React first; replace React with Python."
    )
    batch = index.retrieve(req)
    seen = []
    original = index.retrieve

    def capture(requirements, *, query_text=None):
        seen.append(query_text)
        return original(requirements, query_text=query_text)

    monkeypatch.setattr(index, "retrieve", capture)
    index.expand_retrieval(batch, req)
    assert seen and all(
        "python" in query.casefold() and "react" not in query.casefold() for query in seen
    )


def test_frozen_capability_reaches_query_without_unrelated_whole_brief(index, monkeypatch):
    req = Requirements(must_have=[["python"]], semantic_brief="Current Python requirement. " * 180)
    spec = build_criteria(req)[0]
    standard = EvidenceStandard(
        criterion_id=spec["criterion_id"],
        capability="Diagnosing Python production-service failures",
        requested_depth="application",
        experience_basis="not_applicable",
        evidence_scope="personal_application",
        sufficient_evidence="Source describes personally investigating production failures.",
        equivalence_boundary="Equivalent diagnostic work supports the requested capability.",
        uncertainty_boundary="An unexplained skill entry does not establish diagnostic work.",
    )
    req.evidence_standards = [standard]
    batch = index.retrieve(req)
    seen = []
    original = index.retrieve

    def capture(requirements, *, query_text=None):
        seen.append(query_text)
        return original(requirements, query_text=query_text)

    monkeypatch.setattr(index, "retrieve", capture)
    result = index.expand_retrieval(batch, req)
    assert len(seen) == 1
    assert standard.capability in seen[0]
    assert "Current Python requirement" not in seen[0] and len(seen[0]) <= 1000
    assert result.retrieval_audit["criterion_queries"][spec["criterion_id"]] == seen[0]
    assert req.semantic_brief == "Current Python requirement. " * 180
    assert req.evidence_standards == [standard]
