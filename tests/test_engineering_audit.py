"""Behavioral contracts discovered during the focused engineering audit."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from screening_agent.contracts import Requirements
from screening_agent.metadata import MetadataExtractor, SkillExtractor
from screening_agent.retrieval import ResumeIndex, build_index

ROOT = Path(__file__).resolve().parents[1]


def make_index(tmp_path, texts):
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "data/resumes").mkdir(parents=True)
    shutil.copy(ROOT / "config/skills.yaml", tmp_path / "config/skills.yaml")
    for cid, text in texts.items():
        (tmp_path / f"data/resumes/{cid}.txt").write_text(text)
    build_index(tmp_path, ner_backend="rules", semantic=False)
    return ResumeIndex(tmp_path, backend="bm25")


@pytest.fixture
def equal_index(tmp_path):
    body = "\nEMPLOYMENT\nDeveloper | Example | Jan 2020 - Jan 2025\nBuilt React dashboards.\nSKILLS\nReact"
    return make_index(tmp_path, {"A": "Alex Example" + body, "B": "Blair Example" + body})


def test_equivalent_documents_receive_equal_scores(equal_index):
    req = Requirements(must_have=[["react"]])
    rows = equal_index.rank(
        equal_index.expand_retrieval(equal_index.retrieve(req), req), req
    ).matches
    assert len(rows) == 2
    assert rows[0].score == rows[1].score
    assert rows[0].components == rows[1].components


def test_metadata_tenure_does_not_change_retrieval_points(equal_index):
    req = Requirements(must_have=[["react"]])
    batch = equal_index.expand_retrieval(equal_index.retrieve(req), req)
    candidate = equal_index.candidates["A"]
    candidate.experience_years = 1
    candidate.experience_months = 12
    young = equal_index.rank(batch, req).model_dump(exclude={"timings"})
    candidate.experience_years = 20
    candidate.experience_months = 240
    old = equal_index.rank(batch, req).model_dump(exclude={"timings"})
    assert young == old


def test_retrieval_ranking_rejects_a_batch_from_different_criteria(equal_index):
    req = Requirements(must_have=[["react"]])
    batch = equal_index.retrieve(req)
    with pytest.raises(ValueError, match="requirements"):
        equal_index.rank(batch, Requirements(must_have=[["python"]]))


@pytest.mark.parametrize("text", ["React was not used.", "React has never been used."])
def test_postposed_denial_is_not_a_positive_skill_claim(text):
    assert "react" not in SkillExtractor(ROOT).match(text)


@pytest.mark.parametrize(
    "claim",
    [
        "Partnered with the React team.",
        "The other team built React dashboards.",
        "Hoped to learn React.",
    ],
)
def test_another_persons_or_aspirational_work_does_not_credit_tenure(claim):
    text = "Alex Example\nEMPLOYMENT\nEngineer | Example | Jan 2020 - Jan 2025\n" + claim
    candidate = MetadataExtractor(ROOT, backend="rules").extract("A", text, "A.txt", "a" * 64)
    assert not candidate.skill_years.get("react")


def test_explicit_duration_extraction_does_not_inherit_entire_job(tmp_path):
    index = make_index(
        tmp_path,
        {
            "A": "Alex Example\nEMPLOYMENT\nEngineer | Example | Jan 2020 - Jan 2025\nUsed React for six months to build account screens."
        },
    )
    candidate = index.candidates["A"]
    assert candidate.claimed_skill_months == {"react": 6}
    assert candidate.skill_years["react"] == 0.5
    spans = [e for e in candidate.evidence if e.kind == "skill_duration:react"]
    assert spans and all("six months" in e.quote for e in spans)
    assert all(index.documents["A"]["text"][e.start : e.end] == e.quote for e in spans)


def test_full_role_usage_extraction_records_an_explicit_claim(tmp_path):
    index = make_index(
        tmp_path,
        {
            "A": "Alex Example\nEMPLOYMENT\nEngineer | Example | Jan 2020 - Jan 2025\nUsed React throughout this role to build account screens."
        },
    )
    assert index.candidates["A"].claimed_skill_months["react"] == 60
    spans = [e for e in index.candidates["A"].evidence if e.kind == "skill_duration:react"]
    assert spans and all("throughout this role" in e.quote for e in spans)


def test_metadata_title_alone_never_supplies_explicit_skill_usage_duration(tmp_path):
    index = make_index(
        tmp_path,
        {
            "A": "Alex Example\nEMPLOYMENT\nPython Developer | Example | Jan 2020 - Jan 2025\nSKILLS\nPython"
        },
    )
    assert not index.candidates["A"].skill_years
    assert not index.candidates["A"].claimed_skill_months


def test_legacy_metadata_preserves_month_precision_without_rounding(tmp_path):
    index = make_index(
        tmp_path,
        {
            "A": "Alex Example\nEMPLOYMENT\nEngineer | Example | Jan 2020 - Dec 2020\nBuilt Python services."
        },
    )
    assert index.candidates["A"].experience_months == 11
    assert index.candidates["A"].experience_years == round(11 / 12, 2)


def test_retrieval_and_ranking_have_distinct_output_contracts(equal_index):
    requirements = Requirements(must_have=[["react"]])
    batch = equal_index.retrieve(requirements)
    assert set(batch.candidate_scores) == {"A", "B"}
    assert "matches" not in batch.model_dump()
    with pytest.raises(ValueError, match="every expected criterion ID"):
        equal_index.rank(batch, requirements, top_k=1)
    expanded = equal_index.expand_retrieval(batch, requirements)
    assert len(equal_index.rank(expanded, requirements, top_k=1).matches) == 1
    with pytest.raises(ValueError, match="requirements"):
        equal_index.rank(batch, Requirements(must_have=[["python"]]))


def test_no_lexical_match_receives_no_retrieval_points(equal_index):
    req = Requirements(must_have=[["react"]])
    batch = equal_index.retrieve(req, query_text="zzzxxyy")
    rows = equal_index.rank(equal_index.expand_retrieval(batch, req), req).matches
    assert rows and all(row.components["retrieval"] == 0 for row in rows)


def test_final_recommendation_is_invariant_to_retrieval_score():
    from screening_agent.contracts import CriterionAssessment, Match
    from screening_agent.policy import recommend

    req = Requirements(must_have=[["react"]])
    ledger = [
        CriterionAssessment(
            criterion="Required skill: react",
            mandatory=True,
            status="supported",
            reason="Source-reviewed React implementation.",
        )
    ]
    low = Match(
        candidate_id="A", name="Anonymous", score=0, eligible=True, reviewed_assessments=ledger
    )
    high = low.model_copy(update={"score": 100})
    assert recommend(low, req).recommendation == recommend(high, req).recommendation == "hire"


def test_no_job_criteria_cannot_produce_a_positive_recommendation():
    from screening_agent.contracts import Match
    from screening_agent.policy import recommend

    match = Match(candidate_id="A", name="Anonymous", score=100, eligible=True)
    assert recommend(match, Requirements()).recommendation == "hold"


def test_token_budget_does_not_silently_drop_required_text(equal_index, monkeypatch):
    import screening_agent.retrieval as retrieval

    class Tokenizer:
        all_special_tokens = []

    class Reranker:
        tokenizer = Tokenizer()

        def preprocess(self, pairs, **kwargs):
            assert kwargs["processing_kwargs"]["text"]["truncation"] is False
            return {"attention_mask": np.ones((1, 8193))}

        def predict(self, pairs, **kwargs):
            pytest.fail("Oversized evidence must be rejected before inference")

    monkeypatch.setattr(retrieval, "_reranker", lambda: Reranker())
    req = Requirements(must_have=[["react"]])
    batch = equal_index.retrieve(req)
    monkeypatch.setattr(equal_index, "vectors", np.zeros((len(equal_index.chunks), 1)))
    monkeypatch.setattr(equal_index, "retrieve", lambda *args, **kwargs: batch)
    with pytest.raises(ValueError, match="local ranker limit"):
        equal_index.expand_retrieval(batch, req)


def test_policy_change_invalidates_active_index_and_saved_review(equal_index):
    from types import SimpleNamespace

    import yaml

    from matching_agent import MatchingSession
    from screening_agent.contracts import Plan
    from screening_agent.policy import ScreeningPolicy

    root = equal_index.root
    shared_db = root / "explicit.sqlite"
    planner = SimpleNamespace(
        mode="fixture",
        model="fixture-model",
        client=object(),
        plan=lambda *args, **kwargs: Plan(action="help", explanation="Controlled checkpoint test."),
    )
    with MatchingSession(
        root, index=equal_index, planner=planner, session_id="review", db_path=shared_db
    ) as old:
        old.send("Show help")
    values = ScreeningPolicy().model_dump()
    values["rrf_k"] = 10
    (root / "config/screening_policy.yaml").write_text(yaml.safe_dump(values))
    with pytest.raises(RuntimeError, match="policy changed"):
        equal_index.retrieve(Requirements(must_have=[["react"]]))
    new_index = ResumeIndex(root, backend="bm25")
    assert new_index.engine_fingerprint != equal_index.engine_fingerprint
    with MatchingSession(
        root, index=new_index, planner=planner, session_id="review", db_path=shared_db
    ) as new:
        with pytest.raises(ValueError, match="different screening engine"):
            new.send("Show help")


@pytest.mark.parametrize(
    "req",
    [
        Requirements(must_have=[[]]),
        Requirements(must_have=[[""]]),
        Requirements(unresolved=["Which role?"]),
    ],
)
def test_direct_retrieval_calls_validate_semantic_requirements(equal_index, req):
    with pytest.raises(ValueError):
        equal_index.retrieve(req)


def test_successful_file_read_survives_checkpoint_and_failed_read_clears_it(equal_index):
    from matching_agent import MatchingSession
    from screening_agent.contracts import Plan

    seen = []

    class Recorder:
        mode = "fixture"
        model = "fixture-model"
        client = object()

        def plan(self, text, current, **kwargs):
            seen.append(kwargs["history"])
            return Plan(action="help", explanation="Controlled document-context test.")

    root = equal_index.root
    content = "Frontend engineer. React required. Three years total experience."
    (root / "job.txt").write_text(content)
    with MatchingSession(
        root, index=equal_index, planner=Recorder(), session_id="file-review"
    ) as first:
        read = first.send("Read job.txt")
        assert read["document_context"]["content"] == content
    with MatchingSession(
        root, index=equal_index, planner=Recorder(), session_id="file-review"
    ) as resumed:
        resumed.send("Screen against the document I just opened")
        assert seen[-1][-1]["document"]["content"] == content
        failed = resumed.send("Read missing.txt")
        assert failed["document_context"] is None
        resumed.send("Use the document")
        assert not any(row.get("role") == "document" for row in seen[-1])


@pytest.mark.parametrize(
    "title,expected_months",
    [
        ("Data Engineering Bootcamp Student", 12),
        ("Bootcamp Participant", 12),
        ("Bootcamp Instructor", 24),
        ("Student Research Assistant", 24),
    ],
)
def test_training_label_cannot_become_employment_by_section_placement(
    tmp_path, title, expected_months
):
    index = make_index(
        tmp_path,
        {
            "A": f"Alex Example\nEMPLOYMENT\nDeveloper | Example | Jan 2020 - Jan 2021\nBuilt Python services.\n{title} | Academy | Jan 2021 - Jan 2022\nWorked on class assignments."
        },
    )
    assert index.candidates["A"].experience_months == expected_months
