from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from screening_agent.chunking import chunk_text
from screening_agent.contracts import Requirements
from screening_agent.criteria import build_criteria
from screening_agent.metadata import MetadataExtractor, SkillExtractor, union_months
from screening_agent.retrieval import ResumeIndex, build_index
from screening_agent.roles import role_mentions

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "data/resumes").mkdir(parents=True)
    shutil.copy(ROOT / "config/skills.yaml", tmp_path / "config/skills.yaml")
    texts = {
        "A": "Jane Doe\njane@example.test\nWHERE I WORKED\nFrontend Engineer | Example Systems | Jan 2020 - Jan 2024\nBuilt React and JavaScript account screens. Used TypeScript.\nTECHNOLOGIES\nReact JavaScript TypeScript",
        "B": "John Smith\nFrontend Engineer | Sample Systems | Jan 2023 - Jan 2025\nBuilt React and JavaScript dashboards.\nTECHNOLOGIES\nReact JavaScript",
        "C": "Maya Sample\nOperations Analyst | Example Systems | Jan 2019 - Jan 2024\nPrepared manual invoices and spreadsheet exports.\nTECHNOLOGIES\nReact JavaScript",
        "D": "Nora Sample\nBackend Engineer | Example Systems | Jan 2020 - Jan 2024\nBuilt Python and SQL services.\nTECHNOLOGIES\nPython SQL",
    }
    for cid, text in texts.items():
        (tmp_path / f"data/resumes/{cid}.txt").write_text(text)
    build_index(tmp_path, ner_backend="rules", semantic=False, window=20, overlap=5)
    return tmp_path


def test_source_windows_are_exact_covering_and_overlapping():
    text = "  alpha\t" + "\n  ".join(f"word{i}" for i in range(70)) + "  "
    chunks = chunk_text(text, "C1", window=15, overlap=4)
    assert len(chunks) > 1
    assert all(text[item.start : item.end] == item.text for item in chunks)
    assert all(left.end > right.start for left, right in zip(chunks, chunks[1:], strict=False))
    assert chunks[0].start == 2 and chunks[-1].end == len(text.rstrip())
    assert chunks == chunk_text(text, "C1", window=15, overlap=4)


def test_chunk_invalid_window_cannot_loop():
    with pytest.raises(ValueError):
        chunk_text("text", "C1", window=8, overlap=8)


def test_concurrent_role_intervals_are_merged():
    assert union_months([(24000, 24048), (24024, 24060), (24080, 24092)]) == 72


def test_unknown_heading_retains_dated_skills(corpus):
    index = ResumeIndex(corpus, backend="bm25")
    assert index.candidates["A"].experience_years == 4
    assert index.candidates["A"].skill_years["react"] == 4
    assert index.candidates["C"].experience_years == 5
    assert "react" not in index.candidates["C"].skill_years


def test_negation_and_java_javascript_are_distinct():
    skills = SkillExtractor(ROOT)
    assert skills.match("No experience with React. Built JavaScript systems.") == ["javascript"]


def test_role_query_does_not_invent_a_technology_stack(corpus):
    index = ResumeIndex(corpus, backend="bm25")
    requirements = Requirements(role="software_developer", minimum_years=3)
    assert "React" not in index._query(requirements)
    assert "Python" not in index._query(requirements)
    assert "Role: Software developer." in index._query(requirements)
    batch = index.retrieve(requirements)
    assert set(batch.candidate_scores) == set(index.candidates)


def test_specific_role_changes_and_removal_update_current_query(corpus):
    index = ResumeIndex(corpus, backend="bm25")
    frontend = Requirements(role="frontend_developer", minimum_years=3)
    backend = Requirements(
        role="backend_developer",
        minimum_years=3,
        source_text="Find frontend developers; change role to backend developer.",
        title="Frontend developer",
    )
    assert "Frontend developer" in index._query(frontend)
    assert "Backend developer" in index._query(backend)
    assert "Frontend" not in index._query(backend)
    removed = Requirements(must_have=[["react"]], minimum_years=3)
    assert "Role:" not in index._query(removed)
    assert "React" in index._query(removed)


def test_separate_title_and_date_lines_keep_exact_employment_metadata():
    text = "Jamie Example\nSoftware Developer\nExample Co | Jan 2020 - Jan 2025\nBuilt Python services."
    candidate = MetadataExtractor(ROOT, backend="rules").extract(
        "C1", text, "data/resumes/C1.txt", "a" * 64
    )
    assert candidate.experience_years == 5
    evidence = [item for item in candidate.evidence if item.kind == "dated_employment"]
    assert evidence and all("Jan 2020 - Jan 2025" in item.quote for item in evidence)
    assert all(text[item.start : item.end] == item.quote for item in evidence)


def test_role_aliases_for_queries_prefer_specific_longest_phrase():
    assert role_mentions("Find software developers") == [(5, 24, "software_developer")]
    assert [role for _, _, role in role_mentions("front-end software engineer")] == [
        "frontend_developer"
    ]
    assert [role for _, _, role in role_mentions("backend developers")] == ["backend_developer"]


def test_runtime_never_reads_ground_truth(corpus):
    (corpus / "data/ground_truth").mkdir()
    (corpus / "data/ground_truth/candidates.json").write_text("invalid and forbidden to parse")
    build_index(corpus, ner_backend="rules", semantic=False)
    assert len(ResumeIndex(corpus, backend="bm25").candidates) == 4


def test_modified_source_invalidates_index(corpus):
    (corpus / "data/resumes/A.txt").write_text("changed after indexing")
    with pytest.raises(RuntimeError, match="source changed"):
        ResumeIndex(corpus, backend="bm25")


def test_new_resume_requires_rebuild(corpus):
    (corpus / "data/resumes/NEW.txt").write_text("New candidate\nReact")
    with pytest.raises(RuntimeError, match="source set changed"):
        ResumeIndex(corpus, backend="bm25")


@pytest.mark.parametrize("operation", ["retrieve", "expand", "rank"])
def test_retrieval_stages_recheck_changed_source(corpus, operation):
    index = ResumeIndex(corpus, backend="bm25")
    requirements = Requirements(must_have=[["react"]])
    batch = index.retrieve(requirements)
    (corpus / "data/resumes/A.txt").write_text("changed after index was loaded")
    with pytest.raises(RuntimeError, match="[Ss]ource changed"):
        if operation == "retrieve":
            index.retrieve(requirements)
        elif operation == "expand":
            index.expand_retrieval(batch, requirements)
        else:
            index.rank(batch, requirements)


def test_invalid_span_cannot_survive_review_integrity_even_with_rehashed_record(corpus):
    from screening_agent.contextual import source_packet
    from screening_agent.policy import fingerprint
    from screening_agent.review_integrity import ReviewIntegrity

    index = ResumeIndex(corpus, backend="bm25")
    requirements = Requirements(must_have=[["react"]])
    batch = index.retrieve(requirements)
    match = index.rank(index.expand_retrieval(batch, requirements), requirements).matches[0]
    reviewer = ReviewIntegrity()
    review = {
        "candidate_id": match.candidate_id,
        "reviewer_signature": reviewer.signature,
        "packet": source_packet(index, match.candidate_id, requirements),
        "criteria": [{"evidence": [match.evidence[0].model_dump()]}],
        "observations": [],
    }
    review["integrity"] = fingerprint(review)
    reviewer.verify([review], index, [match], requirements)
    review["criteria"][0]["evidence"][0]["quote"] = "invented achievement"
    review["integrity"] = fingerprint({k: v for k, v in review.items() if k != "integrity"})
    with pytest.raises(ValueError, match="invalid source citation"):
        reviewer.verify([review], index, [match], requirements)


@pytest.mark.model
def test_statistical_ner_finds_name_below_document_title():
    extractor = MetadataExtractor(ROOT, backend="spacy")
    candidate = extractor.extract(
        "C1",
        "Curriculum Vitae\nJane Doe\njane@example.test\nFrontend Engineer | Example Systems | Jan 2020 - Jan 2024\nBuilt React systems.",
        "data/resumes/C1.txt",
        "a" * 64,
    )
    assert candidate.name == "Jane Doe"
    assert any(
        item["label"] == "PERSON" and item["method"].startswith("spacy:")
        for item in candidate.entities
    )


@pytest.mark.model
def test_real_semantic_retrieval_and_resume_context_reranking():
    index = ResumeIndex(ROOT, backend="semantic")
    requirements = Requirements(
        role="software_developer",
        minimum_years=3,
        source_text="Software developer with three years of overall professional experience",
    )
    batch = index.retrieve(requirements)
    expanded = index.expand_retrieval(batch, requirements)
    result = index.rank(expanded, requirements, top_k=10)
    assert result.corpus_size == 200 and len(result.matches) == 10
    assert all(row.candidate_id.startswith("P") and row.source_id for row in result.matches)
    assert expanded.retrieval_audit["local_scored_candidate_count"] == 200
    assert len(result.retrieval_audit["reserve_candidate_ids"]) == 190
    assert expanded.retrieval_audit["candidate_assessment_calls"] == 0
    context = expanded.retrieval_audit["context_scores"]
    assert set(context) == set(index.candidates)
    assert np.isfinite(list(context.values())).all()
    assert expanded.retrieval_audit["criterion_labels"] == {
        spec["criterion_id"]: spec["criterion"] for spec in build_criteria(requirements)
    }
    assert set(expanded.retrieval_audit["context_token_counts"]) == set(index.candidates)
    assert expanded.retrieval_audit["heuristic_exclusions"] is False
    assert all(
        row.screening_status == "needs_review" and not row.assessments for row in result.matches
    )
    assert result.not_shortlisted_count == result.corpus_size - len(result.matches)
    for row in result.matches:
        text = index.documents[row.candidate_id]["text"]
        assert all(text[quote.start : quote.end] == quote.quote for quote in row.evidence)
        assert all(quote.source_id == row.source_id for quote in row.evidence)


def test_retrieve_rejects_source_changed_after_loading(corpus):
    index = ResumeIndex(corpus, backend="bm25")
    (corpus / "data/resumes/A.txt").write_text("different source after index load")
    with pytest.raises(RuntimeError, match="source changed"):
        index.retrieve(Requirements(must_have=[["react"]]))


def test_historical_removed_skill_cannot_influence_current_query(corpus):
    index = ResumeIndex(corpus, backend="bm25")
    req = Requirements(
        title="React Engineer",
        must_have=[["python"]],
        minimum_years=3,
        source_text="Find React candidates. Refinement: Replace React with Python.",
    )
    query = index._query(req)
    assert "React" not in query and "Python" in query
    default = index.retrieve(req)
    experiment = index.retrieve(req, query_text="React JavaScript accessible interfaces")
    assert default.candidate_scores != experiment.candidate_scores
    assert default.requirements_fingerprint == experiment.requirements_fingerprint
    assert (
        set(default.candidate_scores) == set(experiment.candidate_scores) == set(index.candidates)
    )


def test_explicit_partial_ingestion_keeps_valid_resumes_and_failure_report(corpus):
    (corpus / "data/resumes/BROKEN.pdf").write_bytes(b"%PDF-truncated")
    manifest = build_index(corpus, ner_backend="rules", semantic=False, strict=False)
    assert manifest["candidate_count"] == 4
    index = ResumeIndex(corpus, backend="bm25")
    result = index.retrieve(Requirements(must_have=[["react"]]))
    assert result.corpus_size == 4
    assert "BROKEN" not in index.candidates


@pytest.mark.parametrize(
    "body,expected",
    [
        ("Completed a two-day React course during this role.", None),
        ("Used React for only six months during a migration; other work used Python.", 0.5),
        (
            "Completed React training. Built React account screens and maintained them throughout the role.",
            5.0,
        ),
        ("Completed a React course, then built React account screens throughout this role.", 5.0),
    ],
)
def test_course_and_short_duration_do_not_inherit_entire_role(body, expected):
    extractor = MetadataExtractor(ROOT, backend="rules")
    text = (
        "Jane Doe\nEngineer | Example Systems | Jan 2020 - Jan 2025\n" + body + "\nSKILLS\nPython"
    )
    candidate = extractor.extract("C1", text, "resume.txt", "a" * 64)
    assert candidate.experience_years == 5
    assert candidate.skill_years.get("react") == expected
    assert "react" in candidate.skills


def test_overlapping_partial_skill_claims_cannot_double_count():
    extractor = MetadataExtractor(ROOT, backend="rules")
    text = (
        "Jane Doe\nEngineer | Example Systems | Jan 2020 - Jan 2025\nUsed React for only six months.\n"
        "Consultant | Other Systems | Jan 2021 - Jan 2024\nUsed React for six months.\nSKILLS\nReact"
    )
    candidate = extractor.extract("C1", text, "resume.txt", "a" * 64)
    assert candidate.skill_years["react"] == 0.5
