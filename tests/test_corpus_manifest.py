from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from screening_agent.contracts import Requirements
from screening_agent.metadata import MetadataExtractor, redact_identity
from screening_agent.planner import OpenAIPlanner
from screening_agent.retrieval import ResumeIndex, build_index

ROOT = Path(__file__).resolve().parents[1]


def write_manifest(root: Path, manifest: dict) -> None:
    (root / "data/corpus_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def read_manifest(root: Path) -> dict:
    return json.loads((root / "data/corpus_manifest.json").read_text(encoding="utf-8"))


@pytest.fixture
def manifest_corpus(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "data/resumes").mkdir(parents=True)
    shutil.copy(ROOT / "config/skills.yaml", tmp_path / "config/skills.yaml")
    files = []
    for cid, filename, source_id, text in [
        (
            "P001",
            "document_alpha.txt",
            "syn2_original_341",
            "An untrusted header name\nEMPLOYMENT HISTORY\nCashier | Sample Store | 2020-01 - Present\n"
            "Processed transactions and trained new employees.\nEDUCATION\nEngineering Student | 2015-01 - 2020-01\n"
            "SKILLS\nPython",
        ),
        (
            "P002",
            "document_beta.txt",
            "syn2_original_902",
            "Another source header\nEMPLOYMENT HISTORY\nSoftware Developer | Example Co | 2021-01 - 2024-01\n"
            "Built Python services.\nPROJECTS\nSoftware Engineer | 2010-01 - 2020-01\nBuilt React demos.",
        ),
    ]:
        relative = f"data/resumes/{filename}"
        path = tmp_path / relative
        path.write_text(text, encoding="utf-8")
        files.append(
            {
                "candidate_id": cid,
                "source_id": source_id,
                "display_name": f"Candidate {cid}",
                "path": relative,
                "split": "production",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    write_manifest(
        tmp_path,
        {
            "schema_version": 2,
            "dataset_id": "akzaidan/People",
            "snapshot_date": "2026-09-01",
            "files": files,
        },
    )
    build_index(tmp_path, ner_backend="rules", semantic=False, window=20, overlap=5)
    return tmp_path


def test_manifest_identity_survives_search_comparison_and_evidence(manifest_corpus):
    index = ResumeIndex(manifest_corpus, backend="bm25")
    assert set(index.candidates) == {"P001", "P002"}
    candidate = index.candidates["P001"]
    assert candidate.name == "Candidate P001"
    assert candidate.source_id == "syn2_original_341"
    assert candidate.experience_years == 6.67
    assert "python" not in candidate.skill_years
    assert not any("Display name uses" in warning for warning in candidate.warnings)
    matches = search(index, Requirements(minimum_years=3)).matches
    assert {row.candidate_id for row in matches} == {"P001", "P002"}
    for match in matches:
        candidate = index.candidates[match.candidate_id]
        assert match.source_id == candidate.source_id
        assert all(evidence.source_id == candidate.source_id for evidence in match.evidence)
        assert all(
            index.documents[match.candidate_id]["text"][item.start : item.end] == item.quote
            for item in match.evidence
        )
    assert index.manifest["snapshot_date"] == "2026-09-01"
    assert index.manifest["corpus_manifest"]["dataset_id"] == "akzaidan/People"


@pytest.mark.parametrize("operation", ["retrieve", "source_packet", "rank"])
def test_identity_change_invalidates_every_new_operation(manifest_corpus, operation):
    from screening_agent.contextual import source_packet

    index = ResumeIndex(manifest_corpus, backend="bm25")
    req = Requirements(must_have=[["python"]])
    batch = index.retrieve(req)
    manifest = read_manifest(manifest_corpus)
    manifest["files"][0]["source_id"] = "different_original_profile"
    write_manifest(manifest_corpus, manifest)
    with pytest.raises(RuntimeError, match="identity or snapshot manifest changed"):
        if operation == "retrieve":
            index.retrieve(req)
        elif operation == "source_packet":
            source_packet(index, "P001", req)
        else:
            index.rank(batch, req)


def test_source_checks_use_manifest_candidate_id_not_filename(manifest_corpus):
    from screening_agent.contextual import source_packet

    index = ResumeIndex(manifest_corpus, backend="bm25")
    (manifest_corpus / "data/resumes/document_alpha.txt").write_text("changed after loading")
    with pytest.raises(RuntimeError, match="source changed"):
        source_packet(index, "P001", Requirements())


def test_snapshot_changes_require_rebuild_and_recompute_tenure(manifest_corpus):
    manifest = read_manifest(manifest_corpus)
    manifest["snapshot_date"] = "2024-01-01"
    write_manifest(manifest_corpus, manifest)
    with pytest.raises(RuntimeError, match="snapshot manifest changed"):
        ResumeIndex(manifest_corpus, backend="bm25")
    build_index(manifest_corpus, ner_backend="rules", semantic=False)
    index = ResumeIndex(manifest_corpus, backend="bm25")
    assert index.candidates["P001"].experience_years == 4
    assert index.manifest["snapshot_date"] == "2024-01-01"


def test_holdout_regression_and_judgments_are_outside_production(manifest_corpus):
    for directory in [
        "evaluation/holdout/resumes",
        "tests/fixtures/regression/resumes",
        "data/ground_truth",
    ]:
        folder = manifest_corpus / directory
        folder.mkdir(parents=True)
        (folder / "do_not_index.txt").write_text(
            "Software Developer | 2000-01 - Present\nReact Python JavaScript"
        )
        (folder / "judgments.json").write_text("This deliberately is not valid JSON")
    build_index(manifest_corpus, ner_backend="rules", semantic=False)
    index = ResumeIndex(manifest_corpus, backend="bm25")
    assert set(index.candidates) == {"P001", "P002"}
    assert index.retrieve(Requirements()).corpus_size == 2


def test_unlisted_production_file_is_rejected_on_load_turn_and_build(manifest_corpus):
    index = ResumeIndex(manifest_corpus, backend="bm25")
    (manifest_corpus / "data/resumes/HOLDOUT.txt").write_text("Do not index")
    with pytest.raises(RuntimeError, match="source set changed"):
        ResumeIndex(manifest_corpus, backend="bm25")
    with pytest.raises(RuntimeError, match="source set changed"):
        index._check_sources(["P001"])
    with pytest.raises(ValueError, match="membership"):
        build_index(manifest_corpus, ner_backend="rules", semantic=False)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("split", "holdout", "Only production"),
        ("source_id", "", "nonempty source_id"),
        ("path", "data/resumes/../../evaluation/profile.txt", "direct, nonsymlink"),
        ("sha256", "0" * 64, "hash mismatch"),
        ("candidate_id", "../P001", "stable identifier"),
    ],
)
def test_invalid_source_manifest_fails_closed(manifest_corpus, field, value, message):
    manifest = read_manifest(manifest_corpus)
    manifest["files"][0][field] = value
    write_manifest(manifest_corpus, manifest)
    with pytest.raises(ValueError, match=message):
        build_index(manifest_corpus, ner_backend="rules", semantic=False, strict=False)


@pytest.mark.parametrize("field", ["candidate_id", "source_id", "path"])
def test_duplicate_source_identity_is_rejected(manifest_corpus, field):
    manifest = read_manifest(manifest_corpus)
    manifest["files"][1][field] = manifest["files"][0][field]
    write_manifest(manifest_corpus, manifest)
    with pytest.raises(ValueError, match=f"Duplicate corpus {field}"):
        build_index(manifest_corpus, ner_backend="rules", semantic=False)


def test_legacy_manifest_keeps_filename_discovery_and_ignores_identity_override(manifest_corpus):
    write_manifest(
        manifest_corpus,
        {"schema_version": 1, "files": [{"candidate_id": "injected", "display_name": "injected"}]},
    )
    build_index(manifest_corpus, ner_backend="rules", semantic=False)
    index = ResumeIndex(manifest_corpus, backend="bm25")
    assert set(index.candidates) == {"document_alpha", "document_beta"}
    assert all(candidate.source_id is None for candidate in index.candidates.values())
    assert index.manifest["corpus_manifest"] is None


def test_nontechnical_employment_and_concurrent_jobs_use_dated_union():
    text = (
        "Candidate P001\nPROFILE\nClaims 25 years of experience; wants a software developer role.\n"
        "EMPLOYMENT HISTORY\nCrew Member | Cafe | 2018-01 - 2021-01\nServed customers.\n"
        "Cashier | Store | 2020-01 - 2022-01\nHandled tills.\n"
        "Aide | Community Center | 2023-01 - Present\nOrganized activities.\n"
        "EDUCATION\nSoftware Engineer | School | 2000-01 - 2018-01\n"
        "PROJECTS\nSoftware Developer | Personal Project | 2010-01 - Present\nReact"
    )
    candidate = MetadataExtractor(ROOT, backend="rules", snapshot_date=date(2026, 9, 1)).extract(
        "P001", text, "resume.txt", "a" * 64
    )
    assert candidate.experience_years == 7.67
    assert len([item for item in candidate.evidence if item.kind == "dated_employment"]) == 3
    assert "react" not in candidate.skill_years


@pytest.mark.parametrize("end", ["Present", "Current", "Now", "2028-01"])
def test_current_and_future_ends_are_capped_at_snapshot_with_original_dates(end):
    text = (
        f"Candidate P001\nEMPLOYMENT HISTORY\nCashier | Shop | 2020-01 - {end}\nWorked full time."
    )
    candidate = MetadataExtractor(ROOT, backend="rules", snapshot_date=date(2023, 1, 1)).extract(
        "P001", text, "resume.txt", "a" * 64
    )
    assert candidate.experience_years == 3
    employment = candidate.evidence[0]
    assert end in employment.quote and text[employment.start : employment.end] == employment.quote
    assert any("ends after snapshot" in warning for warning in candidate.warnings) is (
        end == "2028-01"
    )


def test_future_start_and_undated_author_claim_do_not_supply_experience():
    text = (
        "Candidate P001\nClaims 20 years as a software engineer.\n"
        "EMPLOYMENT HISTORY\nCashier | Shop | 2027-01 - 2028-01\n"
        "Software Developer | Company | dates not supplied\nBuilt Python software for 20 years."
    )
    candidate = MetadataExtractor(ROOT, backend="rules").extract(
        "P001", text, "resume.txt", "a" * 64
    )
    assert candidate.experience_years is None
    assert candidate.skill_years == {}
    assert any("future employment range" in warning for warning in candidate.warnings)


def test_original_source_identifiers_are_redacted_from_model_ranking_text(manifest_corpus):
    index = ResumeIndex(manifest_corpus, backend="bm25")
    candidate = index.candidates["P001"]
    text = f"{candidate.name}\nSource: {candidate.source_id}\nBuilt Python services."
    redacted = redact_identity(text, candidate)
    assert candidate.name not in redacted and candidate.source_id not in redacted
    assert "Python services" in redacted


def test_default_session_database_isolates_rebuilt_corpus(manifest_corpus):
    from matching_agent import MatchingSession

    root = manifest_corpus
    with MatchingSession(
        root,
        session_id="same-user-session",
        backend="bm25",
        planner=OpenAIPlanner(root, client=object(), model="test"),
    ) as original:
        old_database = original.database_path
    manifest = read_manifest(root)
    manifest["snapshot_date"] = "2025-09-01"
    write_manifest(root, manifest)
    build_index(root, ner_backend="rules", semantic=False, window=20, overlap=5)
    with MatchingSession(
        root,
        session_id="same-user-session",
        backend="bm25",
        planner=OpenAIPlanner(root, client=object(), model="test"),
    ) as replacement:
        assert replacement.database_path != old_database
        assert not replacement.snapshot().get("messages")
    assert old_database.is_file()


def search(index, requirements):
    """Exercise current retrieval without a paid semantic assessment."""
    requirements = index._validate_requirements(requirements)
    batch = index.retrieve(requirements)
    return index.rank(index.expand_retrieval(batch, requirements), requirements)
