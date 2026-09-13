"""Final decision boundaries must validate the physical corpus, not only cached text."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from screening_agent.contracts import Match, Requirements
from screening_agent.policy import fingerprint, requirements_fingerprint
from screening_agent.retrieval import ResumeIndex, build_index
from screening_agent.review_integrity import ReviewIntegrity


@pytest.fixture
def completed_review(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "data/resumes").mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    shutil.copy(root / "config/skills.yaml", tmp_path / "config/skills.yaml")
    files = []
    for candidate_id in ("P001", "P002"):
        path = tmp_path / f"data/resumes/{candidate_id}.txt"
        path.write_text(
            f"Candidate {candidate_id}\nEMPLOYMENT HISTORY\n"
            "Developer | Example | 2020-01 - 2023-01\nBuilt Python services."
        )
        files.append(
            {
                "candidate_id": candidate_id,
                "source_id": f"source_{candidate_id}",
                "display_name": f"Candidate {candidate_id}",
                "path": path.relative_to(tmp_path).as_posix(),
                "split": "production",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (tmp_path / "data/corpus_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset_id": "controlled/source-freshness",
                "snapshot_date": "2026-09-01",
                "files": files,
            }
        )
    )
    build_index(tmp_path, ner_backend="rules", semantic=False)
    index = ResumeIndex(tmp_path, backend="bm25")
    requirements = index._validate_requirements(Requirements(must_have=[["python"]]))
    reviewer = ReviewIntegrity()
    record = index.documents["P001"]
    evidence = index.candidates["P001"].evidence[0].model_dump()
    match = Match(candidate_id="P001", name="Candidate P001", score=70, eligible=True)
    review = {
        "candidate_id": "P001",
        "reviewer_signature": reviewer.signature,
        "packet": {
            "requirements_fingerprint": requirements_fingerprint(requirements),
            "engine_fingerprint": index.engine_fingerprint,
            "source_sha256": record["sha256"],
            "passages": [{"evidence": evidence}],
        },
        "criteria": [{"evidence": [evidence]}],
        "observations": [],
    }
    review["integrity"] = fingerprint(review)
    reviewer.verify([review], index, [match], requirements)
    return reviewer, index, match, requirements, review


@pytest.mark.parametrize("candidate_id", ["P001", "P002"])
def test_changed_file_after_review_blocks_decision_even_when_cached_text_is_unchanged(
    completed_review, candidate_id
):
    reviewer, index, match, requirements, review = completed_review
    cached = index.documents[candidate_id].copy()
    (index.root / f"data/resumes/{candidate_id}.txt").write_text("Changed after the review.")
    assert index.documents[candidate_id] == cached
    with pytest.raises(RuntimeError, match="source changed"):
        reviewer.verify([review], index, [match], requirements)


@pytest.mark.parametrize("mutation", ["add", "remove", "symlink", "identity"])
def test_corpus_membership_and_identity_changes_block_decision(completed_review, mutation):
    reviewer, index, match, requirements, review = completed_review
    path = index.root / "data/resumes/P002.txt"
    if mutation == "add":
        (path.parent / "unlisted.txt").write_text("A new source outside the reviewed corpus.")
    elif mutation == "remove":
        path.unlink()
    elif mutation == "symlink":
        original = path.read_text()
        path.unlink()
        replacement = index.root / "copied-source.txt"
        replacement.write_text(original)
        path.symlink_to(replacement)
    else:
        manifest_path = index.root / "data/corpus_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][1]["source_id"] = "different_original_identity"
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="[Cc]orpus|[Ss]ource"):
        reviewer.verify([review], index, [match], requirements)


def test_current_corpus_still_requires_exact_quoted_evidence(completed_review):
    reviewer, index, match, requirements, review = completed_review
    review["criteria"][0]["evidence"][0]["quote"] = "Invented employment achievement"
    review["integrity"] = fingerprint({k: v for k, v in review.items() if k != "integrity"})
    with pytest.raises(ValueError, match="invalid source citation"):
        reviewer.verify([review], index, [match], requirements)
