"""Import boundaries and preserved regression cases, independent of model rankings."""

import hashlib
import json
from pathlib import Path

import pytest

from screening_agent.file_tools import read_file
from screening_agent.metadata import MetadataExtractor
from scripts.generate_dataset import generate
from scripts.import_people import FIELDS, blocks, date_audit, date_text, sanitize, select

ROOT = Path(__file__).resolve().parents[1]


def test_sensitive_fields_and_freeform_bio_never_reach_screening_text():
    original = {
        "user": {
            "skills": ["Python"],
            "age": 42,
            "ethnicity": "PRIVATE_ETHNICITY",
            "legal_status": "PRIVATE_STATUS",
            "sponsorship_needed": True,
            "bio": "PRIVATE_BIO sponsorship",
            "career_interests": ["Desired software role"],
            "experience_years": 99,
        },
        "experience": [
            {
                "title": "Cashier",
                "company": "Shop",
                "start_date": "01/2020",
                "end_date": "01/2022",
                "description": "Handled payments.",
                "is_internship": False,
                "country": "PRIVATE_COUNTRY",
                "salary": 99999,
            }
        ],
    }
    profile = sanitize(original)
    assert set(profile["user"]) == {"skills"}
    assert set(profile["experience"][0]) == set(FIELDS["experience"])
    row = {
        "display_name": "Candidate P001",
        "source_id": "syn_visa_constrained_001",
        "profile": profile,
    }
    text = "\n".join(value for _, value in blocks(row))
    assert "Python" in text and "Cashier" in text
    assert "PRIVATE" not in text and "visa_constrained" not in text
    assert "Desired software role" not in text


def test_independent_date_oracle_counts_calendar_union_not_summed_jobs():
    profile = {
        "experience": [
            {"start_date": "01/2020", "end_date": "01/2022"},
            {"start_date": "01/2021", "end_date": "07/2023"},
            {"start_date": "01/2024", "end_date": "07/2024"},
        ]
    }
    assert date_audit(profile)["union_months"] == 48
    assert date_audit(profile)["experience_years"] == 4


def test_unknown_reversed_and_future_dates_are_explicit():
    profile = {
        "experience": [
            {"start_date": "01/2026", "end_date": None},
            {"start_date": "13/2020", "end_date": "01/2022"},
            {"start_date": "01/2028", "end_date": "01/2027"},
            {"start_date": "01/2025", "end_date": "01/2027"},
        ]
    }
    audit = date_audit(profile)
    assert audit["union_months"] == 20
    assert len(audit["warnings"]) == 3
    assert date_text("02/2024") == "2024-02"
    assert date_text(None) == "date unavailable"
    assert date_text(None, current=True) == "Present"
    assert (
        date_audit({"experience": [{"start_date": None, "end_date": None}]})["experience_years"]
        is None
    )


def test_unreviewed_source_revision_is_rejected_before_selection(tmp_path):
    file = tmp_path / "unreviewed.parquet"
    file.write_bytes(b"different source revision")
    with pytest.raises(ValueError, match="checksum"):
        select(file, tmp_path)
    assert not (tmp_path / "data").exists()


def test_legacy_generator_cannot_overwrite_imported_corpus(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/corpus_manifest.json").write_text('{"schema_version": 2}')
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        generate(tmp_path)


def test_shipped_source_splits_are_disjoint_and_full_corpus_is_not_shipped():
    data = ROOT / "data"
    active = [json.loads(line) for line in (data / "people/active.jsonl").read_text().splitlines()]
    heldout = [
        json.loads(line) for line in (data / "people/holdout.jsonl").read_text().splitlines()
    ]
    assert len(active) == 200 and len(heldout) == 50
    for key in ["candidate_id", "source_id", "content_hash"]:
        assert len({row[key] for row in active + heldout}) == 250
    assert not list(data.rglob("*.parquet"))
    assert len(list((data / "resumes").iterdir())) == 200
    assert len(list((data / "holdout/resumes").iterdir())) == 50


def test_original_hundred_regression_documents_remain_byte_identical():
    root = ROOT / "data/regression"
    manifest = json.loads((root / "data/corpus_manifest.json").read_text())
    assert len(manifest["files"]) == 100
    for entry in manifest["files"]:
        assert hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    truth = json.loads((root / "data/ground_truth/candidates.json").read_text())
    assert len(truth) == 100
    assert len(json.loads((root / "data/ground_truth/queries.json").read_text())) == 15


def test_preserved_regression_metadata_including_overlaps_and_course_only_skills():
    root = ROOT / "data/regression"
    manifest = json.loads((root / "data/corpus_manifest.json").read_text())
    files = {entry["candidate_id"]: entry for entry in manifest["files"]}
    truth = json.loads((root / "data/ground_truth/candidates.json").read_text())
    extractor = MetadataExtractor(root, backend="rules")
    for expected in truth:
        entry = files[expected["candidate_id"]]
        read = read_file(str(root / entry["path"]))
        assert read["ok"]
        candidate = extractor.extract(
            expected["candidate_id"], read["content"], entry["path"], entry["sha256"]
        )
        assert candidate.experience_years == expected["experience_years"], expected["candidate_id"]
        assert set(candidate.skills) == set(expected["skills"]), expected["candidate_id"]
        assert all(
            expected["skill_years"][skill] == years
            for skill, years in candidate.skill_years.items()
        ), expected["candidate_id"]
        # Historical labels credited technologies in job titles. Preserve those
        # labels, but require a responsibility assertion for current attribution.
        removed = set(expected["skill_years"]) - set(candidate.skill_years)
        body_skills = set()
        for evidence in candidate.evidence:
            if evidence.kind == "dated_employment":
                body_skills.update(extractor.skills.match(evidence.quote.partition("\n")[2]))
        assert not removed & body_skills, expected["candidate_id"]
