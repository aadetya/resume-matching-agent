"""Expansion preserves reviewed sources and samples independently of ranking."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.expand_people import choose, coverage, rows
from scripts.import_people import select

ROOT = Path(__file__).resolve().parents[1]


def test_expansion_preserves_original_sources_and_evaluation_files():
    record = json.loads((ROOT / "data/people/expansion.json").read_text())
    old = record["original_manifest"]["files"]
    current = json.loads((ROOT / "data/corpus_manifest.json").read_text())
    assert current["files"][:100] == old
    assert current["candidate_count"] == 200
    assert current["formats"] == {"pdf": 100, "docx": 60, "txt": 40}
    for entry in old:
        assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    revision = json.loads(
        (ROOT / "data/regression/data/ground_truth/query_revision.json").read_text()
    )
    projections = json.loads((ROOT / "reports/manifest.json").read_text())["annotation_metadata"][
        "files"
    ]
    for path, digest in record["protected_files"].items():
        if path == revision["current"]["path"]:
            # The expansion's original snapshot remains immutable. A subsequent
            # documented label correction preserves that snapshot and links both
            # versions by content hashes; it does not rewrite resume sources.
            prior = revision["superseded"]
            assert prior["sha256"] == digest
            assert hashlib.sha256((ROOT / prior["path"]).read_bytes()).hexdigest() == digest
            digest = revision["current"]["sha256"]
        elif path in projections:
            assert projections[path]["original_sha256"] == digest
            digest = projections[path]["sha256"]
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    lines = (ROOT / "data/people/active.jsonl").read_bytes().splitlines(keepends=True)
    assert (
        hashlib.sha256(b"".join(lines[:100])).hexdigest()
        == record["original_active_records_sha256"]
    )
    assert (
        hashlib.sha256((ROOT / "data/people/holdout.jsonl").read_bytes()).hexdigest()
        == record["holdout_records_sha256"]
    )


def test_coverage_report_is_recomputed_from_source_fields():
    active = rows(ROOT / "data/people/active.jsonl")
    record = json.loads((ROOT / "data/people/expansion.json").read_text())
    assert record["after"] == coverage(active)
    assert record["added"] == coverage(active[100:])
    assert record["before"] == coverage(active[:100])
    assert [r["candidate_id"] for r in active] == [f"P{n:03d}" for n in range(1, 201)]


def test_selection_deduplicates_source_and_content_and_is_order_independent():
    def row(identity, content, role="web developer"):
        return {
            "source_id": identity,
            "content_hash": content,
            "family": "full_stack",
            "profile": {"user": {"skills": ["Python", "React"]}},
            "seed_role": role,
            "cohort": "standard",
            "seniority": "mid",
            "selection_key": identity,
        }

    prior = row("used", "used-content")
    pool = [
        prior,
        row("copy", "used-content"),
        row("a", "same"),
        row("b", "same"),
        row("c", "fresh", "full stack developer"),
    ]
    picked = choose(pool, [prior], {"python_react": 2})
    assert {r["source_id"] for r in picked} == {"a", "c"}
    assert picked == choose(list(reversed(pool)), [prior], {"python_react": 2})
    with pytest.raises(ValueError, match="Insufficient"):
        choose(pool, [prior], {"python_react": 3})


def test_original_selector_cannot_silently_shrink_expanded_corpus(tmp_path):
    marker = tmp_path / "data/people/expansion.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    with pytest.raises(ValueError, match="Refusing to replace"):
        select(tmp_path / "unused.parquet", tmp_path)
