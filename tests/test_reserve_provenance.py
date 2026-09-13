"""The published reserve retains its sealed source and judgment identities."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESERVE = ROOT / "data/evaluation/final_reserve"


def test_sealed_files_and_explicit_publication_projection():
    freeze = json.loads((RESERVE / "freeze.json").read_text())
    publication = json.loads((RESERVE / "publication.json").read_text())
    assert set(publication["files"]) == {
        "selection_manifest.json",
        "selection_protocol.json",
        "review_protocol.json",
        "judgments.json",
    }
    for name, sealed_hash in freeze["hashes"].items():
        actual = hashlib.sha256((RESERVE / name).read_bytes()).hexdigest()
        if name in publication["files"]:
            projection = publication["files"][name]
            assert projection["sealed_sha256"] == sealed_hash
            assert projection["published_sha256"] == actual
        else:
            assert actual == sealed_hash, name

    manifest = json.loads((ROOT / "reports/manifest.json").read_text())
    for path, projection in manifest["annotation_metadata"]["files"].items():
        raw = (ROOT / path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == projection["sha256"]
        content = json.loads(raw)
        for keys in projection["metadata_fields"]:
            assert keys[-1] in {"author", "reviewer_type", "independence_statement"}
            parent = content
            for key in keys[:-1]:
                parent = parent[key]
            del parent[keys[-1]]
        encoded = json.dumps(
            content, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        assert hashlib.sha256(encoded).hexdigest() == projection["unchanged_content_sha256"]


def test_reserve_sources_are_disjoint_and_canonical_text_matches_judgments():
    reserve = json.loads((RESERVE / "corpus_manifest.json").read_text())
    ids = {f["source_id"] for f in reserve["files"]}
    assert len(ids) == 50
    for path in ("data/corpus_manifest.json", "data/holdout/corpus_manifest.json"):
        corpus = json.loads((ROOT / path).read_text())
        assert ids.isdisjoint(f["source_id"] for f in corpus["files"])
    judgments = json.loads((RESERVE / "judgments.json").read_text())
    for case in judgments["cases"]:
        assert (RESERVE / "text" / f"{case['id']}.txt").read_text() == case["resume_text"]
