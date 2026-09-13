"""Independent examples for confusable skill names and responsibility evidence."""

import shutil
from pathlib import Path

import pytest

from screening_agent.contracts import Requirements
from screening_agent.metadata import SkillExtractor
from screening_agent.retrieval import ResumeIndex, build_index

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Built with Vue.js, Next.js and Node.js.", {"nodejs"}),
        ("Used JS and TS for browser code.", {"javascript", "typescript"}),
        ("Styled components with CSS-in-JS.", set()),
        ("CSS–in–JS styling and CSS—in—JS patterns.", set()),
        ("CSS-in-JS and JavaScript applications.", {"javascript"}),
        ("Built JS-based applications and TS-based tooling.", {"javascript", "typescript"}),
        ("Skills:\n- JS\n- TS", {"javascript", "typescript"}),
        ("React.js and JavaScript", {"react", "javascript"}),
        ("ReactJS and ECMAScript", {"react", "javascript"}),
        ("React Native", set()),
        ("React-native applications", set()),
        ("React\nNative mobile development", set()),
        ("React Native and React.js web applications", {"react"}),
        ("No React experience. Used Node.js.", {"nodejs"}),
        ("Learned Java, then used JavaScript.", {"java", "javascript"}),
        ("SQLAlchemy, PostgreSQL and MySQL", {"postgresql", "mysql"}),
    ],
)
def test_confusable_skills_do_not_supply_a_different_core_tool(text, expected):
    extractor = SkillExtractor(ROOT)
    assert set(extractor.match(text)) == expected
    assert all(text[row["start"] : row["end"]] == row["text"] for row in extractor.spans(text))


def test_retrieval_preserves_exact_overlapping_sources_without_role_qualification(tmp_path):
    (tmp_path / "config").mkdir()
    shutil.copy(ROOT / "config/skills.yaml", tmp_path / "config/skills.yaml")
    (tmp_path / "data/resumes").mkdir(parents=True)
    for cid, body in [
        ("BUILD", "Built REST APIs for product inventory."),
        ("OBSERVE", "Collaborated with the backend team building REST APIs."),
    ]:
        (tmp_path / f"data/resumes/{cid}.txt").write_text(
            f"Candidate {cid}\nEMPLOYMENT HISTORY\nSoftware Developer | Cedar | 2020-01 - 2024-01\n{body}"
        )
    build_index(tmp_path, ner_backend="rules", semantic=False, window=12, overlap=4)
    index = ResumeIndex(tmp_path, backend="bm25")
    req = index._validate_requirements(Requirements(role="backend_developer", minimum_years=3))
    batch = index.retrieve(req)
    expanded = index.expand_retrieval(batch, req)
    result = index.rank(expanded, req)
    assert expanded.retrieval_audit["local_scored_candidate_count"] == 2
    assert not expanded.retrieval_audit["heuristic_exclusions"]
    assert result.not_shortlisted_count == result.corpus_size - len(result.matches)
    assert {row.candidate_id for row in result.matches} == {"BUILD", "OBSERVE"}
    assert all(
        row.screening_status == "needs_review" and not row.assessments for row in result.matches
    )
    for row in result.matches:
        text = index.documents[row.candidate_id]["text"]
        assert row.evidence
        assert all(text[e.start : e.end] == e.quote for e in row.evidence)
        assert all(e.candidate_id == row.candidate_id for e in row.evidence)
        assert all(e.source_path == f"data/resumes/{row.candidate_id}.txt" for e in row.evidence)
        chunks = sorted(
            (c for c in index.chunks if c.candidate_id == row.candidate_id), key=lambda c: c.start
        )
        assert len(chunks) > 1
        assert all(left.end > right.start for left, right in zip(chunks, chunks[1:], strict=False))
        assert all(text[c.start : c.end] == c.text for c in chunks)
