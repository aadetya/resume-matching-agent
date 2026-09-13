"""Boundary tests for the browser presentation and downloadable artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def test_rendered_candidate_names_cannot_insert_html():
    from app import shortlist_html

    rendered = shortlist_html(
        {
            "shortlist": [
                {
                    "name": "<script>alert(1)</script>",
                    "candidate_id": "C001",
                    "score": 80,
                    "strengths": ["React"],
                }
            ]
        }
    )
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_requirement_badges_distinguish_optional_and_duration():
    from app import requirements_html

    rendered = requirements_html(
        {
            "requirements_version": 2,
            "requirements": {
                "role": "software_developer",
                "must_have": [["react", "vue"]],
                "nice_to_have": ["typescript"],
                "minimum_years": 3,
                "skill_years": {"react": 2},
            },
        }
    )
    assert "Software developer" in rendered
    assert "react / vue" in rendered
    assert "3+ years overall" in rendered
    assert "2+ years react" in rendered
    assert "typescript · preferred" in rendered
    assert "REVISION 2" in rendered


def test_export_serializes_only_current_review(tmp_path: Path):
    from app import Workspace

    workspace = object.__new__(Workspace)
    workspace.root = tmp_path

    class Session:
        def snapshot(self):
            return {
                "report": "Follow-up comparison",
                "screening_report": "# Current review",
                "requirements_version": 2,
                "shortlist": [],
            }

    workspace.session = lambda _: Session()
    md_path, json_path = workspace.export("89ca794a-4618-4c59-81b2-e011884185c0")
    assert Path(md_path).read_text() == "# Current review\n"
    assert json.loads(Path(json_path).read_text())["requirements_version"] == 2


def test_unknown_dates_show_a_lower_bound_without_claiming_missing_source_coverage():
    from screening_agent.ui_views import contextual_duration_text

    result = contextual_duration_text(
        {"months": 36, "maximum_months": None, "measurable_dates_complete": False}
    )
    assert "At least 36 months" in result and "some dates unknown" in result
    assert "coverage incomplete" not in result
