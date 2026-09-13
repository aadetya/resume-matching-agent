"""UI regressions for action feedback, retained results, exports, and source display."""

from __future__ import annotations

import pytest

from screening_agent.ui_views import (
    candidate_evidence_html,
    match_status,
    questions_html,
    shortlist_html,
    source_html,
)


def test_candidate_source_quotes_keep_lines_and_escape_markup():
    quote = "Engineer | 2020–2024\nBuilt <script>malicious()</script>\nSecond responsibility."
    rendered = source_html(
        {
            "quote": quote,
            "source_path": "data/resumes/A.pdf",
            "start": 12,
            "end": 90,
            "document_sha256": "a" * 64,
        }
    )
    assert "2020–2024\nBuilt &lt;script&gt;" in rendered
    assert "<script>" not in rendered
    assert "<figcaption>A.pdf · characters 12–90</figcaption>" in rendered


def test_interview_presents_source_text_instead_of_serialized_evidence():
    rendered = questions_html(
        {
            "questions": [
                {
                    "candidate_id": "P001",
                    "question": "What did you own?",
                    "purpose": "Verify ownership.",
                    "evidence": [
                        {
                            "quote": "Built React components.",
                            "source_path": "data/resumes/P001.pdf",
                            "source_id": "hidden_cohort_01",
                            "start": 0,
                            "end": 23,
                            "document_sha256": "a" * 64,
                        }
                    ],
                }
            ]
        }
    )
    assert "Built React components." in rendered
    assert "Why ask this" in rendered
    assert "hidden_cohort_01" not in rendered
    assert '"candidate_id"' not in rendered


@pytest.mark.parametrize(
    "state,label",
    [({"round": "initial"}, "Provisional · unverified"), ({"round": "deep"}, "Needs verification")],
)
def test_pending_is_never_a_user_facing_screening_result(state, label):
    match = {"candidate_id": "A", "name": "A", "score": 80, "recommendation": "pending"}
    assert match_status(match, state)[0] == label
    assert "pending" not in shortlist_html({**state, "shortlist": [match]})


def test_legacy_review_requests_current_evidence_without_inventing_recommendation():
    match = {
        "candidate_id": "A",
        "name": "A",
        "score": 95,
        "assessments": [
            {
                "criterion": "React ownership",
                "mandatory": True,
                "status": "uncertain",
                "reason": "Only a skill list is available.",
                "evidence": [],
            }
        ],
    }
    rendered = candidate_evidence_html({"round": "deep", "shortlist": [match]}, "A", {})
    assert "qualification not assessed" in rendered
    assert "full resume" in rendered
    assert "Hire · advance" not in rendered


def test_activity_names_only_the_current_workflow():
    from screening_agent.ui_views import activity_html

    value = activity_html({"node_events": []})
    assert "Contextual screening" in value
    assert "Strict eligibility" not in value and "experimental candidate recovery" not in value


@pytest.mark.parametrize("key", ["", "   "])
def test_missing_key_reaches_the_chat_without_creating_a_review(tmp_path, monkeypatch, key):
    import threading

    from app import Workspace

    monkeypatch.setenv("OPENAI_API_KEY", key)
    workspace = object.__new__(Workspace)
    workspace.root = tmp_path
    workspace.sessions = {}
    workspace.lock = threading.RLock()
    workspace.components = {
        name: name for name in ("status", "chatbot", "message", "send", "reset", "resume_btn")
    }
    workspace.actions = ["send", "reset", "resume_btn"]
    result = workspace.send("Find Python developers.", None, None)
    assert "Screening requires OPENAI_API_KEY" in result["status"]
    assert "Screening requires OPENAI_API_KEY" in result["chatbot"][-1]["content"]
    assert "saved review" not in result["chatbot"][-1]["content"]
    assert not workspace.sessions
    assert not list(tmp_path.rglob("*.sqlite"))
