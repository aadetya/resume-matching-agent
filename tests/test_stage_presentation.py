"""Presentation regressions using source-linked records and changing review order."""

from copy import deepcopy

from screening_agent.ui_views import (
    comparison_html,
    comparison_rows,
    conversation_reply,
    shortlist_html,
    work_history_html,
)


def match(cid, score=50, *, supported=False):
    return {
        "candidate_id": cid,
        "name": f"Candidate {cid}",
        "score": score,
        "eligible": supported,
        "strengths": [],
        "gaps": [],
        "assessments": [],
        "components": {"retrieval": score},
        "reviewed_assessments": [
            {
                "criterion": "Role",
                "mandatory": True,
                "status": "supported" if supported else "uncertain",
                "reason": "Work described in the source.",
                "evidence": [],
            }
        ],
    }


def test_compare_has_current_top_three_after_deep_screen_clears_comparison():
    state = {
        "round": "deep",
        "comparison": {},
        "shortlist": [match("B", 30, supported=True), match("A", 90), match("C"), match("D")],
    }
    assert [r["candidate_id"] for r in comparison_rows(state)] == ["B", "A", "C"]
    rendered = comparison_html(state, {})
    assert "Candidate B" in rendered and "Candidate D" not in rendered
    assert "Requirement findings" in rendered and "Retrieval points" not in rendered
    assert rendered.index("Candidate B") < rendered.index("Candidate A")
    original = deepcopy(state)
    comparison_html(state, {})
    assert state == original


def test_explicit_comparison_keeps_requested_order_but_uses_current_findings():
    state = {
        "round": "deep",
        "comparison": {"candidates": [match("B"), match("A")]},
        "shortlist": [match("A", supported=True), match("B", supported=True)],
    }
    rows = comparison_rows(state)
    assert [r["candidate_id"] for r in rows] == ["B", "A"]
    assert all(r["eligible"] for r in rows)
    assert "Your selected candidates" in comparison_html(state, {})


def test_comparison_reply_does_not_call_higher_retrieval_score_the_deep_leader():
    state = {
        "round": "deep",
        "plan": {"action": "compare"},
        "shortlist": [match("B", 30, supported=True), match("A", 90)],
    }
    reply = conversation_reply(state, 200)
    assert "leads" not in reply and "points" not in reply


def test_employment_history_uses_work_units_and_keeps_projects_separate():
    ev = {
        "quote": "Built <script> with Python.\nSecond responsibility.",
        "source_path": "A.txt",
        "start": 0,
        "end": 57,
    }
    units = [
        {
            "unit_id": "U1",
            "scope": "employment",
            "title": "Developer | A & B",
            "start_date": "2022-01",
            "end_date": None,
            "end_kind": "present",
            "evidence": [ev],
        },
        {
            "unit_id": "U2",
            "scope": "employment",
            "title": "Analyst | Earlier employer",
            "start_date": None,
            "end_date": None,
            "end_kind": "unknown",
            "evidence": [],
        },
        {
            "unit_id": "U3",
            "scope": "project",
            "title": "Personal portfolio",
            "start_date": None,
            "end_date": None,
            "end_kind": "unknown",
            "evidence": [ev],
        },
    ]
    review = {
        "evidence_graph": {"work_units": units},
        "criteria": [{"criterion": "Python", "unit_ids": ["U1"]}],
    }
    rendered = work_history_html(review)
    assert rendered.count('class="work-entry"') == 2
    assert "2022-01 – Present" in rendered and "Start date not stated" in rendered
    assert "Developer | A &amp; B" in rendered and "&lt;script&gt;" in rendered
    assert "<script>" not in rendered and "Personal portfolio" not in rendered
    assert "Personal portfolio" in work_history_html(review, projects=True)


def test_each_stage_promotes_its_own_result_in_candidate_cards():
    row = match("A", 85, supported=True)
    row["recommendation"] = "hire"
    row["recommendation_reason"] = "All requested criteria have supporting evidence."
    initial = shortlist_html(
        {"shortlist": [{**row, "recommendation": "pending"}], "round": "initial"}
    )
    deep = shortlist_html({"shortlist": [{**row, "recommendation": "pending"}], "round": "deep"})
    final = shortlist_html({"shortlist": [row], "round": "final"})
    assert "Candidates to review" in initial and "retrieval points" in initial
    assert "Reviewed candidates" in deep and "required criteria supported" in deep
    assert "retrieval points" not in deep
    assert "Decision shortlist" in final and row["recommendation_reason"] in final
