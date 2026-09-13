"""Current graph flows with recorded proposals; these are not model-accuracy tests."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from matching_agent import MatchingSession
from screening_agent.contextual import ContextualReviewer
from screening_agent.contracts import Candidate, Plan, Requirements, RetrievalBatch, SearchResult
from screening_agent.criteria import RequirementCompiler
from screening_agent.planner import PlanningError, SkillVocabulary

ROOT = Path(__file__).resolve().parents[1]


class ScriptedPlanner:
    mode, model, client = "openai", "recorded-test-inputs", object()

    def __init__(self):
        self.vocabulary = SkillVocabulary(ROOT)
        self.proposals, self.contexts = [], []

    def plan(self, text, current, **kwargs):
        self.contexts.append((text, current, deepcopy(kwargs["history"])))
        proposal = self.proposals.pop(0)
        if isinstance(proposal, Exception):
            raise proposal
        return proposal


@pytest.fixture
def flow(tmp_path, monkeypatch):
    fixture = json.loads((ROOT / "tests/fixtures/graph_flow.json").read_text())
    initial = deep = fixture
    planner = ScriptedPlanner()
    counts = {"search": 0, "deep": 0, "verify": 0}

    class Index:
        engine_fingerprint, manifest, documents = "recorded-index", {}, {}
        candidates = {
            m["candidate_id"]: Candidate(
                candidate_id=m["candidate_id"],
                name=m["name"],
                skills=[],
                source_path=m["evidence"][0]["source_path"],
            )
            for m in initial["shortlist"]
        }

        def _check_sources(self, *args):
            pass

        def retrieve(self, requirements):
            counts["search"] += 1
            return RetrievalBatch(
                requirements_fingerprint="recorded",
                engine_fingerprint=self.engine_fingerprint,
                candidate_scores={},
                evidence_scores=[],
                corpus_size=200,
                retrieved_count=40,
                backend="recorded",
            )

        def expand_retrieval(self, batch, *args, **kwargs):
            value = RetrievalBatch.model_validate(batch)
            value.retrieval_audit = {
                "criteria": [],
                "query_count": 2,
                "local_scored_candidate_count": len(self.candidates),
                "context_scores": {},
            }
            return value

        def rank(self, *args, **kwargs):
            recorded = deepcopy(initial["search_result"])
            result = SearchResult.model_validate(recorded)
            result.matches = result.matches[:10]
            for m in result.matches:
                m.assessments = []
                m.eligible = False
                m.strengths = []
                m.gaps = []
                m.screening_status = "needs_review"
            return result

    class ReplayReviewer(ContextualReviewer):
        def __init__(self, **kwargs):
            self.mode, self.model = "openai", planner.model

        def review(self, *args, **kwargs):
            counts["deep"] += 1
            return deepcopy(deep["detailed_reviews"])

        def verify(self, *args):
            counts["verify"] += 1

    def cohort(reviewer, reviews, matches, req, **kwargs):
        decisions = reviewer.apply_recommendations(reviews, matches, req)
        return decisions, {
            "model_calls": 2,
            "cache_hits": 0,
            "usage": {},
            "comparisons": [],
            "summary": {"text": "Recorded cohort comparison."},
            "audit": {"claims": []},
            "advance_tiers": [],
        }

    monkeypatch.setattr("screening_agent.recommendation.recommend_cohort", cohort)
    monkeypatch.setattr("screening_agent.contextual.ContextualReviewer", ReplayReviewer)
    # This fixture replays recorded semantic outputs to test graph transactions only.
    monkeypatch.setattr(
        RequirementCompiler,
        "compile_with_metadata",
        lambda self, req, **kwargs: (
            req,
            {
                "cache_hit": True,
                "usage": {},
                "model": planner.model,
                "standards_signature": "recorded",
            },
        ),
    )
    session = MatchingSession(tmp_path, index=Index(), planner=planner)

    def send(action, text=None, requirements=None, ids=()):
        planner.proposals.append(
            Plan(action=action, requirements=requirements, candidate_ids=list(ids))
        )
        result = session.send(text or action)
        assert not result.get("error"), result.get("error")
        return result

    yield SimpleNamespace(
        session=session,
        planner=planner,
        counts=counts,
        send=send,
        requirements=Requirements.model_validate(initial["requirements"]),
    )
    session.close()


def test_search_pauses_for_feedback_with_current_nodes(flow):
    result = flow.send("search", "Find Python and SQL experience", flow.requirements)
    assert len(result["shortlist"]) == 10 and result["search_result"]["corpus_size"] == 200
    nodes = [e["node"] for e in result["node_events"]]
    assert (
        nodes.index("search_resumes")
        < nodes.index("expand_search")
        < nodes.index("rank_candidates")
    )
    assert "contextual_screen" not in flow.session.graph.get_graph().nodes
    assert "human_feedback" in flow.session.graph.get_state(flow.session.config).next
    assert result["questions"] == [] and result["detailed_reviews"] == []


def test_staggered_request_retains_context_and_replaces_shortlist_atomically(flow):
    first = Requirements(minimum_years=3)
    flow.send("search", "Find someone with three years experience", first)
    refined = first.model_copy(update={"role": "software_developer"})
    result = flow.send("refine", "Add software developer", refined)
    assert result["requirements"]["minimum_years"] == 3
    assert result["requirements"]["role"] == "software_developer"
    assert result["requirements_version"] == 2 and flow.counts["search"] == 2
    assert flow.planner.contexts[-1][1] == first
    assert any("three years" in m.get("content", "") for m in flow.planner.contexts[-1][2])


def test_deep_then_recommendation_reuses_evidence_and_omits_questions(flow):
    flow.send("search", requirements=flow.requirements)
    deep = flow.send("deep_screen")
    assert deep["round"] == "deep" and len(deep["detailed_reviews"]) == 10
    assert deep["questions"] == [] and all(
        m["recommendation"] == "pending" for m in deep["shortlist"]
    )
    final = flow.send("finalize")
    assert final["round"] == "final" and flow.counts["deep"] == 1
    assert final["detailed_reviews"] == deep["detailed_reviews"]
    assert final["tool_events"][-1]["result_summary"]["model_calls"] == 2
    assert all(m["recommendation"] != "pending" for m in final["shortlist"])


def test_direct_recommendation_runs_missing_deep_round(flow):
    flow.send("search", requirements=flow.requirements)
    final = flow.send("finalize")
    assert final["round"] == "final" and flow.counts["deep"] == 1
    assert final["deep_screen_version"] == final["requirements_version"]


def test_refinement_invalidates_deep_review_before_later_recommendation(flow):
    flow.send("search", requirements=flow.requirements)
    flow.send("deep_screen")
    refined = flow.send(
        "refine",
        "Require five years overall",
        flow.requirements.model_copy(update={"minimum_years": 5}),
    )
    assert refined["deep_screen_version"] is None and refined["detailed_reviews"] == []
    assert refined["questions"] == [] and refined["comparison"] == {}
    final = flow.send("finalize")
    assert flow.counts["deep"] == 2 and final["deep_screen_version"] == 2


def test_failed_interpretation_preserves_committed_review(flow):
    before = flow.send("search", requirements=flow.requirements)
    flow.planner.proposals.append(PlanningError("Recorded provider failure"))
    after = flow.session.send("An interrupted follow-up")
    assert "Recorded provider failure" in after["report"]
    for field in ("requirements", "requirements_version", "shortlist", "search_result"):
        assert after[field] == before[field]


def test_comparison_uses_current_findings_without_reranking(flow):
    before = flow.send("search", requirements=flow.requirements)
    ids = [r["candidate_id"] for r in before["shortlist"][:3]]
    after = flow.send("compare", "Compare these top three", ids=ids)
    assert len(after["comparison"]["candidates"]) == 3
    assert after["shortlist"] == before["shortlist"]
    assert flow.counts == {"search": 1, "deep": 0, "verify": 0}


def test_finish_closes_feedback_loop(flow):
    flow.send("search", requirements=flow.requirements)
    final = flow.send("approve", "Finish this review")
    assert final["status"] == "complete"
    assert not flow.session.graph.get_state(flow.session.config).next
