"""Persistent LangGraph screening workflow and command-line entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
import sqlite3
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Annotated, TypedDict
from uuid import uuid4

from langchain_core.messages import AnyMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from screening_agent.agent_tools import ToolRegistry, detect_file_request
from screening_agent.contracts import Match, Plan, Requirements
from screening_agent.planner import OpenAIPlanner, PlanningError
from screening_agent.roles import ROLE_LABELS


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    request: str
    plan: dict
    requirements: dict | None
    pending_requirements: dict | None
    pending_search_result: dict | None
    decision_memo: dict
    requirements_version: int
    requirements_history: list[dict]
    shortlist: list[dict]
    previous_ranking: list[dict]
    ranking_changes: list[dict]
    search_result: dict
    comparison: dict
    questions: list[dict]
    report: str
    screening_report: str
    round: str
    deep_screen_version: int | None
    detailed_reviews: list[dict]
    tool_events: Annotated[list[dict], operator.add]
    node_events: Annotated[list[dict], operator.add]
    tool_request: dict | None
    tool_result: dict | None
    error: str | None
    status: str
    planner_mode: str
    clarification: str
    document_context: dict | None
    engine_fingerprint: str


def _requirements(state: AgentState) -> Requirements:
    if not state.get("requirements"):
        raise PlanningError("Run a candidate search before screening or comparing candidates.")
    return Requirements.model_validate(state["requirements"])


def _node_event(node: str, summary: str) -> dict:
    return {"event_id": uuid4().hex, "node": node, "summary": summary}


def ranking_changes(before: list[dict], after: list[dict]) -> list[dict]:
    old = {row["candidate_id"]: (rank, row) for rank, row in enumerate(before, 1)}
    new = {row["candidate_id"]: (rank, row) for rank, row in enumerate(after, 1)}
    changes = []
    for candidate_id in dict.fromkeys([*new, *old]):
        previous, current = old.get(candidate_id), new.get(candidate_id)
        if (
            previous
            and current
            and previous[0] == current[0]
            and previous[1]["score"] == current[1]["score"]
        ):
            continue
        row = current[1] if current else previous[1]
        keys = (
            set(previous[1].get("components", {})) | set(current[1].get("components", {}))
            if previous and current
            else set()
        )
        deltas = {
            key: round(
                current[1].get("components", {}).get(key, 0)
                - previous[1].get("components", {}).get(key, 0),
                4,
            )
            for key in sorted(keys)
        }
        changes.append(
            {
                "candidate_id": candidate_id,
                "name": row["name"],
                "previous_rank": previous[0] if previous else None,
                "new_rank": current[0] if current else None,
                "score_delta": round(current[1]["score"] - previous[1]["score"], 2)
                if previous and current
                else None,
                "component_deltas": {key: value for key, value in deltas.items() if value},
            }
        )
    return changes


def _format_evidence(row: dict, maximum: int = 2) -> list[str]:
    lines = []
    for item in row.get("evidence", [])[:maximum]:
        quote = item["quote"].replace("\n", " ").strip()
        lines.append(
            f"> {quote}\n>\n> Source: `{item['source_path']}`, characters {item['start']}–{item['end']}."
        )
    return lines


def _format_changes(state: AgentState) -> list[str]:
    changes = state.get("ranking_changes", [])
    lines = ["Ranking changes:"] if changes else ["The shortlist order and scores did not change."]
    for change in changes:
        old, new = change["previous_rank"], change["new_rank"]
        movement = (
            f"entered at #{new}"
            if old is None
            else (f"left the shortlist (previously #{old})" if new is None else f"#{old} → #{new}")
        )
        score = (
            f"; score change {change['score_delta']:+.2f}"
            if change["score_delta"] is not None
            else ""
        )
        components = ", ".join(
            f"{key} {value:+.3f}" for key, value in change["component_deltas"].items()
        )
        lines.append(
            f"- {change['name']} (`{change['candidate_id']}`): {movement}{score}"
            + (f". Components: {components}." if components else ".")
        )
    if len(state.get("requirements_history", [])) > 1:
        previous, current = state["requirements_history"][-2:]
        changed = [
            key
            for key in ("role", "must_have", "nice_to_have", "minimum_years", "skill_years")
            if previous["requirements"].get(key) != current["requirements"].get(key)
        ]
        for key in changed:
            before, after = previous["requirements"].get(key), current["requirements"].get(key)
            if key == "role":
                old = ROLE_LABELS.get(before, "Any role")
                new = ROLE_LABELS.get(after, "Any role")
                lines.append(f"Required role changed from {old} to {new}.")
            elif key == "must_have":
                old = "; ".join(" OR ".join(group) for group in before) or "none"
                new = "; ".join(" OR ".join(group) for group in after) or "none"
                lines.append(f"Mandatory skills changed from {old} to {new}.")
            elif key == "nice_to_have":
                lines.append(
                    f"Preferred skills changed from {', '.join(before) or 'none'} to {', '.join(after) or 'none'}."
                )
            elif key == "minimum_years":
                lines.append(
                    f"Minimum total experience changed from {before:g} to {after:g} years."
                )
            else:
                lines.append(
                    "Skill-specific experience changed from "
                    + (
                        ", ".join(f"{skill}: {years:g} years" for skill, years in before.items())
                        or "none"
                    )
                    + " to "
                    + (
                        ", ".join(f"{skill}: {years:g} years" for skill, years in after.items())
                        or "none"
                    )
                    + "."
                )
        if changed:
            lines.append("The full corpus was searched again using the updated criteria.")
    return lines


def _joined_findings(items: list[str], fallback: str = "") -> str:
    return "; ".join(item.strip().rstrip(".;") for item in items if item.strip()) or fallback


def render_report(state: AgentState) -> str:
    action = state.get("plan", {}).get("action", "help")
    if state.get("error"):
        return state["error"] + (
            "\n\nThe previous screening state is still available."
            if state.get("requirements")
            else ""
        )
    if state.get("clarification"):
        return state["clarification"]
    if state.get("tool_result") is not None:
        outcome = state["tool_result"]
        if not outcome["ok"]:
            return "The file operation could not be completed: " + outcome["error"]
        payload = json.dumps(outcome["result"], indent=2, ensure_ascii=False)
        # Complete machine-readable tool output remains in state.
        suffix = (
            "\n\nThe full result is available in the session's tool_result field."
            if len(payload) > 12000
            else ""
        )
        return (
            f"Tool `{state['tool_request']['name']}` completed.\n\n```json\n{payload[:12000]}\n```"
            + suffix
        )
    if action == "help":
        if state.get("planner_mode") == "openai" and state.get("plan", {}).get("explanation"):
            return state["plan"]["explanation"]
        return (
            "Start with a job description, or try `Find React candidates with 3+ years overall experience`. "
            "Then compare the top 3, make a skill optional, request a deep screen, or generate final screening recommendations. "
            "Use `finish` when your review is complete.\n\n"
            "The connected planner accepts job descriptions and follow-up requests in the same review."
        )
    if action == "questions":
        lines = ["Candidate screening questions"]
        for item in state.get("questions", []):
            name = item.get("candidate_id", "Candidate")
            lines.append(f"\n**{name}**")
            question = item.get("question", item.get("text", ""))
            if question:
                lines.append(str(question))
            for key in (
                "purpose",
                "reason",
                "evidence",
                "what_to_listen_for",
                "assessment",
                "rubric",
                "follow_up",
                "strong_answer",
            ):
                if item.get(key):
                    value = item[key]
                    lines.append(
                        f"{key.replace('_', ' ').capitalize()}: "
                        + (
                            json.dumps(value, ensure_ascii=False)
                            if isinstance(value, (dict, list))
                            else str(value)
                        )
                    )
        return "\n\n".join(lines)
    if action in {"compare", "explain"}:
        if "ranking changes" in state.get("request", "").casefold():
            return "\n\n".join(_format_changes(state))
        ids = state["plan"].get("candidate_ids", [])
        direct = {
            row["candidate_id"]: row for row in state.get("comparison", {}).get("candidates", [])
        }
        ranked = {row["candidate_id"]: row for row in state.get("shortlist", [])}
        selected = [
            {**direct.get(candidate_id, {}), **ranked.get(candidate_id, {})}
            for candidate_id in ids
            if candidate_id in direct or candidate_id in ranked
        ]
        lines = [
            "Candidate comparison against requirements version "
            + str(state.get("requirements_version", 0))
        ]
        if selected:
            keys = sorted(set().union(*(row.get("components", {}).keys() for row in selected)))
            table_lines = []
            table_lines.append("| " + " | ".join(["Candidate", "Score", "Eligible", *keys]) + " |")
            table_lines.append(
                "|" + "|".join(["---", "---:", ":---:", *(["---:"] * len(keys))]) + "|"
            )
            for row in selected:
                score = f"{row['score']:.2f}" if "score" in row else "Not ranked in this round"
                cells = [
                    f"{row['components'][key]:.3f}" if key in row.get("components", {}) else "—"
                    for key in keys
                ]
                table_lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"{row['name']} (`{row['candidate_id']}`)",
                            score,
                            "Unverified"
                            if not (row.get("reviewed_assessments") or row.get("assessments"))
                            else "Yes"
                            if row["eligible"]
                            else "No",
                            *cells,
                        ]
                    )
                    + " |"
                )
            lines.append("\n".join(table_lines))
            if len(selected) >= 2 and all("score" in row for row in selected[:2]):
                a, b = selected[:2]
                first, second = sorted((a, b), key=lambda row: (-row["score"], row["candidate_id"]))
                delta = first["score"] - second["score"]
                if delta:
                    lines.append(
                        f"{first['name']} has a {delta:.2f}-point score lead over {second['name']}. The table separates the recorded score components; evidence and gaps below show what supports them."
                    )
                else:
                    lines.append(
                        "These candidates have equal scores. Candidate ID provides the deterministic tie-break."
                    )
            for row in selected:
                lines.append(f"\n**{row['name']}**")
                lines.append("Strengths: " + _joined_findings(row.get("strengths", [])) + ".")
                lines.append(
                    "Gaps: " + (_joined_findings(row.get("gaps", [])) or "Not assessed yet") + "."
                )
                lines.extend(_format_evidence(row))
        else:
            lines.append(
                "These candidates are outside the current retrieved set. Their direct criterion comparison is shown below."
            )
        if not selected:
            lines.append(
                "The structured comparison is available in the comparison panel and session export."
            )
        return "\n\n".join(lines)
    from screening_agent.ui_views import report_markdown

    return report_markdown(state)


def build_graph(registry: ToolRegistry, *, checkpointer):
    """Build a fresh graph; external clients remain outside persisted state."""

    def parse_jd(state: AgentState) -> dict:
        base = {
            "error": None,
            "clarification": "",
            "tool_request": None,
            "tool_result": None,
            "status": "running",
            "planner_mode": registry.planner.mode,
            "pending_requirements": None,
            "pending_search_result": None,
        }
        try:
            file_request = detect_file_request(
                state["request"], state.get("screening_report") or state.get("report", "")
            )
            if file_request:
                return {
                    **base,
                    "tool_request": file_request,
                    "plan": {"action": "help"},
                    "node_events": [_node_event("parse_jd", "Selected a validated file tool.")],
                }
            current = (
                Requirements.model_validate(state["requirements"])
                if state.get("requirements")
                else None
            )
            history = [
                {"role": "user" if item.type == "human" else "assistant", "content": item.content}
                for item in state.get("messages", [])
            ]
            if state.get("document_context"):
                history.append({"role": "document", "document": state["document_context"]})
            plan = registry.planner.plan(
                state["request"],
                current,
                candidates=registry.index.candidates,
                shortlist=state.get("shortlist", []),
                history=history,
            )
            if plan.action == "file_tool":
                arguments = plan.file_arguments.model_dump(exclude_none=True)
                if plan.file_tool_name == "write_file" and "content" not in arguments:
                    if not state.get("screening_report"):
                        raise PlanningError(
                            "Generate a screening report before saving it, or supply the exact content to write."
                        )
                    arguments["content"] = state["screening_report"]
                base["tool_request"] = {"name": plan.file_tool_name, "arguments": arguments}
            if (
                plan.action not in {"search", "refine", "help", "approve", "file_tool"}
                and current is None
            ):
                raise PlanningError(
                    "Run a candidate search before screening or comparing candidates."
                )
            if plan.requirements and plan.requirements.unresolved:
                base["clarification"] = (
                    "Clarify these criteria before running the search: "
                    + "; ".join(plan.requirements.unresolved)
                )
            return {
                **base,
                "plan": plan.model_dump(mode="json"),
                "node_events": [
                    _node_event("parse_jd", f"Planned {plan.action}: {plan.explanation}")
                ],
            }
        except PlanningError as exc:
            return {
                **base,
                "error": str(exc),
                "plan": {"action": "help"},
                "node_events": [
                    _node_event("parse_jd", "Request needs correction or clarification.")
                ],
            }

    def extract_requirements(state: AgentState) -> dict:
        plan = Plan.model_validate(state["plan"])
        result = registry.invoke(
            "extract_requirements",
            {"jd": state["request"]},
            proposal=plan.requirements,
            requirements=_requirements(state) if plan.action == "refine" else None,
            changed_criterion_ids=plan.changed_criterion_ids,
        )
        if not result["ok"]:
            return {"error": result["error"], "tool_events": [result["event"]]}
        return {
            "pending_requirements": result["result"],
            "tool_events": [result["event"]],
            "node_events": [
                _node_event(
                    "extract_requirements",
                    "Validated the proposed criteria before retrieval; the previous screening remains available.",
                )
            ],
        }

    def search_resumes(state: AgentState) -> dict:
        result = registry.invoke(
            "rag_search",
            {"requirements": state["pending_requirements"], "top_k": 10},
            retrieval_only=True,
        )
        if not result["ok"]:
            return {
                "error": result["error"],
                "tool_events": [result["event"]],
                "pending_requirements": None,
                "pending_search_result": None,
            }
        return {
            "pending_search_result": result["result"],
            "tool_events": [result["event"]],
            "node_events": [
                _node_event(
                    "search_resumes", f"Searched {result['result']['corpus_size']} indexed resumes."
                )
            ],
        }

    def expand_search(state: AgentState) -> dict:
        from langgraph.config import get_stream_writer

        started = time.perf_counter()
        try:
            batch = registry.index.expand_retrieval(
                state["pending_search_result"],
                Requirements.model_validate(state["pending_requirements"]),
                progress=get_stream_writer(),
            ).model_dump(mode="json")
            audit = batch["retrieval_audit"]
            return {
                "pending_search_result": batch,
                "tool_events": [
                    {
                        "event_id": uuid4().hex,
                        "tool": "expand_search",
                        "arguments": {"criteria": audit["criteria"]},
                        "status": "success",
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                        "result_summary": {
                            "query_count": audit["query_count"],
                            "locally_scored_candidates": audit["local_scored_candidate_count"],
                            "resume_context_ranking": bool(audit["context_scores"]),
                            "openai_calls": 0,
                        },
                    }
                ],
                "node_events": [
                    _node_event(
                        "expand_search",
                        f"Retrieved source passages and ranked {audit['local_scored_candidate_count']} complete resumes with the local relevance model, without candidate assessment calls.",
                    )
                ],
            }
        except (OSError, RuntimeError, ValueError) as exc:
            return {"error": f"Requirement retrieval failed: {exc}"}

    def rank_candidates(state: AgentState) -> dict:
        try:
            result = registry.index.rank(
                state["pending_search_result"],
                Requirements.model_validate(state["pending_requirements"]),
                top_k=10,
            ).model_dump(mode="json")
            # A source may change while its model assessment is running.
            registry.index._check_sources()
            result["not_shortlisted_count"] += max(0, len(result["matches"]) - 10)
            result["matches"] = result["matches"][:10]
        except (OSError, RuntimeError, ValueError) as exc:
            return {
                "error": f"Ranking failed: {exc}",
                "pending_requirements": None,
                "pending_search_result": None,
            }
        matches = result["matches"]
        version = state.get("requirements_version", 0) + 1
        requirements = state["pending_requirements"]
        previous = state.get("shortlist", []) if state["plan"]["action"] == "refine" else []
        return {
            "shortlist": matches,
            "comparison": {},
            "questions": [],
            "search_result": result,
            "requirements": requirements,
            "requirements_version": version,
            "requirements_history": [
                *state.get("requirements_history", []),
                {"version": version, "requirements": requirements, "request": state["request"]},
            ],
            "previous_ranking": previous,
            "round": "initial",
            "deep_screen_version": None,
            "detailed_reviews": [],
            "decision_memo": {},
            "pending_requirements": None,
            "pending_search_result": None,
            "ranking_changes": ranking_changes(previous, matches)
            if state["plan"]["action"] == "refine"
            else [],
            "node_events": [
                _node_event(
                    "rank_candidates",
                    f"Committed requirements version {version} with {len(matches)} profiles selected for review.",
                )
            ],
        }

    def compare_candidates(state: AgentState) -> dict:
        if "ranking changes" in state["request"].casefold():
            return {"comparison": {"ranking_changes": state.get("ranking_changes", [])}}
        result = registry.invoke(
            "compare_candidates",
            {"candidate_ids": state["plan"]["candidate_ids"]},
            requirements=_requirements(state),
            context_matches=state.get("shortlist", []),
        )
        return {
            "comparison": result.get("result", {}),
            "error": result.get("error"),
            "tool_events": [result["event"]],
            "node_events": [
                _node_event(
                    "compare_candidates", "Compared candidate evidence against the same criteria."
                )
            ],
        }

    def interview_questions(state: AgentState) -> dict:
        questions, events = [], []
        for candidate_id in state["plan"]["candidate_ids"]:
            review = next(
                (
                    r
                    for r in state.get("detailed_reviews", [])
                    if r["candidate_id"] == candidate_id and r["criteria"]
                ),
                None,
            )
            try:
                if review:
                    registry.deep_reviewer.verify(
                        [review],
                        registry.index,
                        [
                            Match.model_validate(m)
                            for m in state["shortlist"]
                            if m["candidate_id"] == candidate_id
                        ],
                        _requirements(state),
                    )
            except (OSError, RuntimeError, ValueError) as exc:
                return {"error": f"Interview evidence could not be verified: {exc}"}
            result = registry.invoke(
                "generate_interview_questions",
                {"candidate_id": candidate_id},
                requirements=_requirements(state),
                review=review,
            )
            events.append(result["event"])
            if not result["ok"]:
                return {"error": result["error"], "tool_events": events}
            questions.extend(
                {**question, "candidate_id": candidate_id} for question in result["result"]
            )
        return {
            "questions": questions,
            "tool_events": events,
            "node_events": [
                _node_event(
                    "interview_questions",
                    f"Prepared {len(questions)} questions from evidence and gaps.",
                )
            ],
        }

    def deep_screen(state: AgentState) -> dict:
        from langgraph.config import get_stream_writer

        progress = get_stream_writer()
        started = time.perf_counter()
        if not state.get("shortlist"):
            return {
                "error": "No candidates are available for deep screening. Adjust the requirements first."
            }
        try:
            matches = [Match.model_validate(row) for row in state["shortlist"]]
            reviews = registry.deep_reviewer.review(
                registry.index, matches, _requirements(state), progress=progress
            )
            rows = [match.model_dump(mode="json") for match in matches]
            rows = registry.deep_reviewer.apply_evidence(reviews, rows, _requirements(state))
            rows.sort(key=registry.deep_reviewer.rank_key)
        except (OSError, RuntimeError, ValueError) as exc:
            return {"error": f"Deep screening failed: {exc}"}
        return {
            "shortlist": rows,
            "detailed_reviews": reviews,
            "decision_memo": {},
            "comparison": {},
            "questions": [],
            "ranking_changes": ranking_changes(state["shortlist"], rows),
            "round": "deep",
            "deep_screen_version": state["requirements_version"],
            "tool_events": [
                {
                    "event_id": uuid4().hex,
                    "tool": "deep_screen",
                    "arguments": {
                        "candidate_ids": [row["candidate_id"] for row in state["shortlist"]]
                    },
                    "status": "success",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    "result_summary": {
                        "candidate_count": len(rows),
                        "model_calls": sum(r.get("model_calls", 0) for r in reviews),
                        "cache_hits": sum(r.get("cache_hits", 0) for r in reviews),
                        "evidence_count": sum(len(row["evidence"]) for row in rows),
                        "review_mode": registry.deep_reviewer.mode,
                        "model": registry.deep_reviewer.model,
                        "changed_criteria": sum(
                            c["changed"] for r in reviews for c in r["changes"]
                        ),
                        "model_input_tokens": sum(
                            r["usage"].get("input_tokens", 0) for r in reviews
                        ),
                        "model_output_tokens": sum(
                            r["usage"].get("output_tokens", 0) for r in reviews
                        ),
                        "source_checked_claims": sum(
                            len(r.get("claim_checks", [])) for r in reviews
                        ),
                        "revised_or_removed_claims": sum(
                            c["changed"] for r in reviews for c in r.get("changes", [])
                        ),
                    },
                }
            ],
            "node_events": [
                _node_event(
                    "deep_screen",
                    f"Completed detailed source review for {sum(r.get('stage') == 'deep' for r in reviews)} candidates; {sum(r.get('stage') == 'pending' for r in reviews)} remain pending. Audit disagreements remain visible for verification.",
                )
            ],
        }

    def final_screen(state: AgentState) -> dict:
        started = time.perf_counter()
        if state.get("deep_screen_version") != state.get("requirements_version"):
            return {
                "error": "Final screening requires deep analysis of the current requirements version."
            }
        try:
            registry.deep_reviewer.verify(
                state.get("detailed_reviews", []),
                registry.index,
                [Match.model_validate(value) for value in state.get("shortlist", [])],
                _requirements(state),
            )
            matches = deepcopy(state["shortlist"])
            from langgraph.config import get_stream_writer

            from screening_agent.recommendation import recommend_cohort

            matches, memo = recommend_cohort(
                registry.deep_reviewer,
                state["detailed_reviews"],
                matches,
                _requirements(state),
                progress=get_stream_writer(),
            )
            registry.deep_reviewer.verify(
                state["detailed_reviews"],
                registry.index,
                [Match.model_validate(m) for m in matches],
                _requirements(state),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return {"error": f"Final screening failed: {exc}"}
        return {
            "shortlist": matches,
            "decision_memo": memo,
            "round": "final",
            "tool_events": [
                {
                    "event_id": uuid4().hex,
                    "tool": "apply_recommendations",
                    "arguments": {"requirements_version": state["requirements_version"]},
                    "status": "success",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    "result_summary": {
                        "candidate_count": len(matches),
                        "source_integrity_rechecked": True,
                        "review_reused": True,
                        "model_calls": memo["model_calls"],
                        "cache_hits": memo["cache_hits"],
                        "model_input_tokens": memo["usage"].get("input_tokens", 0),
                        "model_output_tokens": memo["usage"].get("output_tokens", 0),
                        "comparison_count": len(memo["comparisons"]),
                        "claims_checked": len(memo["audit"]["claims"]),
                        "decisions": {
                            label: sum(m["recommendation"] == label for m in matches)
                            for label in ["hire", "hold", "no_hire"]
                        },
                    },
                }
            ],
            "node_events": [
                _node_event(
                    "final_screen",
                    "Compared the reviewed candidates and verified the decision memo against the same requirements.",
                )
            ],
        }

    def file_tool(state: AgentState) -> dict:
        request = state["tool_request"]
        requirements = _requirements(state) if state.get("requirements") else None
        result = registry.invoke(
            request["name"],
            request["arguments"],
            requirements=requirements,
            context_matches=state.get("shortlist", []),
        )
        update = {}
        if request["name"] == "read_file":
            # A failed read clears the previous document rather than reusing it.
            update["document_context"] = None
            if result["ok"]:
                document = result["result"]
                if len(document["content"]) <= 100_000:
                    update["document_context"] = {
                        "path": request["arguments"]["filepath"],
                        "sha256": document["metadata"]["sha256"],
                        "content": document["content"],
                    }
        return {
            **update,
            "tool_result": result,
            "tool_events": [result["event"]],
            "node_events": [
                _node_event(
                    "file_tool",
                    f"Completed {request['name']} with status {result['event']['status']}.",
                )
            ],
        }

    def generate_report(state: AgentState) -> dict:
        report = render_report(state)
        from screening_agent.ui_views import report_markdown

        complete_report = (
            report_markdown(state)
            if state.get("plan", {}).get("action") == "finalize" and not state.get("error")
            else report
        )
        saved = (
            complete_report
            if state.get("plan", {}).get("action")
            in {"search", "refine", "deep_screen", "finalize"}
            and not state.get("error")
            else state.get("screening_report", "")
        )
        return {
            "report": report,
            "screening_report": saved,
            "messages": [{"role": "assistant", "content": report}],
            "status": "awaiting_feedback",
            "node_events": [
                _node_event(
                    "generate_report",
                    "Prepared the evidence-linked response and saved the conversation.",
                )
            ],
        }

    def human_feedback(state: AgentState) -> dict:
        # This node starts again on resume. Keep everything before interrupt free of side effects.
        feedback = interrupt(
            {
                "type": "human_feedback",
                "requirements_version": state.get("requirements_version", 0),
                "round": state.get("round", "initial"),
                "prompt": "Review the report, change criteria, or finish.",
            }
        )
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("Human feedback must be a non-empty message.")
        return {
            "request": feedback.strip(),
            "messages": [{"role": "user", "content": feedback.strip()}],
            "status": "running",
        }

    def finish_review(state: AgentState) -> dict:
        completion = "Your review is recorded as complete. Screening recommendations remain advisory; the final employment decision belongs to the reviewer."
        pending_footer = "The graph is paused for your feedback. Compare candidates, change criteria, ask for questions, or finish the review."
        report = (
            state.get("screening_report", "").removesuffix(pending_footer).rstrip()
            + "\n\n"
            + completion
        )
        return {
            "report": report,
            "screening_report": report,
            "status": "complete",
            "messages": [{"role": "assistant", "content": report}],
            "node_events": [_node_event("finish_review", "Human explicitly completed the review.")],
        }

    def route_request(state: AgentState) -> str:
        if state.get("error") or state.get("clarification"):
            return "generate_report"
        if state.get("tool_request"):
            return "file_tool"
        action = state["plan"]["action"]
        if (
            action == "finalize"
            and state.get("detailed_reviews")
            and state.get("deep_screen_version") == state.get("requirements_version")
        ):
            return "final_screen"
        return {
            "search": "extract_requirements",
            "refine": "extract_requirements",
            "compare": "compare_candidates",
            "explain": "compare_candidates",
            "questions": "interview_questions",
            "deep_screen": "deep_screen",
            "finalize": "deep_screen",
            "approve": "finish_review",
            "help": "generate_report",
        }[action]

    builder = StateGraph(AgentState)
    nodes = {
        "parse_jd": parse_jd,
        "extract_requirements": extract_requirements,
        "search_resumes": search_resumes,
        "expand_search": expand_search,
        "rank_candidates": rank_candidates,
        "compare_candidates": compare_candidates,
        "interview_questions": interview_questions,
        "deep_screen": deep_screen,
        "final_screen": final_screen,
        "file_tool": file_tool,
        "generate_report": generate_report,
        "human_feedback": human_feedback,
        "finish_review": finish_review,
    }
    for name, function in nodes.items():
        builder.add_node(name, function)
    builder.add_edge(START, "parse_jd")
    builder.add_conditional_edges(
        "parse_jd",
        route_request,
        {
            name: name
            for name in nodes
            if name
            not in {
                "parse_jd",
                "search_resumes",
                "rank_candidates",
                "human_feedback",
                "expand_search",
            }
        },
    )
    builder.add_conditional_edges(
        "extract_requirements",
        lambda state: "generate_report" if state.get("error") else "search_resumes",
        ["generate_report", "search_resumes"],
    )
    builder.add_conditional_edges(
        "search_resumes",
        lambda state: "generate_report" if state.get("error") else "expand_search",
        ["generate_report", "expand_search"],
    )
    builder.add_conditional_edges(
        "expand_search",
        lambda state: "generate_report" if state.get("error") else "rank_candidates",
        ["generate_report", "rank_candidates"],
    )
    builder.add_edge("rank_candidates", "generate_report")
    builder.add_conditional_edges(
        "deep_screen",
        lambda state: (
            "final_screen"
            if not state.get("error") and state["plan"]["action"] == "finalize"
            else "generate_report"
        ),
        ["final_screen", "generate_report"],
    )
    for name in ("compare_candidates", "interview_questions", "file_tool", "final_screen"):
        builder.add_edge(name, "generate_report")
    builder.add_edge("generate_report", "human_feedback")
    builder.add_edge("human_feedback", "parse_jd")
    builder.add_edge("finish_review", END)
    return builder.compile(checkpointer=checkpointer)


class MatchingSession:
    """One resumable conversation. Models/indexes may be shared; graph state cannot be."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        session_id: str | None = None,
        backend: str = "semantic",
        planner=None,
        db_path: str | Path | None = None,
        index=None,
    ):
        self.root = Path(root or Path(__file__).parent).resolve()
        self.session_id = session_id or str(uuid4())
        if not self.session_id or len(self.session_id) > 200:
            raise ValueError("Session IDs must contain between one and 200 characters.")
        if index is None:
            from screening_agent.retrieval import ResumeIndex

            index = ResumeIndex(self.root, backend=backend)
        provenance = getattr(index, "manifest", {}).get("corpus_manifest") or {}
        corpus_sha = provenance.get("sha256")
        from screening_agent.policy import ENGINE_VERSION, fingerprint, load_policy

        self.planner = planner or OpenAIPlanner.from_env(self.root)
        planner = self.planner
        index_fingerprint = getattr(
            index,
            "engine_fingerprint",
            fingerprint(
                {
                    "engine": ENGINE_VERSION,
                    "corpus": corpus_sha,
                    "policy": load_policy(self.root).model_dump(),
                }
            ),
        )
        self.engine_fingerprint = fingerprint(
            {
                "index": index_fingerprint,
                "workflow": "contextual",
                "graph_code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "planner_model": getattr(planner, "model", None),
                "planner_mode": planner.mode,
                "contextual_code": {
                    name: hashlib.sha256(
                        (Path(__file__).parent / "src/screening_agent" / name).read_bytes()
                    ).hexdigest()
                    for name in ("contextual.py", "contextual_prompts.py")
                }
                if isinstance(planner, OpenAIPlanner)
                else None,
            }
        )
        database_name = f"sessions-{self.engine_fingerprint[:16]}.sqlite"
        database = Path(db_path) if db_path else self.root / "artifacts" / database_name
        self.database_path = database
        database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(database), check_same_thread=False, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.registry = ToolRegistry(self.root, index, self.planner)
        from screening_agent.contextual import ContextualReviewer

        self.registry.deep_reviewer = ContextualReviewer(
            root=self.root, client=self.planner.client, model=self.planner.model
        )
        self.graph = build_graph(self.registry, checkpointer=SqliteSaver(self.connection))
        self.config = {"configurable": {"thread_id": self.session_id}, "recursion_limit": 35}
        self._lock = threading.RLock()

    def send(self, text: str) -> dict:
        for _ in self.stream(text):
            pass
        return self.snapshot()

    def stream(self, text: str):
        """Yield completed graph updates; the checkpoint remains the source of truth.

        The UI can announce real node completion without guessing progress or
        running a separate copy of the screening workflow.
        """
        from langgraph.types import Command

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter a non-empty message.")
        if len(text) > 100_000:
            raise ValueError("Messages must contain at most 100,000 characters.")
        with self._lock:
            previous = self.graph.get_state(self.config)
            if (
                previous.values
                and previous.values.get("engine_fingerprint") != self.engine_fingerprint
            ):
                raise ValueError(
                    "This saved review uses a different screening engine. Start a new review; previous results remain in the saved database."
                )
            if previous.next:
                graph_input = Command(resume=text.strip())
            else:
                graph_input = {
                    "request": text.strip(),
                    "engine_fingerprint": self.engine_fingerprint,
                    "messages": [{"role": "user", "content": text.strip()}],
                    "status": "running",
                }
            for kind, event in self.graph.stream(
                graph_input, config=self.config, stream_mode=["updates", "custom"]
            ):
                yield {"__progress__": event} if kind == "custom" else event

    def call_tool(self, name: str, arguments: dict) -> dict:
        return self.send("/tool " + name + " " + json.dumps(arguments))

    def snapshot(self) -> dict:
        with self._lock:
            snapshot = self.graph.get_state(self.config)
            state = dict(snapshot.values)
            state["messages"] = [
                {
                    "role": "user" if message.type == "human" else "assistant",
                    "content": message.content,
                    "id": message.id,
                }
                for message in state.get("messages", [])
            ]
            for key, default in {
                "requirements": None,
                "requirements_version": 0,
                "requirements_history": [],
                "shortlist": [],
                "report": "",
                "comparison": {},
                "questions": [],
                "ranking_changes": [],
                "round": "initial",
                "tool_events": [],
                "node_events": [],
                "status": "new",
                "planner_mode": self.planner.mode,
            }.items():
                state.setdefault(key, default)
            state["session_id"] = self.session_id
            state["pending_nodes"] = list(snapshot.next)
            return state

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resume screening with a persistent human feedback loop"
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--backend", choices=["semantic", "dense", "bm25"], default="semantic")
    parser.add_argument("--session-id")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--query")
    parser.add_argument("--export-graph", type=Path)
    args = parser.parse_args(argv)
    try:
        planner = OpenAIPlanner.from_env(args.root)
        with MatchingSession(
            args.root,
            session_id=args.session_id,
            backend=args.backend,
            planner=planner,
            db_path=args.database,
        ) as session:
            if args.export_graph:
                args.export_graph.parent.mkdir(parents=True, exist_ok=True)
                args.export_graph.write_text(
                    session.graph.get_graph().draw_mermaid(), encoding="utf-8"
                )
                print(f"Graph saved to {args.export_graph}")
                return 0
            print(planner.disclosure)
            print(f"Session: {session.session_id}")
            if args.query:
                print(session.send(args.query)["report"])
                return 0
            while True:
                try:
                    message = input("\nYou: ")
                except (EOFError, KeyboardInterrupt):
                    print("\nSession saved. Reuse --session-id to continue.")
                    return 0
                if not message.strip():
                    continue
                state = session.send(message)
                print("\n" + state["report"])
                if state["status"] == "complete":
                    return 0
    except (PlanningError, OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
