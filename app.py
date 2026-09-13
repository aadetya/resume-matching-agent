"""Gradio review workspace with concise chat and explicit screening results."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import queue
import threading
import uuid
from pathlib import Path

import gradio as gr

from matching_agent import MatchingSession
from screening_agent.planner import OpenAIPlanner, PlanningError
from screening_agent.retrieval import ResumeIndex
from screening_agent.ui_views import (
    NODES,
    VIEWS,
    activity_html,
    candidate_evidence_html,
    changes_html,
    comparison_html,
    conversation_reply,
    metrics_html,
    questions_html,
    recommendations_html,
    report_markdown,
    requirements_html,
    result_view,
    shortlist_html,
    stage_html,
    status_html,
)

ROOT = Path(__file__).resolve().parent
CSS = (ROOT / "assets/workspace.css").read_text()
JS = (ROOT / "assets/workspace.js").read_text()
WELCOME = [
    {
        "role": "assistant",
        "content": "Tell me about the role you’re hiring for. I’ll find matching candidates, explain the evidence, and help you narrow the shortlist.\n\nYou can start with a few requirements and adjust them as we go.",
    }
]


def create_theme():
    theme = gr.themes.Base(
        primary_hue="emerald",
        neutral_hue="slate",
        font=[gr.themes.Font("system-ui")],
        font_mono=[gr.themes.Font("ui-monospace")],
    )
    theme.set(
        body_background_fill="#f5f7f8",
        body_text_color="#203331",
        body_text_color_subdued="#526764",
        block_background_fill="#ffffff",
        block_border_color="#dce4e2",
        block_label_background_fill="#ffffff",
        block_label_text_color="#526764",
        input_background_fill="#ffffff",
        input_border_color="#cbd7d3",
        input_placeholder_color="#647773",
        button_primary_background_fill="#146450",
        button_primary_background_fill_hover="#104f40",
        button_primary_border_color="#146450",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_hover="#edf4f1",
        button_secondary_border_color="#d2ded8",
        button_secondary_text_color="#284b42",
        background_fill_primary="#f5f7f8",
        background_fill_secondary="#eef3f1",
        border_color_primary="#dce4e2",
    )
    values = theme.to_dict()["theme"]
    theme.set(
        **{key: values[key[:-5]] for key in values if key.endswith("_dark") and key[:-5] in values}
    )
    return theme


def markup(value: str, **kwargs):
    return gr.HTML(value, padding=False, **kwargs)


class Workspace:
    def __init__(self, root: Path, backend: str):
        self.root = root
        self.index = ResumeIndex(root, backend=backend)
        self.sessions: dict[str, MatchingSession] = {}
        self.lock = threading.RLock()
        self.components: dict = {}
        self.groups: dict = {}
        self.actions: list = []

    def session(self, session_id: str | None) -> MatchingSession:
        session_id = str(uuid.UUID(session_id)) if session_id else str(uuid.uuid4())
        if session_id not in self.sessions:
            planner = OpenAIPlanner.from_env(self.root)
            self.sessions[session_id] = MatchingSession(
                self.root, session_id=session_id, index=self.index, planner=planner
            )
        return self.sessions[session_id]

    def history_path(self, session_id: str) -> Path:
        return self.root / "runtime" / "conversations" / f"{uuid.UUID(session_id)}.json"

    def save_history(self, session_id: str, history: list) -> None:
        path = self.history_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # UI prose is separate from the complete, unchanged graph transcript.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {"engine": self.session(session_id).engine_fingerprint, "history": history},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(path)

    def navigation(self, view: str) -> dict:
        view = view if view in VIEWS else "Candidates"
        return {
            **{group: gr.update(visible=name == view) for name, group in self.groups.items()},
            self.components["navigation"]: view,
        }

    def export(self, session_id: str | None) -> tuple[str, str]:
        if not session_id:
            raise gr.Error("Start a review before exporting.")
        session_id = str(uuid.UUID(session_id))
        state = self.session(session_id).snapshot()
        markdown = report_markdown(state) + "\n"
        payload = json.dumps(state, indent=2, ensure_ascii=False, default=str) + "\n"
        revision = state.get("requirements_version", 0)
        identity = hashlib.sha256((markdown + payload).encode()).hexdigest()[:12]
        directory = self.root / "runtime" / "exports" / session_id
        directory.mkdir(parents=True, exist_ok=True)
        md_path = directory / f"screening-report-v{revision}-{identity}.md"
        json_path = directory / f"review-data-v{revision}-{identity}.json"
        md_path.write_text(markdown, encoding="utf-8")
        json_path.write_text(payload, encoding="utf-8")
        return str(md_path), str(json_path)

    def view(
        self,
        state: dict,
        sid: str,
        history: list,
        *,
        selected: str | None = None,
        view: str | None = None,
        error: str | None = None,
    ) -> dict:
        c = self.components
        choices = [(m["name"], m["candidate_id"]) for m in state.get("shortlist", [])]
        selected = (
            selected if selected in {x[1] for x in choices} else choices[0][1] if choices else None
        )
        if state.get("plan", {}).get("action") == "questions" and state.get("questions"):
            selected = state["questions"][0]["candidate_id"]
        ready = bool(state.get("requirements"))
        active = ready and state.get("status") != "complete"
        md, data = self.export(sid) if ready else (None, None)
        status = error or state.get("error") or state.get("clarification")
        if not status:
            status = (
                (
                    "Review saved. You can download the report."
                    if state.get("status") == "complete"
                    else "Done. "
                    + {
                        "Compare": "The comparison is open below.",
                        "Evidence": "Candidate findings are open below.",
                        "Recommendations": "Decisions and next actions are open below.",
                        "Interview": "The interview guide is open below.",
                    }.get(view or result_view(state), "Your results are ready.")
                )
                if ready
                else "Ready when you are. Describe a role to begin."
            )
        values = {
            "chatbot": history,
            "message": gr.update(value="", interactive=state.get("status") != "complete"),
            "session_id": sid,
            "criteria": requirements_html(state),
            "shortlist": {"html": shortlist_html(state), "selected": selected},
            "stage": stage_html(state),
            "metrics": metrics_html(state, len(self.index.candidates)),
            "report": report_markdown(state),
            "candidate": gr.update(choices=choices, value=selected, interactive=bool(choices)),
            "interview_candidate": gr.update(
                choices=choices, value=selected, interactive=bool(choices)
            ),
            "evidence": candidate_evidence_html(state, selected, self.index.candidates),
            "comparison": comparison_html(state, self.index.candidates),
            "questions": questions_html(state),
            "recommendations": {"html": recommendations_html(state), "selected": None},
            "activity": activity_html(state),
            "changes": changes_html(state),
            "saved": f"**Review ID**\n\n`{sid}`\n\nSaved on this computer. Copy this ID to resume the review later.",
            "status": status_html(status, error=bool(error or state.get("error"))),
            "md_file": gr.update(value=md, interactive=bool(md)),
            "json_file": gr.update(value=data, interactive=bool(data)),
            "download_note": f"Report ready · revision {state.get('requirements_version', 0)} · {len(state.get('shortlist', []))} shortlisted candidates"
            if ready
            else "Downloads become available after your first search.",
            "trace": {
                "steps": state.get("node_events", []),
                "tools": state.get("tool_events", []),
                "last_file_result": state.get("tool_result"),
            },
            "send": gr.update(interactive=state.get("status") != "complete"),
            "reset": gr.update(interactive=True),
            "resume_btn": gr.update(interactive=True),
            "compare": gr.update(interactive=active and len(choices) >= 2),
            "deep": gr.update(interactive=active and bool(choices)),
            "final": gr.update(interactive=active and bool(choices)),
            "interview": gr.update(interactive=active and state.get("round") in {"deep", "final"}),
            "finish": gr.update(interactive=active),
            "next_step": gr.update(
                visible=ready,
                interactive=bool(choices),
                value="Open the report"
                if state.get("round") == "final" or state.get("status") == "complete"
                else "Continue to recommendations"
                if state.get("round") == "deep"
                else "Next: detailed screening",
            ),
        }
        return {
            **{c[key]: value for key, value in values.items()},
            **self.navigation(view or result_view(state)),
        }

    def respond(
        self, text: str, sid: str | None, history: list | None, selected: str | None = None
    ):
        """Bridge one graph worker to Gradio's potentially different iteration threads.

        No thread-owned lock crosses a UI yield. The producer owns the graph,
        checkpoint, shared index, and export work through the complete request.
        """
        if not text or not text.strip():
            yield {self.components["status"]: status_html("Enter a request before sending.")}
            return
        history = list(history or WELCOME)
        history.append({"role": "user", "content": text.strip()})
        events = queue.Queue()
        acknowledged = threading.Event()

        def produce():
            try:
                with self.lock:
                    session = self.session(sid)
                    review_id = session.session_id
                    events.put(("begin", review_id))
                    acknowledged.wait()
                    error = None
                    try:
                        for update in session.stream(text):
                            for node, delta in update.items():
                                if node == "__progress__":
                                    events.put(("candidate_progress", delta))
                                elif not node.startswith("__"):
                                    events.put(
                                        (
                                            "progress",
                                            (
                                                node,
                                                delta.get("error")
                                                if isinstance(delta, dict)
                                                else None,
                                            ),
                                        )
                                    )
                        state = session.snapshot()
                        reply = conversation_reply(state, len(self.index.candidates))
                    except (ValueError, RuntimeError, OSError) as exc:
                        state = session.snapshot()
                        error = str(exc)
                        reply = f"I couldn’t complete that request: {error}\n\nYour previously completed results are still available."
                    finished_history = [*history, {"role": "assistant", "content": reply}]
                    self.save_history(review_id, finished_history)
                    events.put(
                        (
                            "result",
                            self.view(
                                state, review_id, finished_history, selected=selected, error=error
                            ),
                        )
                    )
            except Exception as exc:
                if isinstance(exc, PlanningError):
                    message = str(exc)
                else:
                    logging.getLogger(__name__).exception(
                        "The interface could not finish a review request"
                    )
                    message = "The interface could not finish this request. Your saved review is still available. Resume it from Saved, or start a new review."
                events.put(
                    (
                        "result",
                        {
                            self.components["status"]: status_html(message, error=True),
                            self.components["chatbot"]: [
                                *history,
                                {"role": "assistant", "content": message},
                            ],
                            self.components["message"]: gr.update(interactive=True),
                            **{
                                component: gr.update(
                                    interactive=component
                                    in [self.components[k] for k in ("send", "reset", "resume_btn")]
                                )
                                for component in self.actions
                            },
                        },
                    )
                )
            finally:
                events.put(("end", None))

        threading.Thread(target=produce, name="screening-review", daemon=True).start()
        completed = []
        try:
            while True:
                kind, payload = events.get()
                if kind == "end":
                    return
                if kind == "begin":
                    yield {
                        self.components["chatbot"]: [
                            *history,
                            {
                                "role": "assistant",
                                "content": "I’m working on your request. Progress will appear above the results.",
                            },
                        ],
                        self.components["message"]: gr.update(value="", interactive=False),
                        self.components["session_id"]: payload,
                        self.components["status"]: status_html(
                            "Understanding your request…", busy=True
                        ),
                        **{component: gr.update(interactive=False) for component in self.actions},
                    }
                    acknowledged.set()
                elif kind == "candidate_progress":
                    message = payload.get("message") or (
                        f"Detailed screening: {payload['completed']} of {payload['total']} candidates processed. "
                        f"{payload['candidate_id']} {'needs a retry' if payload['pending'] else 'has completed evidence checks'}."
                    )
                    yield {self.components["status"]: status_html(message, busy=True)}
                elif kind == "progress":
                    node, failure = payload
                    completed.append(node)
                    yield {
                        self.components["status"]: status_html(
                            failure or f"{NODES.get(node, node)} — complete. Continuing…",
                            busy=not failure,
                            error=bool(failure),
                        ),
                        self.components["activity"]: activity_html(
                            {}, completed=completed, busy=True
                        ),
                    }
                else:
                    yield payload
        finally:
            # A disconnected browser must not strand the graph's worker lock.
            acknowledged.set()

    def send(
        self, text: str, sid: str | None, history: list | None, selected: str | None = None
    ) -> dict:
        final = {}
        for update in self.respond(text, sid, history, selected):
            final = update
        return final

    def new_review(self) -> dict:
        with self.lock:
            session = self.session(None)
            return self.view({}, session.session_id, list(WELCOME), view="Candidates")

    def resume(self, sid: str) -> dict:
        try:
            sid = str(uuid.UUID(sid.strip()))
        except (ValueError, AttributeError) as exc:
            raise gr.Error("Enter the review ID copied from a saved review.") from exc
        with self.lock:
            session = self.session(sid)
            state = session.snapshot()
            if not state.get("messages"):
                raise gr.Error(
                    "No saved conversation was found for this ID in the current app and workflow version."
                )
            path = self.history_path(sid)
            history = None
            if path.exists():
                saved = json.loads(path.read_text())
                if saved.get("engine") == session.engine_fingerprint:
                    history = saved.get("history")
            if not history:
                # Old transcripts remain complete in the JSON download. Avoid
                # putting their many-page reports back into the small chat pane.
                history = [
                    {
                        "role": "assistant",
                        "content": "I restored your saved review. The complete earlier conversation is in the review-data download; the current findings are available in the result views.",
                    },
                    {
                        "role": "assistant",
                        "content": conversation_reply(state, len(self.index.candidates)),
                    },
                ]
            return self.view(state, sid, history)

    def select_candidate(self, cid: str | None, sid: str | None) -> dict:
        if not sid:
            return {}
        state = self.session(sid).snapshot()
        return {
            self.components["evidence"]: candidate_evidence_html(state, cid, self.index.candidates)
        }

    def open_candidate(self, value: dict, sid: str | None) -> dict:
        cid = value.get("selected")
        return {
            **self.select_candidate(cid, sid),
            self.components["candidate"]: cid,
            **self.navigation("Evidence"),
        }


def create_app(root: Path = ROOT, backend: str = "semantic") -> gr.Blocks:
    workspace = Workspace(root, backend)
    configured = bool(os.getenv("OPENAI_API_KEY", "").strip())
    mode = "OpenAI configured" if configured else "OpenAI key required"
    with gr.Blocks(title="Shortlist · Candidate review") as demo:
        c = workspace.components
        c["session_id"] = gr.State()
        markup(
            f'<header class="masthead"><div class="brand-lockup"><span class="brand-icon" aria-hidden="true">s.</span><div><h1>Shortlist</h1></div></div><span class="connection"><span aria-hidden="true">●</span> {mode}</span></header>'
        )
        gr.HTML(
            '<div class="mobile-switch" role="tablist" aria-label="Workspace"><button role="tab" aria-selected="true" tabindex="0" aria-controls="conversation-column" data-pane="chat">Chat</button><button role="tab" aria-selected="false" tabindex="-1" aria-controls="results-column" data-pane="results">Results</button></div>',
            padding=False,
            js_on_load=(ROOT / "assets/mobile-navigation.js").read_text(),
        )
        c["status"] = markup(
            status_html(
                "Ready when you are. Describe a role to begin."
                if configured
                else "Add OPENAI_API_KEY to .env and restart with uv run --env-file .env python app.py to start a review."
            ),
            elem_id="request-status",
        )
        with gr.Row(elem_id="review-layout", equal_height=False):
            with gr.Column(scale=4, min_width=300, elem_id="conversation-column"):
                markup(
                    '<div class="section-heading"><h2>Your hiring brief</h2><p>Start a search or refine the current requirements.</p></div>'
                )
                c["chatbot"] = gr.Chatbot(
                    value=WELCOME,
                    height=440,
                    elem_id="conversation",
                    show_label=False,
                    layout="bubble",
                    autoscroll=False,
                    buttons=["copy_all"],
                    feedback_options=[],
                    line_breaks=False,
                )
                c["message"] = gr.Textbox(
                    label="Message",
                    show_label=False,
                    placeholder="Describe the role, skills, or experience…",
                    lines=2,
                    max_lines=6,
                    elem_id="composer",
                    max_length=100000,
                )
                with gr.Row(elem_id="composer-actions"):
                    c["reset"] = gr.Button("New review", size="sm", scale=1)
                    c["send"] = gr.Button(
                        "Send message", variant="primary", scale=2, elem_id="send-message"
                    )
                with gr.Accordion("Example requests", open=False):
                    gr.Examples(
                        examples=[
                            ["Find software developers with 3+ years experience"],
                            ["Add React as a must-have"],
                            ["Compare the top 3 candidates"],
                            ["Make TypeScript optional"],
                        ],
                        inputs=c["message"],
                    )
                markup(
                    '<p class="privacy-note">OpenAI interprets requests and reviews the complete text of retrieved resumes. Search, relevance scoring, date calculations and source checks run locally.</p>'
                )
            with gr.Column(scale=7, min_width=320, elem_id="results-column"):
                c["criteria"] = markup(requirements_html({}))
                c["stage"] = markup(stage_html({}))
                c["next_step"] = gr.Button(
                    "Next: detailed screening",
                    elem_id="next-step",
                    variant="primary",
                    visible=False,
                    interactive=False,
                    size="sm",
                )
                c["navigation"] = gr.Radio(
                    VIEWS,
                    value="Candidates",
                    label="Result views",
                    show_label=False,
                    container=False,
                    elem_id="result-navigation",
                )
                with gr.Group(elem_id="candidates-view") as workspace.groups["Candidates"]:
                    c["metrics"] = markup(metrics_html({}, len(workspace.index.candidates)))
                    with gr.Row(elem_id="screening-actions"):
                        c["compare"] = gr.Button("Compare top 3", interactive=False, size="sm")
                        c["deep"] = gr.Button(
                            "Run detailed screening", interactive=False, size="sm"
                        )
                        c["final"] = gr.Button("Get recommendations", interactive=False, size="sm")
                    markup(
                        '<p class="action-guide"><strong>Detailed screening</strong> checks each requirement against source evidence. <strong>Recommendations</strong> decide who advances, identify blockers and show next actions using the completed evidence review. Interview questions are prepared separately.</p>'
                    )
                    c["shortlist"] = gr.HTML(
                        {"html": shortlist_html({}), "selected": None},
                        html_template="${value.html}",
                        js_on_load="element.addEventListener('click', (event) => { const button = event.target.closest('button[data-candidate]'); if (button) { props.value = {...props.value, selected: button.dataset.candidate}; trigger('select'); } });",
                        padding=False,
                        elem_id="candidate-list",
                    )
                    with gr.Accordion("Ranking changes", open=False):
                        c["changes"] = markup(changes_html({}))
                    markup(
                        f'<p class="small-note">{len(workspace.index.candidates)} synthetic profiles · anonymous candidate IDs · experience calculated at the recorded corpus date. Scores show retrieval relevance.</p>'
                    )
                with gr.Group(visible=False, elem_id="compare-view") as workspace.groups["Compare"]:
                    c["comparison"] = markup(comparison_html({}, workspace.index.candidates))
                with gr.Group(visible=False, elem_id="evidence-view") as workspace.groups[
                    "Evidence"
                ]:
                    c["candidate"] = gr.Dropdown(
                        label="Candidate",
                        choices=[],
                        interactive=False,
                        elem_id="evidence-candidate",
                    )
                    c["evidence"] = markup(
                        candidate_evidence_html({}, None, workspace.index.candidates)
                    )
                    gr.Markdown(
                        "Use **Candidates → Run detailed screening** to assess the provisional shortlist, then **Get recommendations** to compare the reviewed options and decide who advances.",
                        elem_classes="small-note",
                    )
                with gr.Group(visible=False, elem_id="recommendations-view") as workspace.groups[
                    "Recommendations"
                ]:
                    c["recommendations"] = gr.HTML(
                        {"html": recommendations_html({}), "selected": None},
                        html_template="${value.html}",
                        js_on_load="element.addEventListener('click', (event) => { const button = event.target.closest('button[data-candidate]'); if (button) { props.value = {...props.value, selected: button.dataset.candidate}; trigger('select'); } });",
                        padding=False,
                        elem_id="decision-board",
                    )
                with gr.Group(visible=False, elem_id="interview-view") as workspace.groups[
                    "Interview"
                ]:
                    c["interview_candidate"] = gr.Dropdown(
                        label="Candidate for interview", choices=[], interactive=False
                    )
                    c["interview"] = gr.Button(
                        "Prepare interview questions", variant="primary", interactive=False
                    )
                    c["questions"] = markup(questions_html({}))
                with gr.Group(visible=False, elem_id="report-view") as workspace.groups["Report"]:
                    markup(
                        '<div class="view-intro"><h2>Screening report</h2><p>Read the current review or download a copy. The JSON file includes the complete conversation, findings, and tool records.</p></div>'
                    )
                    c["download_note"] = gr.Markdown(
                        "Downloads become available after your first search.",
                        elem_classes="small-note",
                    )
                    with gr.Row():
                        c["md_file"] = gr.DownloadButton(
                            "Download report (.md)", interactive=False, variant="primary"
                        )
                        c["json_file"] = gr.DownloadButton(
                            "Download review data (.json)", interactive=False
                        )
                    c["report"] = gr.Markdown(
                        "Run a search to create a screening report.",
                        elem_id="report-preview",
                        sanitize_html=True,
                        line_breaks=False,
                    )
                    c["finish"] = gr.Button("Mark review complete", interactive=False)
                with gr.Group(visible=False, elem_id="activity-view") as workspace.groups[
                    "Activity"
                ]:
                    c["activity"] = markup(activity_html({}))
                    with gr.Accordion("Workflow architecture", open=False):
                        diagram_path = root / "docs/architecture.svg"
                        markup(
                            diagram_path.read_text()
                            if diagram_path.exists()
                            else "Workflow diagram unavailable.",
                            elem_id="workflow-diagram",
                        )
                    with gr.Accordion("Technical record", open=False):
                        c["trace"] = gr.JSON(label="Recorded graph and tool events", value={})
                with gr.Group(visible=False, elem_id="saved-view") as workspace.groups["Saved"]:
                    markup(
                        '<div class="view-intro"><h2>Continue a saved review</h2><p>Each review keeps its own requirements, candidates, and conversation.</p></div>'
                    )
                    c["saved"] = gr.Markdown("Your review ID appears after the first message.")
                    resume_id = gr.Textbox(label="Saved review ID", placeholder="Paste a review ID")
                    c["resume_btn"] = gr.Button("Resume review")
        workspace.actions = [
            c[k]
            for k in [
                "send",
                "reset",
                "compare",
                "deep",
                "final",
                "interview",
                "finish",
                "resume_btn",
                "candidate",
                "interview_candidate",
                "next_step",
            ]
        ]
        outputs = list(c.values()) + list(workspace.groups.values())
        options = {
            "outputs": outputs,
            "concurrency_id": "review-mutations",
            "concurrency_limit": 1,
            "show_progress": "hidden",
            "scroll_to_output": False,
            "trigger_mode": "once",
        }
        inputs = [c["message"], c["session_id"], c["chatbot"], c["candidate"]]
        c["send"].click(workspace.respond, inputs=inputs, api_name="send_request", **options)
        c["message"].submit(workspace.respond, inputs=inputs, **options)

        def action(command):
            def respond(sid, history, selected):
                yield from workspace.respond(command, sid, history, selected)

            return respond

        for key, command in [
            ("compare", "Compare the top 3 candidates side by side"),
            ("deep", "Run deep screening of the top 10"),
            ("final", "Generate final hire or no-hire recommendations"),
            ("finish", "Finish"),
        ]:
            c[key].click(
                action(command),
                inputs=[c["session_id"], c["chatbot"], c["candidate"]],
                api_name=key,
                **options,
            )

        def interview(cid, sid, history):
            if not cid:
                yield {
                    c["status"]: status_html(
                        "Select a candidate before preparing questions.", error=True
                    )
                }
                return
            yield from workspace.respond(
                f"Generate interview questions for {cid}", sid, history, cid
            )

        c["interview"].click(
            interview,
            inputs=[c["interview_candidate"], c["session_id"], c["chatbot"]],
            api_name="interview",
            **options,
        )

        def next_step(sid, history, selected):
            state = workspace.session(sid).snapshot()
            if state.get("round") == "final" or state.get("status") == "complete":
                yield workspace.navigation("Report")
                return
            command = (
                "Generate final hire or no-hire recommendations"
                if state.get("round") == "deep"
                else "Run deep screening of the top 10"
            )
            yield from workspace.respond(command, sid, history, selected)

        c["next_step"].click(
            next_step,
            [c["session_id"], c["chatbot"], c["candidate"]],
            **options,
            api_name="next_step",
        )

        def select_interview(cid, sid):
            return (
                questions_html(workspace.session(sid).snapshot(), cid)
                if sid
                else questions_html({})
            )

        c["interview_candidate"].input(
            select_interview,
            [c["interview_candidate"], c["session_id"]],
            c["questions"],
            show_progress="hidden",
        )
        c["navigation"].input(
            workspace.navigation,
            c["navigation"],
            [*workspace.groups.values(), c["navigation"]],
            queue=False,
            show_progress="hidden",
        )
        c["candidate"].input(
            workspace.select_candidate,
            [c["candidate"], c["session_id"]],
            [c["evidence"]],
            show_progress="hidden",
        )
        c["shortlist"].select(
            workspace.open_candidate,
            [c["shortlist"], c["session_id"]],
            [c["evidence"], c["candidate"], c["navigation"], *workspace.groups.values()],
            show_progress="hidden",
        ).then(
            fn=None,
            inputs=[],
            outputs=[],
            queue=False,
            js="() => { requestAnimationFrame(() => { const nav=document.querySelector('#result-navigation'); if(nav && nav.getBoundingClientRect().top < 0) window.scrollBy({top:nav.getBoundingClientRect().top-16,behavior:'instant'}); }); }",
        )
        c["recommendations"].select(
            workspace.open_candidate,
            [c["recommendations"], c["session_id"]],
            [c["evidence"], c["candidate"], c["navigation"], *workspace.groups.values()],
            show_progress="hidden",
        ).then(
            fn=None,
            inputs=[],
            outputs=[],
            queue=False,
            js="() => { requestAnimationFrame(() => { const nav=document.querySelector('#result-navigation'); if(nav && nav.getBoundingClientRect().top < 0) window.scrollBy({top:nav.getBoundingClientRect().top-16,behavior:'instant'}); }); }",
        )
        c["reset"].click(workspace.new_review, **options, api_name="new_review")
        c["chatbot"].clear(workspace.new_review, **options)
        c["chatbot"].change(
            fn=None, inputs=[], outputs=[], js=JS, queue=False, show_progress="hidden"
        )
        c["resume_btn"].click(workspace.resume, resume_id, **options, api_name="resume_review")
    demo.queue(default_concurrency_limit=1)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open the local candidate review workspace.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--backend", choices=["semantic", "dense", "bm25"], default="semantic")
    args = parser.parse_args()
    create_app(backend=args.backend).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        css=CSS,
        theme=create_theme(),
        footer_links=[],
    )
