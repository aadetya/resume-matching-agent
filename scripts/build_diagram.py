"""Render both connected LangGraph workflows as self-contained offline artifacts."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from types import SimpleNamespace  # noqa: E402

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from matching_agent import build_graph  # noqa: E402
from screening_agent.agent_tools import ToolRegistry  # noqa: E402
from screening_agent.contextual import ContextualReviewer  # noqa: E402
from screening_agent.model_config import DEFAULT_MODEL  # noqa: E402

STEPS = {
    "__start__": (
        "START",
        "A new review",
        "A new conversation enters the compiled graph. Its review ID isolates the saved state.",
        [],
    ),
    "parse_jd": (
        "Parse request",
        "Select the next action",
        "Interpret a job description or follow-up, select an action, and propose complete criteria. Clarification, tool requests, and review actions follow their declared branches.",
        ["messages", "request", "plan"],
    ),
    "extract_requirements": (
        "Extract requirements",
        "Validate proposed criteria",
        "Validate criteria and retain exact user conditions, examples and interpretation assumptions. Preserve unchanged meanings and evidence standards through refinements; compile new or changed meanings before candidate assessment. Keep uncommitted requirements separate until the new search succeeds.",
        ["pending_requirements"],
    ),
    "search_resumes": (
        "Search resumes",
        "Retrieve from the corpus",
        "Search source windows with local dense and lexical retrieval. This step retrieves candidates; it does not use heuristic skill or duration gaps to reject candidates in the connected contextual workflow.",
        ["pending_search_result", "tool_events"],
    ),
    "expand_search": (
        "Expand retrieval",
        "Qwen3 full-resume ranking",
        "Retrieve original passages for each criterion, then score every complete resume with the pinned Qwen3 Reranker 4B. Learned relevance determines the shortlist without a lexical score blend or an early candidate cut. This stage makes no per-candidate OpenAI call and no qualification decision.",
        ["pending_search_result", "tool_events"],
    ),
    "rank_candidates": (
        "Rank candidates",
        "Select ten provisional profiles",
        "Select ten provisional candidates by retrieval relevance. Retain per-criterion passage leads and reserve candidate IDs. Do not infer eligibility, years or strengths. A changed requirement revision clears previous detailed findings and recommendations.",
        ["requirements", "requirements_version", "shortlist", "ranking_changes"],
    ),
    "generate_report": (
        "Generate report",
        "Render current state",
        "Render the current screening findings, comparisons, questions, or error response from stored state. Keep source excerpts and unresolved evidence visible in the conversation.",
        ["report", "screening_report", "messages"],
    ),
    "human_feedback": (
        "Human feedback",
        "Save, pause, and resume",
        "A LangGraph interrupt pauses the conversation. A new message resumes the same review through its checkpoint and returns to request interpretation. The application uses durable SQLite checkpoints.",
        ["checkpoint", "session_id", "pending_nodes"],
    ),
    "compare_candidates": (
        "Compare candidates",
        "Explain the differences",
        "Resolve candidate IDs and compare the available source evidence and current criterion findings. A comparison does not silently change the job requirements.",
        ["comparison", "tool_events"],
    ),
    "interview_questions": (
        "Interview questions",
        "Explicit request after review",
        "On request, prepare candidate-specific questions from evidence gaps and reported work, with follow-ups and assessment guidance.",
        ["questions", "tool_events"],
    ),
    "file_tool": (
        "File tools",
        "Read, list, search, write",
        "Use the inherited hardened filesystem tools. Reads stay inside the project; generated reports are written inside the designated report directory.",
        ["tool_request", "tool_result"],
    ),
    "deep_screen": (
        "Detailed screening",
        "Review source evidence",
        "Run the first full-source qualification assessment on up to ten candidates. Extract and audit work records, assess requirements and useful context, calculate dates locally, then audit evidence. Publish completed and pending records with candidate progress.",
        ["shortlist", "detailed_reviews", "round", "deep_screen_version"],
    ),
    "final_screen": (
        "Recommendations",
        "Decision policy + audited memo",
        "Check completed evidence and derive advance, hold or no-hire boundaries. Generate a cohort memo with comparisons, justified priority groups and decision contingencies. Audit its claims against reviewed facts and frozen requirements before publishing. No full-resume reassessment.",
        ["shortlist", "decision_memo", "round"],
    ),
    "finish_review": (
        "Complete review",
        "Record human completion",
        "An explicit completion request ends this conversation workflow while preserving the saved screening report and its evidence.",
        ["status", "messages"],
    ),
    "__end__": (
        "END",
        "Review complete",
        "The human reviewer explicitly completed this workflow.",
        ["status"],
    ),
}

REVIEW_STEPS = {
    "__start__": (
        "START",
        "Detailed review begins",
        "The nested graph receives a source packet for one candidate and the current frozen criteria. It does not contain a separate human interrupt.",
        ["packet"],
    ),
    "extract_and_audit_work_history": (
        "Prepare work records",
        "Source extraction + audit",
        "Extract and separately audit the complete source inventory within detailed screening. Preserve all employment, source IDs and dates. Reuse these source-only records on later requests; they are never required for initial retrieval.",
        ["packet", "steps"],
    ),
    "assess_requirements": (
        "Assess requirements",
        "Criteria and work observations",
        "Assess the frozen criteria against source-linked work records. Return criterion findings, positive date associations and useful work observations together. Code calculates experience. No questions or recommendations.",
        ["packet", "first", "steps"],
    ),
    "audit_evidence": (
        "Audit final evidence",
        "Source, dates and claim checks",
        "Check the draft criteria, positive duration links, endpoints and observations against the full source and frozen standard. Remove rejected links before local arithmetic. A support-status disagreement remains uncertain with both judgments retained; agreed support can retain a narrative correction. Return the source-bound assessment and explicit disputes.",
        ["packet", "final", "observation_audit", "steps"],
    ),
    "__end__": (
        "END",
        "Return reviewed evidence",
        "Return the nested review state to detailed screening. The outer graph generates the report and waits for human feedback.",
        ["final", "observation_audit", "steps"],
    ),
}

MAIN_POSITIONS = {
    "__start__": (30, 231),
    "__end__": (1230, 702),
    "parse_jd": (90, 190),
    "extract_requirements": (350, 190),
    "search_resumes": (610, 190),
    "expand_search": (870, 190),
    "rank_candidates": (1140, 190),
    "generate_report": (1140, 370),
    "human_feedback": (870, 370),
    "compare_candidates": (90, 555),
    "interview_questions": (300, 555),
    "file_tool": (510, 555),
    "deep_screen": (720, 555),
    "final_screen": (930, 555),
    "finish_review": (1140, 555),
}
REVIEW_POSITIONS = {
    "__start__": (45, 231),
    "extract_and_audit_work_history": (100, 190),
    "assess_requirements": (460, 190),
    "audit_evidence": (820, 190),
    "__end__": (1290, 231),
}
SEARCH_PATH = [
    "__start__",
    "parse_jd",
    "extract_requirements",
    "search_resumes",
    "expand_search",
    "rank_candidates",
    "generate_report",
    "human_feedback",
]
MAIN_ROUTES = {
    "all": ("All transitions", []),
    "search": ("Initial search", SEARCH_PATH),
    "refine": ("Refine requirements", ["human_feedback", *SEARCH_PATH[1:]]),
    "deep": (
        "Detailed screening",
        ["human_feedback", "parse_jd", "deep_screen", "generate_report", "human_feedback"],
    ),
    "final": (
        "Recommendations",
        [
            "human_feedback",
            "parse_jd",
            "deep_screen",
            "final_screen",
            "generate_report",
            "human_feedback",
        ],
    ),
    "finish": ("Complete review", ["human_feedback", "parse_jd", "finish_review", "__end__"]),
}
REVIEW_ROUTES = {
    "all": ("All transitions", []),
    "complete": (
        "Complete evidence",
        [
            "__start__",
            "extract_and_audit_work_history",
            "assess_requirements",
            "audit_evidence",
            "__end__",
        ],
    ),
}


def metadata(graph, steps, positions, routes, title, description):
    if set(graph.nodes) != set(steps) or set(graph.nodes) != set(positions):
        raise ValueError(
            f"Diagram metadata differs from compiled graph: {set(graph.nodes) ^ set(steps)}"
        )
    edges = [
        {"source": edge.source, "target": edge.target, "conditional": edge.conditional}
        for edge in graph.edges
    ]
    actual_edges = {(edge["source"], edge["target"]) for edge in edges}
    for label, path in routes.values():
        for source, target in zip(path, path[1:], strict=False):
            if (source, target) not in actual_edges:
                raise ValueError(f"Highlighted route {label} invents an edge: {source} -> {target}")
    return {
        "title": title,
        "description": description,
        "nodes": {
            node: {
                "title": values[0],
                "subtitle": values[1],
                "detail": values[2],
                "fields": values[3],
            }
            for node, values in steps.items()
        },
        "edges": edges,
        "routes": {key: {"title": value[0], "path": value[1]} for key, value in routes.items()},
    }


NODE_KINDS = {
    "parse_jd": "OpenAI / routing",
    "extract_requirements": "OpenAI / validation",
    "search_resumes": "Local / BGE + BM25",
    "expand_search": "Local / Qwen3 4B",
    "rank_candidates": "Local / selection",
    "generate_report": "Local / presentation",
    "human_feedback": "State / interrupt",
    "compare_candidates": "Local / reviewed state",
    "interview_questions": "OpenAI / generation",
    "file_tool": "Local / filesystem",
    "deep_screen": "Nested graph / 4 workers",
    "final_screen": "OpenAI + local policy",
    "finish_review": "State / completion",
    "extract_and_audit_work_history": "OpenAI / 2 source tasks",
    "assess_requirements": "OpenAI / assessment",
    "audit_evidence": "OpenAI + local validation",
}


def node_svg(node, info, positions, *, review=False):
    x, y = positions[node]
    title = html.escape(info["title"])
    accessible = html.escape(info["title"] + ". " + info["subtitle"], quote=True)
    if node.startswith("__"):
        return f'<g class="node terminal" data-node="{node}" tabindex="0" role="button" aria-label="{accessible}"><circle cx="{x}" cy="{y}" r="20" fill="#334155"/><text x="{x}" y="{y + 4}" fill="white" text-anchor="middle" font-size="9" font-weight="700">{title}</text></g>'
    kind = NODE_KINDS[node]
    color = "#6d28d9" if "OpenAI" in kind else "#0369a1" if "Local" in kind else "#475569"
    bg = "#faf5ff" if "OpenAI" in kind else "#f0f9ff" if "Local" in kind else "#f8fafc"
    width = 260 if review else 180
    size = 13 if len(info["title"]) > 18 else 14
    return f'<g class="node" data-node="{node}" tabindex="0" role="button" aria-label="{accessible}"><title>{html.escape(info["detail"])}</title><rect x="{x}" y="{y}" width="{width}" height="82" rx="5" fill="{bg}" stroke="#cbd5e1"/><rect x="{x}" y="{y}" width="4" height="82" rx="2" fill="{color}"/><text x="{x + 13}" y="{y + 19}" fill="{color}" font-size="9.2">{kind}</text><text x="{x + 13}" y="{y + 41}" fill="#0f172a" font-size="{size}" font-weight="600">{title}</text><text x="{x + 13}" y="{y + 63}" fill="#475569" font-size="9.5">{html.escape(info["subtitle"])}</text></g>'


def edge_path(source, target, positions, *, review=False):
    sx, sy = positions[source]
    tx, ty = positions[target]
    width = 260 if review else 180
    half = width / 2
    if source == "__start__":
        return f"M {sx + 20} {sy} H {tx - 7}"
    if target == "__end__":
        return (
            f"M {sx + width} {sy + 41} H {tx - 27} V {ty}"
            if review
            else f"M {sx + half} {sy + 82} V {ty - 27}"
        )
    if review:
        return f"M {sx + width} {sy + 41} H {tx - 7}"
    if source == "human_feedback":
        return f"M {sx + half} {sy} V 130 H {tx + half} V {ty - 7}"
    if source == "parse_jd" and ty == 555:
        return f"M {sx} {sy + 41} H 65 V 524 H {tx + half} V {ty - 7}"
    if target == "generate_report" and source != "rank_candidates":
        if sy == 555:
            return f"M {sx + half} {sy} V 489 H {tx + half} V {ty + 89}"
        return f"M {sx + half} {sy + 82} V 314 H {tx + half} V {ty - 7}"
    if target == "rank_candidates" and source != "expand_search":
        return f"M {sx + half} {sy + 82} V 337 H {tx + half} V {ty - 7}"
    if sx == tx:
        return f"M {sx + half} {sy + 82} V {ty - 7}"
    if sy == ty:
        return (
            f"M {sx + width} {sy + 41} H {tx - 7}"
            if tx > sx
            else f"M {sx} {sy + 41} H {tx + width + 7}"
        )
    raise ValueError(f"No edge layout for compiled transition {source} -> {target}")


def svg(graph, positions, *, review=False):
    key, height = ("review", 500) if review else ("workflow", 885)
    title = "Detailed screening subgraph" if review else "LangGraph conversation state machine"
    subtitle = (
        "Invocation: deep_screen → ContextualReviewer.review_graph"
        if review
        else "Resume matching agent • Three screening stages • SQLite checkpoints"
    )
    primary = REVIEW_ROUTES["complete"][1] if review else SEARCH_PATH
    emphasis = set(zip(primary, primary[1:], strict=False))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" data-graph="{key}" viewBox="0 0 1410 {height}" role="img" aria-labelledby="{key}-title {key}-desc">',
        f'<title id="{key}-title">{html.escape(graph["title"])}</title><desc id="{key}-desc">{html.escape(graph["description"])}</desc>',
        f'<defs><marker id="{key}-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker></defs>',
        f'<rect width="1410" height="{height}" fill="#ffffff"/>',
        '<g font-family="Arial, sans-serif">',
        f'<text x="60" y="43" fill="#64748b" font-size="11" letter-spacing="1">{"02 / CANDIDATE REVIEW" if review else "01 / CONVERSATION STATE"}</text>',
        f'<text x="60" y="82" fill="#0f172a" font-size="28" font-weight="600">{title}</text>',
        f'<text x="60" y="108" fill="#475569" font-size="13">{subtitle}</text>',
    ]
    if review:
        parts += [
            '<text x="100" y="164" fill="#475569" font-size="11">CANDIDATE SOURCE PACKET AND CURRENT REQUIREMENTS</text>',
            '<text x="100" y="321" fill="#334155" font-size="12">Work records: complete source lines, employment entries, date tokens and record IDs.</text>',
            '<text x="100" y="345" fill="#334155" font-size="12">Validation: exact quotations, source hashes, employment coverage and local interval arithmetic.</text>',
        ]
    else:
        parts += [
            '<text x="90" y="165" fill="#475569" font-size="11">INITIAL SCREENING</text>',
            '<text x="870" y="352" fill="#475569" font-size="11">REPORT AND HUMAN FEEDBACK</text>',
            '<text x="90" y="508" fill="#475569" font-size="11">CONDITIONAL ACTIONS FROM REQUEST ROUTER</text>',
            '<text x="90" y="389" fill="#334155" font-size="12">Input: job description or conversation request</text>',
            '<text x="90" y="413" fill="#334155" font-size="12">Corpus: 200 PDF / DOCX / TXT profiles</text>',
            '<text x="90" y="437" fill="#334155" font-size="12">Initial output: 10 provisional candidates; qualification unassessed</text>',
            '<text x="740" y="679" fill="#475569" font-size="11">Detailed review prerequisite</text>',
            '<text x="740" y="697" fill="#64748b" font-size="10">finalize → deep_screen → final_screen when a review is missing</text>',
        ]
    for edge in graph["edges"]:
        main = (edge["source"], edge["target"]) in emphasis
        dash = ' stroke-dasharray="5 4"' if edge["conditional"] else ""
        path = edge_path(edge["source"], edge["target"], positions, review=review)
        parts.append(
            f'<path class="edge" data-source="{edge["source"]}" data-target="{edge["target"]}" data-conditional="{str(edge["conditional"]).lower()}" d="{path}" fill="none" stroke="{"#0369a1" if main else "#94a3b8"}" stroke-width="{2.2 if main else 1.2}" opacity="{1 if main else 0.7}"{dash} marker-end="url(#{key}-arrow)"/>'
        )
    parts.extend(
        node_svg(node, info, positions, review=review) for node, info in graph["nodes"].items()
    )
    cy = 385 if review else 765
    entries = (
        [
            (100, "SOURCE STATE", "packet.source_inventory"),
            (470, "ASSESSMENT STATE", "first · final · observation_audit"),
            (840, "RETURN", "findings · date bounds · disputes"),
        ]
        if review
        else [
            (90, "REQUIREMENTS", "messages · requirements · requirements_version"),
            (460, "REVIEW STATE", "shortlist · detailed_reviews · decision_memo"),
            (830, "PERSISTENCE", "SQLite checkpoint · interrupt / resume"),
        ]
    )
    for x, label, detail in entries:
        parts.append(
            f'<path d="M {x} {cy} H {x + 340}" stroke="#cbd5e1"/><text x="{x}" y="{cy + 22}" fill="#475569" font-size="10" letter-spacing=".6">{label}</text><text x="{x}" y="{cy + 44}" fill="#334155" font-size="10.5">{detail}</text>'
        )
    parts.append(
        f'<text x="60" y="{height - 22}" fill="#64748b" font-size="11">{len(graph["nodes"])} nodes · {len(graph["edges"])} transitions · Solid: fixed transition / dashed: conditional transition</text>'
    )
    parts.append("</g></svg>")
    return "".join(parts)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Resume matching agent · Architecture</title><style>
*{box-sizing:border-box}body{margin:0;background:#f1f5f9;color:#0f172a;font-family:system-ui,-apple-system,sans-serif}header{padding:24px 34px 18px;display:flex;align-items:center;justify-content:space-between;gap:20px}header strong{font-size:19px;letter-spacing:-.5px}header span{font-size:12px;color:#64748b}button{font:inherit;border:1px solid #cbd5e1;background:#ffffff;color:#334155;border-radius:8px;padding:9px 13px;cursor:pointer;font-size:12px}button:hover{background:#e2e8f0}button:focus-visible,.node:focus-visible{outline:3px solid #2563eb;outline-offset:3px}button.active{background:#1d4ed8;border-color:#1d4ed8;color:white}.tabs{display:flex;gap:8px;margin:0 34px 15px}.tabs button{font-size:13px;padding:11px 16px}.routes{display:flex;gap:7px;flex-wrap:wrap;padding:0 34px 18px}.workspace{display:grid;grid-template-columns:minmax(0,1fr) 292px;gap:20px;padding:0 26px 20px}.canvas{background:#ffffff;border:1px solid #cbd5e1;border-radius:8px;overflow:hidden;align-self:start}.canvas svg{display:block;width:100%;height:auto}.node{cursor:pointer;transition:opacity .18s}.node:hover rect,.node:focus rect{stroke:#2563eb;stroke-width:3}.node.selected rect,.node.selected circle{stroke:#2563eb;stroke-width:3}.edge{transition:opacity .18s}aside{background:#fff;border:1px solid #cbd5e1;border-radius:8px;padding:22px;align-self:start}aside h2{font-family:system-ui,sans-serif;font-size:22px;line-height:1.15;margin:10px 0 14px;font-weight:500}aside p{font-size:13px;line-height:1.65;color:#475569}.label{font-size:10px;letter-spacing:1.25px;color:#64748b;text-transform:uppercase}.tag{display:inline-block;margin:3px 3px 3px 0;padding:5px 7px;background:#f1f5f9;border-radius:5px;font:10px ui-monospace,monospace;overflow-wrap:anywhere}.transitions{display:grid;gap:7px;margin-top:10px}.transitions button{text-align:left;background:#ffffff;padding:8px 10px}.transitions small{display:block;color:#64748b;font-size:10px;margin-top:2px}#open-review{width:100%;margin-top:18px;border-color:#94a3b8;background:#eff6ff}.note{margin:0 34px 16px;max-width:1080px;font-size:12px;color:#475569;line-height:1.6}.legend{display:flex;gap:18px;flex-wrap:wrap;margin:0 34px 25px;color:#64748b;font-size:11px}.legend i{display:inline-block;width:24px;border-top:2px solid #64748b;vertical-align:middle;margin-right:6px}.legend i.conditional{border-top:2px dashed #94a3b8}[hidden]{display:none!important}@media(max-width:1020px){.workspace{grid-template-columns:1fr}aside{display:grid;grid-template-columns:1fr 1fr;gap:10px 24px}aside h2,#step-detail{grid-column:1/-1}header span{max-width:200px;text-align:right}}@media(max-width:700px){header{padding:18px}.tabs,.note,.legend{margin-left:18px;margin-right:18px}.routes{padding-left:18px;padding-right:18px}.workspace{padding:0 12px 20px}.canvas{overflow:auto}.canvas svg{min-width:1000px}aside{display:block}header span{display:none}}
</style></head><body>
<header><strong>Resume matching agent / Architecture</strong><span>Compiled LangGraph • Local diagram</span></header>
<nav class="tabs" aria-label="Diagram views"><button data-view="workflow" class="active" aria-pressed="true">01 Conversation workflow</button><button data-view="review" aria-pressed="false">02 Detailed screening subgraph</button></nav>
<div id="routes" class="routes" aria-label="Highlight a graph route"></div>
<main class="workspace"><div id="workflow-canvas" class="canvas">WORKFLOW_SVG</div><div id="review-canvas" class="canvas" hidden>REVIEW_SVG</div>
<aside aria-live="polite"><div class="label">Selected step</div><h2 id="step-title"></h2><p id="step-detail"></p><div><div class="label">State fields</div><div id="fields"></div></div><div><div class="label" style="margin-top:20px">Outgoing transitions</div><div id="transitions" class="transitions"></div></div><button id="open-review" hidden>Open the detailed review graph →</button></aside></main>
<p id="view-note" class="note"></p><div class="legend"><span><i></i>Highlighted route</span><span><i class="conditional"></i>Conditional transition</span><span>Click a step or an outgoing transition to inspect it. All declared edges remain available.</span></div>
<script>
const bundle=GRAPH_DATA;
let view='workflow',selected={workflow:'rank_candidates',review:'assess_requirements'},route={workflow:'search',review:'all'};
const graph=()=>bundle.graphs[view];
const canvas=()=>document.getElementById(view+'-canvas');
function pick(id){if(!graph().nodes[id])throw Error('Unknown graph node '+id);selected[view]=id;const n=graph().nodes[id];document.getElementById('step-title').textContent=n.title;document.getElementById('step-detail').textContent=n.detail;document.getElementById('fields').replaceChildren(...n.fields.map(f=>{const s=document.createElement('span');s.className='tag';s.textContent=f;return s}));document.getElementById('transitions').replaceChildren(...graph().edges.filter(e=>e.source===id).map(e=>{const b=document.createElement('button');b.textContent=graph().nodes[e.target].title;const s=document.createElement('small');s.textContent=e.conditional?'Conditional transition':'Transition';b.append(s);b.onclick=()=>pick(e.target);return b}));canvas().querySelectorAll('.node').forEach(el=>el.classList.toggle('selected',el.dataset.node===id));document.getElementById('open-review').hidden=!(view==='workflow'&&id==='deep_screen');}
function highlight(key){route[view]=key;const path=graph().routes[key].path;document.querySelectorAll('[data-route]').forEach(b=>b.classList.toggle('active',b.dataset.route===key));canvas().querySelectorAll('.node').forEach(n=>n.style.opacity=key==='all'||path.includes(n.dataset.node)?1:.24);canvas().querySelectorAll('.edge').forEach(e=>{const active=key==='all'||path.some((n,i)=>n===e.dataset.source&&path[i+1]===e.dataset.target);e.style.opacity=active?1:.06;e.style.strokeWidth=active?2.5:1.1;});}
function show(id){view=id;document.querySelectorAll('[data-view]').forEach(b=>{const on=b.dataset.view===id;b.classList.toggle('active',on);b.setAttribute('aria-pressed',String(on));});document.getElementById('workflow-canvas').hidden=id!=='workflow';document.getElementById('review-canvas').hidden=id!=='review';document.getElementById('routes').replaceChildren(...Object.entries(graph().routes).map(([key,r])=>{const b=document.createElement('button');b.dataset.route=key;b.textContent=r.title;b.onclick=()=>highlight(key);return b;}));document.getElementById('view-note').textContent=id==='workflow'?'Initial screening: full-corpus local scoring → ten provisional candidates. No candidate OpenAI assessment runs in this stage. The diagram retains all declared conditional edges; their presence does not mean every branch runs. Detailed screening invokes the detailed review graph.':'Actual nested ContextualReviewer.review_graph, invoked by detailed screening. Source inventory and requirement assessment run here for the first time. The audit checks proposed findings, dates and useful observations. Support-status disputes return as uncertain. Local validation applies date arithmetic and source-integrity checks. The nested graph returns its reviewed state to deep_screen.';pick(selected[id]);highlight(route[id]);}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>show(b.dataset.view));document.querySelectorAll('.canvas .node').forEach(n=>{n.onclick=()=>pick(n.dataset.node);n.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick(n.dataset.node);}};});document.getElementById('open-review').onclick=()=>show('review');show('workflow');
</script></body></html>"""


def mermaid(graphs):
    lines = [
        "%% Both node and edge sets come from the compiled graphs; prefixes avoid ID collisions.",
        "flowchart TB",
    ]
    for key, graph in graphs.items():
        lines.append(f'  subgraph {key}["{graph["title"]}"]')
        for node, info in graph["nodes"].items():
            lines.append(f'    {key}_{node}["{info["title"]}"]')
        for edge in graph["edges"]:
            arrow = "-.->" if edge["conditional"] else "-->"
            lines.append(f"    {key}_{edge['source']} {arrow} {key}_{edge['target']}")
        lines.append("  end")
    return "\n".join(lines) + "\n"


def build():
    # No node is invoked. A temporary root keeps constructor cache setup outside the project.
    with TemporaryDirectory(prefix="matching-diagram-") as scratch:
        reviewer = ContextualReviewer(root=Path(scratch), client=object(), model=DEFAULT_MODEL)
        registry = ToolRegistry(ROOT, None, SimpleNamespace(mode="openai"))
        registry.deep_reviewer = reviewer
        compiled = build_graph(registry, checkpointer=InMemorySaver()).get_graph()
        nested = reviewer.review_graph.get_graph()
        workflow = metadata(
            compiled,
            STEPS,
            MAIN_POSITIONS,
            MAIN_ROUTES,
            "Resume matching: connected conversation workflow",
            "Compiled conversation graph with local initial retrieval, detailed candidate assessment, comparative recommendations and persistent feedback.",
        )
        review = metadata(
            nested,
            REVIEW_STEPS,
            REVIEW_POSITIONS,
            REVIEW_ROUTES,
            "Contextual reviewer: detailed evidence review",
            "Actual nested graph: extract and audit work records, assess requirements and work context, then audit the proposed findings while retaining unresolved disputes.",
        )
    bundle = {
        "schema_version": 2,
        "source": "Compiled build_graph with ContextualReviewer connected, and ContextualReviewer.review_graph",
        "provider_calls_during_generation": 0,
        "workflow_policy": "contextual",
        "graphs": {"workflow": workflow, "review": review},
        "nested_invocation": {
            "outer_node": "deep_screen",
            "graph": "review",
            "kind": "method invocation",
        },
    }
    main_svg, review_svg = svg(workflow, MAIN_POSITIONS), svg(review, REVIEW_POSITIONS, review=True)
    standalone = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1410 1405" role="img" aria-labelledby="architecture-title"><title id="architecture-title">Resume matching conversation and detailed review: two compiled LangGraph workflows</title>'
        + '<rect width="1410" height="1405" fill="white"/><g data-graph="workflow">'
        + main_svg.split(">", 1)[1].removesuffix("</svg>")
        + "</g>"
        + '<g data-graph="review" transform="translate(0,905)">'
        + review_svg.split(">", 1)[1].removesuffix("</svg>")
        + "</g>"
        + "</svg>"
    )
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "architecture.svg").write_text(standalone)
    (docs / "architecture.mmd").write_text(mermaid(bundle["graphs"]))
    (docs / "architecture.json").write_text(json.dumps(bundle, indent=2) + "\n")
    (docs / "architecture.html").write_text(
        PAGE.replace("WORKFLOW_SVG", main_svg)
        .replace("REVIEW_SVG", review_svg)
        .replace("GRAPH_DATA", json.dumps(bundle).replace("</", "<\\/"))
    )
    print(
        json.dumps(
            {
                "workflow": {"nodes": len(workflow["nodes"]), "edges": len(workflow["edges"])},
                "review": {"nodes": len(review["nodes"]), "edges": len(review["edges"])},
                "provider_calls": 0,
                "interactive": "docs/architecture.html",
                "svg": "docs/architecture.svg",
            }
        )
    )


if __name__ == "__main__":
    build()
