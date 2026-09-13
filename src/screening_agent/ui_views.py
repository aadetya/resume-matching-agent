"""Presentation of saved screening facts; no retrieval or decision rules live here."""

from __future__ import annotations

import html
from collections import Counter
from pathlib import Path

from .roles import ROLE_LABELS

VIEWS = [
    "Candidates",
    "Compare",
    "Evidence",
    "Recommendations",
    "Interview",
    "Report",
    "Activity",
    "Saved",
]
COMPONENTS = {
    "retrieval": "Whole-request search",
    "learned_relevance": "Resume relevance",
}
NODES = {
    "parse_jd": "Understand the request",
    "extract_requirements": "Set shared evidence standards",
    "search_resumes": "Search the resume corpus",
    "expand_search": "Find source leads and score resume context",
    "rank_candidates": "Select candidates for review",
    "deep_screen": "Assess and audit the shortlist",
    "final_screen": "Compare reviewed options and check the decision memo",
    "compare_candidates": "Compare selected candidates",
    "interview_questions": "Prepare interview questions",
    "file_tool": "Read or write a file",
    "generate_report": "Prepare the result",
    "human_feedback": "Receive your feedback",
    "finish_review": "Complete the review",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def empty(title: str, description: str) -> str:
    return f'<div class="empty"><h3>{esc(title)}</h3><p>{esc(description)}</p></div>'


def badge(label: str, tone: str = "neutral") -> str:
    return f'<span class="badge {esc(tone)}">{esc(label)}</span>'


def match_status(match: dict, state: dict) -> tuple[str, str]:
    recommendation = match.get("recommendation")
    if recommendation == "hire":
        return "Hire · advance to interview", "positive"
    if recommendation == "hold":
        return "Hold · verify evidence", "warning"
    if recommendation == "no_hire":
        return "No hire · criteria not met", "negative"
    if state.get("round") in {"deep", "final"}:
        review = next(
            (
                r
                for r in state.get("detailed_reviews", [])
                if r["candidate_id"] == match["candidate_id"]
            ),
            None,
        )
        if review and (
            not review["packet"]["coverage"]["complete"]
            or any(c["mandatory"] and c["review_status"] != "supported" for c in review["changes"])
        ):
            return "Needs verification", "warning"
        mandatory = [
            a
            for a in (match.get("reviewed_assessments") or match.get("assessments", []))
            if a.get("mandatory")
        ]
        if not mandatory or any(a["status"] != "supported" for a in mandatory):
            return "Needs verification", "warning"
        return "Evidence reviewed", "positive"
    return "Provisional · unverified", "neutral"


def requirements_html(state: dict) -> str:
    req = state.get("requirements") or {}
    required = [ROLE_LABELS.get(req["role"], req["role"])] if req.get("role") else []
    required += [" / ".join(g) for g in req.get("must_have", [])]
    if req.get("minimum_years"):
        required.append(f"{req['minimum_years']:g}+ years overall")
    required += [f"{v:g}+ years {k}" for k, v in req.get("skill_years", {}).items()]
    pills = "".join(f'<span class="pill">{esc(x)}</span>' for x in required)
    pills += "".join(
        f'<span class="pill preferred">{esc(x)} · preferred</span>'
        for x in req.get("nice_to_have", [])
    )
    return (
        f'<div class="criteria"><div class="eyebrow">CURRENT REQUIREMENTS · REVISION {state.get("requirements_version", 0)}</div>'
        + (pills or '<span class="muted">Describe a role, skills, or experience to begin.</span>')
        + "".join(f"<p>Needs clarification: {esc(x)}</p>" for x in req.get("unresolved", []))
        + (
            "<details class=score-details><summary>How these requirements will be assessed</summary><p>The same evidence standard applies to every candidate in this revision.</p>"
            + "".join(
                f"<article><h4>{esc(s['capability'])}</h4><p>{esc(s['sufficient_evidence'])}</p><p><strong>Equivalent evidence:</strong> {esc(s['equivalence_boundary'])}</p><p><strong>When evidence is unclear:</strong> {esc(s['uncertainty_boundary'])}</p></article>"
                for s in req["evidence_standards"]
            )
            + "</details>"
            if req.get("evidence_standards")
            else ""
        )
        + "</div>"
    )


def stage_html(state: dict) -> str:
    if not state.get("requirements"):
        title, detail = (
            "Ready for a new search",
            "Start in the conversation. Results will appear here.",
        )
    elif state.get("status") == "complete":
        title, detail = (
            "Review complete",
            "The report is saved. Start a new review for another role.",
        )
    elif state.get("round") == "final":
        title, detail = (
            "Recommendations ready",
            "Open Recommendations for decisions, blockers and next actions. The detailed evidence remains in Evidence.",
        )
    elif state.get("round") == "deep":
        title, detail = (
            "Detailed screening complete",
            "Open Evidence for the requirement findings, work history and gaps. Continue to Recommendations for decisions and priorities.",
        )
    else:
        title, detail = (
            "Initial shortlist ready",
            "Compare candidates or run detailed screening to check each requirement against the source.",
        )
    audit = state.get("search_result", {}).get("retrieval_audit", {})
    pending = audit.get("pending_count", 0) + sum(
        r.get("stage") == "pending" for r in state.get("detailed_reviews", [])
    )
    if pending:
        title = "Provisional review · assessments pending"
        detail = f"{pending} candidate assessments remain incomplete. Retry the relevant screening step; pending profiles are not rejected."
    return f'<div class="stage-summary"><span class="stage-dot" aria-hidden="true"></span><div><strong>{esc(title)}</strong><p>{esc(detail)}</p></div></div>'


def metrics_html(state: dict, corpus_size: int) -> str:
    searched = bool(state.get("requirements"))
    shortlisted = len(state.get("shortlist", []))
    remaining = max(0, corpus_size - shortlisted) if searched else 0
    pending = (
        state.get("search_result", {}).get("retrieval_audit", {}).get("pending_assessments", [])
    )
    values = [
        (corpus_size, "Profiles"),
        (shortlisted, "Shortlisted"),
        (
            sum(r.get("stage") == "deep" for r in state.get("detailed_reviews", [])),
            "Detailed reviews",
        ),
    ]
    return (
        '<div class="metrics">'
        + "".join(
            f"<div><strong>{value}</strong><span>{label}</span></div>" for value, label in values
        )
        + "</div>"
        + (
            '<details class="callout"><summary>Pending candidate assessments</summary><p>These profiles remain unassessed after a processing failure. Retry the search.</p><ul>'
            + "".join(
                "<li>"
                + esc(r.get("name", r["candidate_id"]))
                + " · "
                + esc(r["candidate_id"])
                + "</li>"
                for r in pending
            )
            + "</ul></details>"
            if pending
            else ""
        )
        + (
            f'<p class="small-note">{remaining} other profiles were not shortlisted. They have not received a detailed qualification decision.</p>'
            if remaining
            else ""
        )
    )


def shortlist_html(state: dict) -> str:
    matches = state.get("shortlist", [])
    if not matches:
        return empty(
            "Your shortlist starts here", "Describe the role to select candidates for review."
        )
    stage = state.get("round", "initial")
    titles = {
        "initial": (
            "Candidates to review",
            "A provisional selection from the full collection. Open a profile for the passages behind its selection.",
        ),
        "deep": (
            "Reviewed candidates",
            "Requirement findings now determine the order. Open a profile to inspect its work history and supporting evidence.",
        ),
        "final": (
            "Decision shortlist",
            "The screening decision and next action for each reviewed candidate. Open Recommendations for comparisons and priorities.",
        ),
    }
    title, description = titles.get(stage, titles["initial"])
    rows = [
        f'<div class="view-intro stage-intro {esc(stage)}"><h2>{title}</h2><p>{description}</p></div>'
    ]
    for rank, match in enumerate(matches, 1):
        label, tone = match_status(match, state)
        ledger = match.get("reviewed_assessments") or match.get("assessments", [])
        mandatory = [a for a in ledger if a.get("mandatory")]
        supported = sum(a["status"] == "supported" for a in mandatory)
        metric = (
            f"<strong>{match['score']:.1f}</strong><span>retrieval points</span>"
            if stage == "initial"
            else f"<strong>{supported}/{len(mandatory)}</strong><span>required criteria supported</span>"
        )
        rows.append(
            f'<article class="candidate-card stage-{esc(stage)}"><div class="candidate-top"><span class="rank">{rank:02}</span><div class="candidate-name"><h3>{esc(match["name"])}</h3>{badge(label, tone)}</div><div class="score">{metric}</div></div>'
        )
        if stage == "initial":
            leads = match.get("retrieval_leads", [])
            rows.append(
                '<div class="candidate-criteria">'
                + "".join(f'<span class="pill">{esc(r["criterion"])}</span>' for r in leads)
                + "</div>"
            )
            source = next(iter(match.get("evidence", [])), {}).get("quote", "")
            if source:
                rows.append(
                    '<details class="excerpt"><summary>Preview source passage</summary><p class="source-preview">'
                    + esc(source)
                    + "</p></details>"
                )
        elif stage == "final":
            rows.append(
                '<p class="candidate-finding">'
                + esc(match.get("recommendation_reason", "Decision awaiting review."))
                + "</p>"
            )
            if match.get("recommendation_actions"):
                rows.append(
                    '<p class="gap-note"><strong>Next:</strong> '
                    + esc(match["recommendation_actions"][0])
                    + "</p>"
                )
        else:
            rows.append(
                '<p class="candidate-finding">'
                + esc(
                    next(
                        iter(match.get("strengths", [])),
                        "No supported strength recorded in the completed findings.",
                    )
                )
                + "</p>"
            )
            if match.get("gaps"):
                rows.append(
                    '<p class="gap-note"><strong>Unresolved:</strong> '
                    + esc(match["gaps"][0])
                    + "</p>"
                )
        rows.append(
            f'<button class="text-button" data-candidate="{esc(match["candidate_id"])}" aria-label="View evidence for {esc(match["name"])}">{"Open profile" if stage == "initial" else "Read reviewed evidence"} <span aria-hidden="true">→</span></button></article>'
        )
    return '<div class="candidate-list">' + "".join(rows) + "</div>"


def source_html(evidence: dict) -> str:
    """Preserve the exact source text, including line breaks; escape all markup."""
    return (
        '<figure class="source"><blockquote>' + esc(evidence.get("quote", "")) + "</blockquote>"
        f"<figcaption>{esc(Path(evidence.get('source_path', '')).name)} · characters {evidence.get('start', 0)}–{evidence.get('end', 0)}</figcaption>"
        '<details class="source-meta"><summary>Source details</summary>'
        f"<p>{esc(evidence.get('source_path', ''))}</p><p>Document checksum: <code>{esc(evidence.get('document_sha256', ''))}</code></p></details></figure>"
    )


def findings_html(items: list[str], fallback: str) -> str:
    return (
        '<ul class="findings">' + "".join(f"<li>{esc(x)}</li>" for x in items) + "</ul>"
        if items
        else f'<p class="muted">{esc(fallback)}</p>'
    )


def detailed_review_markdown(review: dict) -> str:
    if review.get("stage") == "pending":
        return (
            "### Detailed assessment pending\n\nThe source assessment did not complete. Retry detailed screening. This candidate remains unresolved; the processing failure is not evidence against their qualifications.\n\nDiagnostic record: "
            + review["failure_record"]
        )
    evidence = {}

    def references(rows: list[dict]) -> str:
        labels = []
        for row in rows:
            key = (row["source_path"], row["start"], row["end"])
            if key not in evidence:
                evidence[key] = (f"E{len(evidence) + 1}", row)
            labels.append(evidence[key][0])
        return (
            "Source evidence: " + ", ".join(dict.fromkeys(labels))
            if labels
            else "No supporting passage identified."
        )

    lines = [
        "### Detailed evidence assessment",
        f"Review mode: {review['mode']} · model: {review['model'] or 'not used'}.",
        "Full-source requirement assessment, employment calculations and independent evidence audit. Initial screening supplied retrieval leads only.",
    ]
    lines.append("#### Work and results")
    for item in review["observations"]:
        label = {
            "contribution": "Personal work",
            "reported_outcome": "Reported result",
            "limitation": "Evidence limitation",
        }.get(item["kind"], item["kind"].replace("_", " ").capitalize())
        lines += ["**" + label + ":** " + item["claim"]]
        lines.append(references(item["evidence"]))
    lines.append("#### Requirement findings")
    for row in review["criteria"]:
        change = next(c for c in review["changes"] if c["criterion"] == row["criterion"])
        prefix = (
            "Unconfirmed model interpretation: "
            if row["status"] == "supported" and change["review_status"] != "supported"
            else ""
        )
        lines += [
            f"**{row['criterion']} · {change['review_status']} (model category: {row['evidence_depth']})**",
            prefix + row["finding"],
        ]
        lines.append(references(row["evidence"]))
    for scope, title in [("employment", "Employment history"), ("project", "Projects")]:
        units = [
            u for u in review.get("evidence_graph", {}).get("work_units", []) if u["scope"] == scope
        ]
        if not units:
            continue
        lines.append("#### " + title)
        for unit in sorted(units, key=lambda u: u.get("start_date") or "", reverse=True):
            start = unit.get("start_date") or "Start date not stated"
            end = (
                "Present"
                if unit.get("end_kind") == "present"
                else unit.get("end_date") or "End date not stated"
            )
            lines.extend(
                [
                    "**" + unit["title"] + "**",
                    start + " to " + end,
                    references(unit.get("evidence", [])),
                ]
            )
    if evidence:
        lines.append("#### Source evidence for this detailed review")
        for label, source in evidence.values():
            lines.extend([f"**{label}**", quote_markdown(source)])
    coverage = review["packet"]["coverage"]
    lines += [
        f"Review coverage: {coverage['reviewed_windows']}/{coverage['available_windows']} windows; {coverage['nonspace_character_fraction']:.0%} of extracted non-whitespace text. Full coverage: {coverage['complete']}.",
        "Citations verify source location, not entailment or the truth of a resume assertion.",
    ]
    return "\n\n".join(lines)


def contextual_duration_text(duration: dict) -> str:
    lower, upper = duration["months"], duration.get("maximum_months")
    if lower is None:
        return "Duration unresolved"
    if upper is None:
        detail = (
            "some dates unknown"
            if duration.get("measurable_dates_complete") is False
            else "coverage incomplete"
        )
        return f"At least {lower} months · {detail}"
    if upper != lower:
        return f"{lower}–{upper} months"
    return f"{lower} months"


def review_section(title: str, body: str, *, opened: bool = False, note: str = "") -> str:
    return (
        f'<details class="review-section"{" open" if opened else ""}><summary><span>{esc(title)}</span>'
        + (f"<small>{esc(note)}</small>" if note else "")
        + '</summary><div class="section-content">'
        + body
        + "</div></details>"
    )


def work_history_html(review: dict, *, projects: bool = False) -> str:
    scope = "project" if projects else "employment"
    units = [u for u in review["evidence_graph"]["work_units"] if u["scope"] == scope]
    # Presentation order only: retain every work record and its original evidence.
    units.sort(key=lambda unit: unit.get("start_date") or "", reverse=True)
    if not units:
        return (
            '<p class="muted">No '
            + ("project entries" if projects else "employment entries")
            + " were identified in this resume.</p>"
        )
    rows = ['<ol class="work-history">']
    for unit in units:
        start = unit.get("start_date") or "Start date not stated"
        end = (
            "Present"
            if unit.get("end_kind") == "present"
            else unit.get("end_date") or "End date not stated"
        )
        dates = (
            f"{start} – {end}"
            if not projects or unit.get("start_date") or unit.get("end_date")
            else "Project dates not stated"
        )
        related = [c["criterion"] for c in review["criteria"] if unit["unit_id"] in c["unit_ids"]]
        rows.append(
            '<li><article class="work-entry"><div class="work-period">'
            + esc(dates)
            + "</div><h4>"
            + esc(unit["title"])
            + "</h4>"
        )
        if related:
            rows.append(
                '<p class="small-note"><strong>Reviewed against:</strong> '
                + esc(" · ".join(related))
                + "</p>"
            )
        rows.append(
            "<details><summary>Responsibilities and original source</summary>"
            + "".join(source_html(e) for e in unit.get("evidence", []))
            + "</details></article></li>"
        )
    return "".join(rows) + "</ol>"


def requirement_findings_html(review: dict) -> str:
    units = {u["unit_id"]: u for u in review["evidence_graph"]["work_units"]}
    labels = {
        "supported": "Supported",
        "uncertain": "Needs clarification",
        "not_demonstrated": "Not demonstrated",
    }
    pieces = []
    for row in review["criteria"]:
        tone = "positive" if row["status"] == "supported" else "warning"
        pieces.append(
            f'<article class="criterion {tone}"><div class="criterion-heading"><h4>{esc(row["criterion"])}</h4>{badge(labels[row["status"]], tone)}</div><p>{esc(row["finding"])}</p>'
        )
        duration = row.get("duration")
        if duration:
            pieces.append(
                f'<div class="duration-summary"><strong>{contextual_duration_text(duration)}</strong><span>{esc(duration["basis"].replace("_", " "))} · as of {esc(duration["snapshot_date"])}</span></div>'
            )
        pieces.append("<details><summary>Work context and supporting excerpts</summary>")
        for uid in row["unit_ids"]:
            unit = units[uid]
            credit = (
                (
                    "Counted toward this duration"
                    if uid in duration["unit_ids"]
                    else "Context only · no time credited"
                )
                if duration
                else "Related source context"
            )
            pieces.append(
                '<div class="relation-card"><strong>'
                + esc(unit["title"])
                + '</strong><p class="small-note">'
                + esc(credit)
                + "</p></div>"
            )
            association = next(
                (a for a in row.get("duration_associations", []) if a["unit_id"] == uid), {}
            )
            if association.get("temporal_scope") == "dated_usage":
                pieces.append(
                    '<p class="small-note">Counted usage: '
                    + esc(association["usage_start_date"])
                    + " – "
                    + esc(association["usage_end_date"])
                    + "</p>"
                )
        pieces.extend(source_html(e) for e in row["evidence"])
        pieces.append(
            '<p class="small-note">Evidence: '
            + esc(row["evidence_depth"].replace("_", " "))
            + " · "
            + esc(row["relationship"])
            + " relationship.</p></details></article>"
        )
    return "".join(pieces)


def contextual_review_html(review: dict) -> str:
    if review.get("stage") == "pending":
        return '<div class="callout warning"><strong>Review incomplete</strong><p>Retry detailed screening to finish this candidate’s assessment. A processing failure does not establish a qualification gap.</p></div>'
    counts = Counter(c["status"] for c in review["criteria"])
    pieces = [
        '<div class="review-totals">'
        + "".join(
            f'<span class="{tone}"><strong>{counts[key]}</strong> {label}</span>'
            for key, label, tone in [
                ("supported", "supported", "positive"),
                ("uncertain", "to clarify", "warning"),
                ("not_demonstrated", "not demonstrated", "negative"),
            ]
        )
        + "</div>"
    ]
    pieces.append(
        review_section(
            "Requirements",
            requirement_findings_html(review),
            opened=True,
            note="Findings and source evidence",
        )
    )
    observations = []
    for row in review.get("observations", []):
        label = {
            "contribution": "Work described",
            "reported_outcome": "Reported result",
            "limitation": "Evidence limit",
        }[row["kind"]]
        observations.append(
            '<article class="work-observation"><span class="eyebrow">'
            + label
            + "</span><p>"
            + esc(row["claim"])
            + "</p><details><summary>Read the supporting source</summary>"
            + "".join(source_html(e) for e in row["evidence"])
            + '<p class="small-note">'
            + esc(row["source_check"])
            + "</p></details></article>"
        )
    if observations:
        pieces.append(
            review_section(
                "Work and results",
                "".join(observations),
                note=f"{len(observations)} source-backed observations",
            )
        )
    pieces.append(
        review_section(
            "Employment history",
            work_history_html(review),
            note="Roles, dates and responsibilities",
        )
    )
    if any(u["scope"] == "project" for u in review["evidence_graph"]["work_units"]):
        pieces.append(
            review_section(
                "Projects",
                work_history_html(review, projects=True),
                note="Project context and source excerpts",
            )
        )
    notes = []
    for dispute in review.get("disputes", []):
        notes.append(
            '<article class="criterion"><h4>Unresolved interpretation</h4><p><strong>Assessment:</strong> '
            + esc(dispute["initial"]["finding"])
            + "</p><p><strong>Source audit:</strong> "
            + esc(dispute["audit"]["reason"])
            + "</p><p><strong>Agreed standard:</strong> "
            + esc(dispute["audit"].get("standard_clause", ""))
            + "</p></article>"
        )
    for correction in review.get("corrections", []):
        notes.append(
            '<article class="criterion"><h4>Explanation corrected</h4><p><strong>Earlier:</strong> '
            + esc(correction["initial"]["finding"])
            + "</p><p><strong>Checked:</strong> "
            + esc(correction["audit"]["reason"])
            + "</p></article>"
        )
    inventory = review["packet"].get("source_inventory", {})
    notes.append(
        "<p><strong>Employment extraction:</strong> "
        + ("Complete" if inventory.get("employment_complete") else "Needs review")
        + ".</p><p>"
        + esc(inventory.get("check", {}).get("reason", ""))
        + "</p>"
    )
    notes.append(
        '<ol class="review-trace">'
        + "".join("<li>" + esc(step.replace("_", " ")) + "</li>" for step in review["review_steps"])
        + "</ol>"
    )
    notes.append(
        f'<p class="small-note">{esc(review["model"])} · {review["usage"].get("input_tokens", 0):,} input tokens · {review["usage"].get("output_tokens", 0):,} output tokens · {review.get("cache_hits", 0)} cached steps. Source positions and date arithmetic are checked in code. A source citation does not independently verify the resume’s claims.</p>'
    )
    pieces.append(
        review_section(
            "Review notes",
            "".join(notes),
            opened=bool(review.get("disputes")),
            note="Source checks and unresolved interpretations",
        )
    )
    return "".join(pieces)


def candidate_evidence_html(state: dict, candidate_id: str | None, candidates: dict) -> str:
    match = next((m for m in state.get("shortlist", []) if m["candidate_id"] == candidate_id), None)
    if not match:
        return empty("Inspect a candidate", "Choose a shortlisted candidate to see their evidence.")
    label, tone = match_status(match, state)
    review = next(
        (r for r in state.get("detailed_reviews", []) if r["candidate_id"] == candidate_id), None
    )
    stage_label = "REVIEWED PROFILE" if review else "INITIAL SCREENING"
    heading = f'<section class="detail-panel"><div class="detail-heading"><div><div class="eyebrow">{stage_label}</div><h2>{esc(match["name"])}</h2></div>{badge(label, tone)}</div>'
    if review:
        return heading + contextual_review_html(review) + "</section>"
    pieces = [
        heading,
        '<p class="review-summary">These passages explain why this profile was retrieved. Detailed screening will examine the full resume and verify the work behind each requirement.</p>',
        '<div class="callout"><strong>Candidate qualification not assessed</strong><p>The initial screen identifies relevant source material. Experience dates and requirement support remain unverified.</p></div>',
    ]
    for number, lead in enumerate(match.get("retrieval_leads", [])):
        body = (
            "".join(source_html(e) for e in lead.get("evidence", []))
            or '<p class="muted">No separate source lead available.</p>'
        )
        pieces.append(
            review_section(
                lead["criterion"],
                body,
                opened=number == 0,
                note="Source lead · awaiting detailed review",
            )
        )
    if not match.get("retrieval_leads"):
        pieces.append(
            review_section(
                "Source passages",
                "".join(source_html(e) for e in match.get("evidence", [])),
                opened=True,
            )
        )
    components = "".join(
        '<div class="score-component"><span>'
        + esc(COMPONENTS.get(key, key))
        + "</span><strong>"
        + f"{value:.2f}"
        + "</strong></div>"
        for key, value in match.get("components", {}).items()
    )
    pieces.append(
        review_section(
            "Retrieval score",
            f"<p>{match['score']:.1f} / 100 retrieval points. This is not a match probability.</p>"
            + components,
        )
    )
    return "".join(pieces) + "</section>"


def decision_memo_html(state: dict) -> str:
    memo = state.get("decision_memo")
    if not memo:
        return ""
    names = {m["candidate_id"]: m["name"] for m in state.get("shortlist", [])}
    pieces = [
        '<section class="detail-panel"><div class="eyebrow">RECOMMENDATIONS</div><h2>Recommendation summary</h2><p>'
        + esc(memo["summary"]["text"].replace("no_hire", "no hire"))
        + "</p>"
    ]
    if memo["advance_tiers"]:
        pieces.append("<h4>Interview priority</h4>")
        for n, tier in enumerate(memo["advance_tiers"], 1):
            pieces.append(
                f"<p><strong>Priority {n}:</strong> "
                + esc(", ".join(names[cid] for cid in tier))
                + (" · no justified preference within this group" if len(tier) > 1 else "")
                + "</p>"
            )
    comparisons = []
    for pair in memo["comparisons"]:
        comparisons.append(
            '<article class="criterion"><h4>'
            + esc(" / ".join(names[cid] for cid in pair["candidate_ids"]))
            + "</h4><p>"
            + esc(pair["conclusion"]["text"])
            + "</p><details><summary>Reviewed facts behind this comparison</summary>"
        )
        for key in pair["conclusion"]["fact_ids"]:
            fact = memo["facts"][key]
            comparisons.append(
                "<p><strong>"
                + esc(names.get(fact["candidate_id"], "Reviewed cohort"))
                + ":</strong> "
                + esc(fact.get("finding", fact.get("claim", "")))
                + "</p>"
            )
        comparisons.append("</details></article>")
    if comparisons:
        pieces.append(
            review_section(
                "Why these candidates differ",
                "".join(comparisons),
                note=f"{len(memo['comparisons'])} comparisons using reviewed facts",
            )
        )
    pieces.append(
        f'<p class="small-note">{len(memo["audit"]["claims"])} memo claims checked against detailed findings. Priority groups checked against the requested criteria. No additional resume assessment.</p></section>'
    )
    return "".join(pieces)


def recommendations_html(state: dict) -> str:
    """Present saved decisions; this view neither scores nor decides."""
    if state.get("round") != "final":
        return empty(
            "Screening decisions appear here",
            "Complete detailed screening, then select Get recommendations. Evidence findings remain in Evidence; this view shows the decision and next action for each candidate.",
        )
    matches = state.get("shortlist", [])
    counts = Counter(m.get("recommendation") for m in matches)
    pieces = [
        decision_memo_html(state),
        '<section class="decision-board"><div class="view-intro"><div class="eyebrow">FINAL SCREENING · REQUIREMENTS V'
        + esc(state.get("requirements_version", 0))
        + "</div><h2>Decisions and next steps</h2><p>Decisions use the completed evidence review. A positive screening recommendation means advance to a human interview.</p></div>",
        '<div class="metrics decision-counts">'
        + "".join(
            f'<div class="{tone}"><strong>{counts[key]}</strong><span>{label}</span></div>'
            for key, label, tone in [
                ("hire", "Advance to interview", "positive"),
                ("hold", "Hold for evidence", "warning"),
                ("no_hire", "Criteria not met", "negative"),
            ]
        )
        + "</div>",
    ]
    for decision, title in [
        ("hire", "Advance to interview"),
        ("hold", "Resolve evidence before advancing"),
        ("no_hire", "Do not advance on the current evidence"),
    ]:
        group = [m for m in matches if m.get("recommendation") == decision]
        if not group:
            continue
        group_cards = []
        for m in group:
            ledger = m.get("reviewed_assessments") or m.get("assessments", [])
            mandatory = [a for a in ledger if a["mandatory"]]
            supported = sum(a["status"] == "supported" for a in mandatory)
            blockers = [a for a in mandatory if a["status"] != "supported"]
            label, tone = match_status(m, state)
            group_cards.append(
                f'<article class="decision-card {tone}" data-decision="{esc(decision)}"><div class="detail-heading"><h3>{esc(m["name"])}</h3>{badge(label, tone)}</div><p class="decision-proof">{supported} / {len(mandatory)} mandatory criteria supported</p><p class="decision-reason">{esc(m.get("recommendation_reason", ""))}</p>'
            )
            if blockers:
                group_cards.append(
                    '<div class="decision-blockers"><h4>What prevents advancement</h4>'
                    + findings_html([a["criterion"] + ": " + a["reason"] for a in blockers], "")
                    + "</div>"
                )
            group_cards.append(
                "<h4>Next actions</h4>"
                + findings_html(
                    m.get("recommendation_actions", []),
                    "Review the decision and source evidence before proceeding.",
                )
            )
            group_cards.append(
                '<details><summary>Decision basis</summary><ul class="findings">'
                + "".join(
                    f"<li><strong>{esc(a['criterion'])}</strong> · {esc(a['status'].replace('_', ' '))}</li>"
                    for a in mandatory
                )
                + '</ul><p class="small-note">Preferred criteria affect ordering; they do not block advancement. Screening points do not determine this decision.</p></details>'
            )
            group_cards.append(
                f'<button class="text-button" data-candidate="{esc(m["candidate_id"])}">Inspect detailed evidence →</button></article>'
            )
        pieces.append(
            review_section(
                title,
                "".join(group_cards),
                opened=decision != "no_hire",
                note=f"{len(group)} candidates",
            )
        )
    pieces.append("</section>")
    return "".join(pieces)


def comparison_rows(state: dict) -> list[dict]:
    comparison = state.get("comparison", {})
    direct = {r["candidate_id"]: r for r in comparison.get("candidates", [])}
    ranked = {r["candidate_id"]: r for r in state.get("shortlist", [])}
    ids = comparison.get("candidate_ids") or list(direct) or list(ranked)[:3]
    return [
        {**direct.get(cid, {}), **ranked.get(cid, {})}
        for cid in ids
        if cid in direct or cid in ranked
    ]


def comparison_table(rows: list[dict], cells: list[tuple[str, list[str]]]) -> str:
    head = "".join(f'<th scope="col">{esc(r["name"])}</th>' for r in rows)
    body = "".join(
        '<tr><th scope="row">'
        + esc(label)
        + "</th>"
        + "".join("<td>" + value + "</td>" for value in values)
        + "</tr>"
        for label, values in cells
    )
    return (
        '<div class="table-scroll" role="region" aria-label="Candidate comparison" tabindex="0"><table class="comparison-table"><thead><tr><th scope="col">Compare</th>'
        + head
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def comparison_html(state: dict, candidates: dict) -> str:
    rows = comparison_rows(state)
    if not rows:
        return empty(
            "Compare candidates side by side",
            "Run an initial search. The current top three will appear here; you can also request specific candidates in the conversation.",
        )
    initial = state.get("round", "initial") == "initial"
    explicit = bool(
        state.get("comparison", {}).get("candidate_ids")
        or state.get("comparison", {}).get("candidates")
    )
    description = (
        "Your selected candidates" if explicit else "The current top three"
    ) + f" · requirements revision {state.get('requirements_version', 0)}."
    description += (
        " These are retrieval leads; qualifications have not been assessed."
        if initial
        else " This comparison uses the completed requirement findings. It makes no additional model calls."
    )
    pieces = [
        '<div class="view-intro"><h2>Candidate comparison</h2><p>' + esc(description) + "</p></div>"
    ]
    ledgers = [r.get("reviewed_assessments") or r.get("assessments", []) for r in rows]
    overview = [("Review status", [badge(*match_status(r, state)) for r in rows])]
    if initial:
        overview.append(
            (
                "Retrieval points",
                [
                    f"{r['score']:.1f} / 100" if "score" in r else "Not ranked in this search"
                    for r in rows
                ],
            )
        )
    else:
        overview.append(
            (
                "Required criteria supported",
                [
                    f"{sum(a['status'] == 'supported' for a in ledger if a['mandatory'])} / {sum(a['mandatory'] for a in ledger)}"
                    if ledger
                    else "Review incomplete"
                    for ledger in ledgers
                ],
            )
        )
    if state.get("round") == "final":
        overview.append(
            (
                "Decision reason",
                [esc(r.get("recommendation_reason", "No decision recorded")) for r in rows],
            )
        )
    pieces.append(comparison_table(rows, overview))
    criteria = list(dict.fromkeys(a["criterion"] for ledger in ledgers for a in ledger))
    if criteria:
        findings = []
        labels = {
            "supported": "Supported",
            "uncertain": "Needs clarification",
            "not_demonstrated": "Not demonstrated",
        }
        for criterion in criteria:
            values = []
            for ledger in ledgers:
                row = next((a for a in ledger if a["criterion"] == criterion), None)
                if row:
                    values.append(
                        '<span class="cell-status">'
                        + badge(
                            labels.get(row["status"], row["status"]),
                            "positive" if row["status"] == "supported" else "warning",
                        )
                        + "</span>"
                        + esc(row["reason"])
                    )
                else:
                    values.append("Not assessed for this candidate")
            findings.append((criterion, values))
        pieces.append(
            review_section("Requirement findings", comparison_table(rows, findings), opened=True)
        )
    elif initial:
        criteria = list(
            dict.fromkeys(a["criterion"] for r in rows for a in r.get("retrieval_leads", []))
        )
        findings = []
        for criterion in criteria:
            values = []
            for r in rows:
                lead = next(
                    (a for a in r.get("retrieval_leads", []) if a["criterion"] == criterion), {}
                )
                values.append(
                    "".join(source_html(e) for e in lead.get("evidence", []))
                    or "No separate passage lead available"
                )
            findings.append((criterion, values))
        if findings:
            pieces.append(
                review_section(
                    "Source leads",
                    comparison_table(rows, findings),
                    note="Unverified passages for each requested condition",
                )
            )
    reviews = {r["candidate_id"]: r for r in state.get("detailed_reviews", [])}
    if not initial:
        work = [
            (
                "Work and results",
                [
                    findings_html(
                        [
                            o["claim"]
                            for o in reviews.get(r["candidate_id"], {}).get("observations", [])
                        ],
                        "No reviewed observations available",
                    )
                    for r in rows
                ],
            )
        ]
        pieces.append(review_section("Work and results", comparison_table(rows, work)))
    if initial:
        components = sorted(set().union(*(r.get("components", {}) for r in rows)))
        scoring = [
            (
                COMPONENTS.get(k, k),
                [
                    f"{r['components'][k]:.2f}" if k in r.get("components", {}) else "Not ranked"
                    for r in rows
                ],
            )
            for k in components
        ]
        pieces.append(
            review_section(
                "How the retrieval points were calculated",
                '<p class="small-note">Points compare source relevance within this request. They are not a probability of suitability or a verified experience measure.</p>'
                + comparison_table(rows, scoring),
            )
        )
    return "".join(pieces)


def questions_html(state: dict, candidate_id: str | None = None) -> str:
    questions = [
        q
        for q in state.get("questions", [])
        if not candidate_id or q.get("candidate_id") == candidate_id
    ]
    if not questions:
        return empty(
            "Prepare a focused interview",
            "Complete detailed screening, then choose a candidate and prepare questions. Each question includes its purpose, source evidence, and what to listen for.",
        )
    pieces = [
        f'<div class="view-intro"><h2>Interview guide</h2><p>{len(questions)} question{"s" if len(questions) != 1 else ""} based on the current requirements and candidate evidence.</p></div>'
    ]
    for n, q in enumerate(questions, 1):
        pieces += [
            f'<article class="question-card"><div class="eyebrow">QUESTION {n:02} · {esc(q.get("candidate_id", "Candidate"))}</div><h3>{esc(q.get("question", q.get("text", "")))}</h3>'
        ]
        for key, label in [
            ("purpose", "Why ask this"),
            ("reason", "Reason"),
            ("follow_up", "Follow-up"),
            ("strong_answer", "What to listen for"),
            ("what_to_listen_for", "Assessment notes"),
        ]:
            if q.get(key):
                pieces.append(f"<h4>{label}</h4><p>{esc(q[key])}</p>")
        if isinstance(q.get("evidence"), list):
            pieces.append(
                "<details><summary>Source evidence behind this question</summary>"
                + "".join(source_html(e) for e in q["evidence"] if isinstance(e, dict))
                + "</details>"
            )
        pieces.append("</article>")
    return "".join(pieces)


def changes_html(state: dict) -> str:
    changes = state.get("ranking_changes", [])
    if not changes:
        return '<p class="muted">Changes in rank will appear after you refine the requirements.</p>'
    rows = []
    for c in changes:
        old, new = c.get("previous_rank"), c.get("new_rank")
        movement = (
            f"Entered at #{new}"
            if old is None
            else f"Left the shortlist (was #{old})"
            if new is None
            else f"#{old} → #{new}"
        )
        delta = f" · {c['score_delta']:+.2f} points" if c.get("score_delta") is not None else ""
        detail = ", ".join(
            f"{COMPONENTS.get(k, k)} {v:+.2f}" for k, v in c.get("component_deltas", {}).items()
        )
        rows.append(
            f"<li><strong>{esc(c['name'])}</strong><span>{esc(movement + delta)}</span>"
            + (f"<small>{esc(detail)}</small>" if detail else "")
            + "</li>"
        )
    return (
        '<p class="muted">The full corpus was searched using the revised criteria. Points reflect the weights active in each revision.</p><ul class="changes">'
        + "".join(rows)
        + "</ul>"
    )


def result_view(state: dict) -> str:
    if state.get("error") or state.get("clarification"):
        return "Candidates"
    return {
        "compare": "Compare",
        "explain": "Compare",
        "questions": "Interview",
        "deep_screen": "Evidence",
        "finalize": "Recommendations",
        "approve": "Report",
        "file_tool": "Activity",
    }.get(state.get("plan", {}).get("action"), "Candidates")


def conversation_reply(state: dict, corpus_size: int) -> str:
    """Lead with the completed action, then the result and a useful next step."""
    if state.get("error"):
        return f"I couldn’t complete that request: {state['error']}\n\nThe previous results are still available."
    if state.get("clarification"):
        return state["clarification"]
    if state.get("tool_result") is not None:
        outcome = state["tool_result"]
        if not outcome.get("ok"):
            return "The file operation failed: " + str(outcome.get("error", "Unknown error"))
        name = state.get("tool_request", {}).get("name", "file operation")
        result = outcome.get("result", {})
        if name == "read_file" and isinstance(result, dict):
            content = str(result.get("content", ""))
            return (
                f"I read the document ({len(content):,} characters). You can now ask me to use its job requirements.\n\n"
                + content[:350]
                + ("…" if len(content) > 350 else "")
            )
        return f"The {name.replace('_', ' ')} operation is complete. Open **Activity** to inspect its result."
    action = state.get("plan", {}).get("action", "help")
    matches = state.get("shortlist", [])
    if action in {"search", "refine"}:
        if not matches:
            return "No retrieval shortlist is available. Inspect the search status before changing your requirements."
        tops = "\n".join(f"- **{m['name']}** — {m['score']:.1f}/100 relevance" for m in matches[:3])
        movement = (
            " The full corpus was searched again with the revised criteria; see Ranking changes for movement."
            if action == "refine"
            else ""
        )
        return (
            f"I searched **{corpus_size} profiles** and selected **{len(matches)} provisional candidates** by relevance.{movement}\n\n"
            + tops
            + "\n\nThe evidence view shows relevant passages and the requirements still to verify. **No candidate qualification or experience calculation has been performed yet.** Run **detailed screening** for that assessment. Other profiles remain available as retrieval alternatives."
        )
    if action == "deep_screen":
        reviews = state.get("detailed_reviews", [])
        pending = sum(r.get("stage") == "pending" for r in reviews)
        counts = Counter(
            c["status"] for r in reviews if r["stage"] == "deep" for c in r["criteria"]
        )
        return (
            f"I completed detailed screening for **{len(reviews) - pending} candidate{'s' if len(reviews) - pending != 1 else ''}**. "
            + (
                f"**{pending} review{'s remain' if pending != 1 else ' remains'} pending; retry detailed screening to finish.** "
                if pending
                else ""
            )
            + f"Across their requirements, **{counts['supported']} findings are supported**, **{counts['uncertain']} need clarification**, and **{counts['not_demonstrated']} are not demonstrated** in the resumes.\n\n"
            + "Open **Evidence** for the requirement findings, employment history, projects, and reported results. The list now follows the reviewed evidence. **Compare** shows the current top three side by side.\n\n"
            + "Continue to **Recommendations** for decisions and the reasons to prioritize one candidate over another."
        )
    if action == "finalize":
        counts = Counter(m.get("recommendation") for m in matches)
        memo = state.get("decision_memo", {})
        return (
            f"Recommendations are ready: **{counts['hire']} advance**, **{counts['hold']} hold**, and **{counts['no_hire']} no hire**.\n\n"
            + memo.get("summary", {}).get("text", "").replace("no_hire", "no hire")
            + "\n\nOpen **Recommendations** for the comparative memo, justified priority groups, and what would change each decision. The memo was checked against completed detailed findings and your criteria. These are screening recommendations for your review."
        )
    if action in {"compare", "explain"}:
        rows = comparison_rows(state)
        if not rows:
            return "The ranking changes are recorded under **Candidates → Ranking changes**."
        text = (
            f"I compared **{len(rows)} provisional candidates** by retrieval relevance. Their qualifications and experience remain unverified until detailed screening."
            if state.get("round", "initial") == "initial"
            else f"I compared **{len(rows)} candidates** against the same requirements. The **Compare** view shows their experience, strengths, gaps, and review status side by side."
        )
        if (
            state.get("round", "initial") == "initial"
            and len(rows) >= 2
            and all("score" in r for r in rows[:2])
        ):
            leader, other = sorted(rows[:2], key=lambda r: -r["score"])
            gap = leader["score"] - other["score"]
            text += f"\n\n{leader['name']} leads {other['name']} by **{gap:.2f} points**."
            if bool(leader.get("eligible")) != bool(other.get("eligible")):
                text += " The shortlist places supported evidence ahead of unresolved evidence before comparing relevance points."
            deltas = [
                (k, v - other.get("components", {}).get(k, 0))
                for k, v in leader.get("components", {}).items()
            ]
            changed = sorted(
                ((k, v) for k, v in deltas if abs(v) >= 0.005), key=lambda pair: -abs(pair[1])
            )
            if changed:
                text += (
                    " The largest difference is "
                    + COMPONENTS.get(changed[0][0], changed[0][0]).lower()
                    + f" ({changed[0][1]:+.2f} points)."
                )
        return text
    if action == "questions":
        qs = state.get("questions", [])
        ids = list(dict.fromkeys(q.get("candidate_id", "") for q in qs))
        return f"I prepared **{len(qs)} interview question{'s' if len(qs) != 1 else ''}** for {', '.join(ids)}.\n\nOpen **Interview** for the questions, why each matters, suggested follow-ups, and the source evidence."
    if action == "approve" or state.get("status") == "complete":
        return "Your review is marked complete and saved locally. The **Report** view contains the final document and download links."
    return "I can help you find candidates, refine requirements, compare matches, and prepare interviews. Start with a role and any skills or experience you need.\n\nFor example: “Find software developers with React and at least three years of experience.”"


def status_html(message: str, *, busy: bool = False, error: bool = False) -> str:
    tone = "error" if error else "busy" if busy else "ready"
    return f'<div class="request-status {tone}" role="{"alert" if error else "status"}" aria-live="polite" aria-atomic="true"><span class="status-icon" aria-hidden="true">{"!" if error else ""}</span><span>{esc(message)}</span></div>'


def activity_html(state: dict, *, completed: list[str] | None = None, busy: bool = False) -> str:
    workflow = "Contextual screening with detailed review of the shortlist"
    events = state.get("node_events", [])
    if completed is None:
        start = next(
            (i for i in range(len(events) - 1, -1, -1) if events[i].get("node") == "parse_jd"),
            len(events),
        )
        recent = events[start:]
        completed = [e["node"] for e in recent]
    else:
        recent = []
    steps = "".join(
        f'<li><span class="step-check" aria-hidden="true">✓</span><div><strong>{esc(NODES.get(node, node))}</strong><small>Completed</small></div></li>'
        for node in completed
        if node != "human_feedback"
    )
    if busy:
        steps += '<li><span class="running-dot" aria-hidden="true"></span><div><strong>Working on the next step</strong><small>Updates appear when a graph step completes.</small></div></li>'
    elif completed and state.get("status") != "complete":
        steps += '<li><span class="step-pause" aria-hidden="true">Ⅱ</span><div><strong>Waiting for your feedback</strong><small>Compare, refine, continue screening, or complete the review.</small></div></li>'
    trace = (
        '<div class="activity-flow"><h3>Latest request</h3><ol>' + steps + "</ol></div>"
        if steps
        else empty(
            "Your activity will appear here",
            "Send a request to see the steps the agent actually completes.",
        )
    )
    tools = []
    for event in reversed(state.get("tool_events", [])[-20:]):
        label = event.get("tool", "").replace("_", " ").capitalize()
        summary = event.get("result_summary", {})
        result = ", ".join(
            f"{k.replace('_', ' ')}: {v}"
            for k, v in summary.items()
            if not isinstance(v, (dict, list))
        )
        tone = "positive" if event.get("status") == "success" else "negative"
        elapsed = event.get("elapsed_ms", 0) / 1000
        tools.append(
            f'<div class="tool-event"><div><strong>{esc(label)}</strong>{badge(event.get("status", "unknown").capitalize(), tone)}</div><p>{esc(event.get("error") or result or "Result saved in the review record.")}</p><small>{elapsed:.2f} seconds</small></div>'
        )
    return (
        '<div class="view-intro"><h2>How this result was produced</h2><p>Actual workflow steps and tool outcomes from this review, with source evidence behind each decision.</p>'
        + f"<p><strong>Workflow:</strong> {esc(workflow)}</p>"
        + "</div>"
        + trace
        + "<h3>Recent tool activity</h3>"
        + ("".join(tools) or '<p class="muted">No tool has completed yet.</p>')
    )


def quote_markdown(ev: dict) -> str:
    quote = "\n".join("> " + line for line in ev.get("quote", "").splitlines())
    return (
        quote
        + f"\n\nSource: `{ev.get('source_path', '')}` · characters {ev.get('start', 0)}–{ev.get('end', 0)}.\n"
    )


def report_markdown(state: dict) -> str:
    if not state.get("shortlist"):
        return state.get("screening_report") or "Run a search to create a screening report."
    req = state.get("requirements") or {}
    stage = {
        "initial": "Initial shortlist",
        "deep": "Detailed screening",
        "final": "Recommendations",
    }.get(state.get("round"), "Screening")
    lines = [
        "# Candidate screening report",
        f"{stage} · requirements revision {state.get('requirements_version', 0)}",
        "## Requirements",
    ]
    if req.get("role"):
        lines.append("- Role: " + ROLE_LABELS.get(req["role"], req["role"]))
    lines += [
        "- Mandatory skills: "
        + ("; ".join(" or ".join(g) for g in req.get("must_have", [])) or "None specified"),
        f"- Minimum total employment: {req.get('minimum_years', 0):g} years",
        "- Preferred skills: " + (", ".join(req.get("nice_to_have", [])) or "None specified"),
    ]
    lines += [
        f"- Minimum {skill} experience: {years:g} years"
        for skill, years in req.get("skill_years", {}).items()
    ]
    if req.get("evidence_standards"):
        lines += [
            "## Shared evidence standards",
            "These standards were set from this request before assessing candidates and apply throughout this revision.",
        ]
        for standard in req["evidence_standards"]:
            lines += [
                "**" + standard["capability"] + "**",
                standard["sufficient_evidence"],
                "Equivalent evidence: " + standard["equivalence_boundary"],
                "Uncertainty: " + standard["uncertainty_boundary"],
            ]
    result = state.get("search_result", {})
    lines += [
        "## Screening outcome",
        f"Searched {result.get('corpus_size', 'the available')} profiles and selected {len(state['shortlist'])} for detailed review. Other profiles remain unassessed; retrieval does not reject them.",
        "Scores describe retrieval relevance, not qualification or hiring probability. Initial results are unverified. After detailed screening, mandatory evidence coverage determines order before relevance.",
    ]
    if state.get("round") == "final":
        lines += [
            "## Screening decisions",
            "These decisions use the completed evidence review. Detailed findings follow below.",
        ]
        table = ["| Candidate | Decision | Basis |", "|---|---|---|"]
        for m in state.get("shortlist", []):
            table.append(
                "| "
                + " | ".join(
                    str(v).replace("|", "\\|").replace("\n", " ")
                    for v in [
                        m["name"],
                        m.get("recommendation", "pending").replace("_", " "),
                        m.get("recommendation_reason", ""),
                    ]
                )
                + " |"
            )
        lines.append("\n".join(table))
        for m in state.get("shortlist", []):
            lines.append(
                "Next actions for "
                + m["name"]
                + ": "
                + "; ".join(m.get("recommendation_actions", []))
            )
        lines.append("## Detailed evidence")
    for n, m in enumerate(state["shortlist"], 1):
        lines += [
            f"## {n}. {m['name']}",
            f"Initial retrieval points: **{m['score']:.2f}/100** · {match_status(m, state)[0]}",
        ]
        if m.get("recommendation_reason"):
            lines.append(m["recommendation_reason"])
        review = next(
            (
                r
                for r in state.get("detailed_reviews", [])
                if r["candidate_id"] == m["candidate_id"]
            ),
            None,
        )
        if review:
            lines.append(detailed_review_markdown(review))
        else:
            lines += [
                "### Retrieval leads · unverified",
                "No qualification or duration assessment has been performed.",
            ]
            for lead in m.get("retrieval_leads", []):
                lines.append("**" + lead["criterion"] + " · unverified**")
                lines.extend(quote_markdown(ev) for ev in lead["evidence"])
        if m.get("improvement_suggestions"):
            lines += [
                "### Evidence to strengthen",
                "\n".join("- " + x for x in m["improvement_suggestions"]),
            ]
    memo = state.get("decision_memo")
    if memo:
        lines += ["## Comparative decision memo", memo["summary"]["text"]]
        names = {m["candidate_id"]: m["name"] for m in state["shortlist"]}
        for n, tier in enumerate(memo["advance_tiers"], 1):
            lines.append(
                f"Priority {n}: "
                + ", ".join(names[cid] for cid in tier)
                + (" (no justified preference within this group)." if len(tier) > 1 else ".")
            )
        for pair in memo["comparisons"]:
            lines += [
                "**" + " / ".join(names[cid] for cid in pair["candidate_ids"]) + "**",
                pair["conclusion"]["text"],
                "Reviewed fact references: " + ", ".join(pair["conclusion"]["fact_ids"]),
            ]
    if state.get("questions"):
        lines.append("## Interview guide")
        for q in state["questions"]:
            lines += [
                f"### {q['candidate_id']}",
                q.get("question", ""),
                "**Purpose:** " + q.get("purpose", ""),
            ]
            for key in ("follow_up", "strong_answer"):
                if q.get(key):
                    lines.append("**" + key.replace("_", " ").capitalize() + ":** " + str(q[key]))
            lines += [quote_markdown(e) for e in q.get("evidence", []) if isinstance(e, dict)]
    lines += [
        "## Review status",
        "Review complete." if state.get("status") == "complete" else "Waiting for human review.",
        "Screening recommendations are based on the supplied documents. The reviewer verifies the evidence and makes the employment decision.",
    ]
    return "\n\n".join(lines)
