"""All-corpus local relevance scoring before bounded contextual review.

Relevance selects documents for review. Qualification is a separate evidence
decision. Missing dictionary matches never prove a negative qualification.
"""

from __future__ import annotations

import time

import numpy as np

from .contracts import Match, Requirements, RetrievalBatch, SearchResult
from .criteria import build_criteria
from .neural_ranking import (
    MAX_PAIR_TOKENS,
    RANKER_LOCK,
    RERANKER_MODEL,
    RERANKER_REVISION,
    score_complete_pairs,
)
from .policy import requirements_fingerprint

MAX_CRITERION_QUERY_CHARS = 1000


def validate_batch(index, batch, requirements, *, expanded=False):
    index._check_sources()
    if (
        batch.engine_fingerprint != index.engine_fingerprint
        or batch.requirements_fingerprint != requirements_fingerprint(requirements)
    ):
        raise ValueError("Retrieval requirements or engine changed; retrieve again.")
    if (
        set(batch.candidate_scores) != set(index.candidates)
        or len(batch.evidence_scores) != len(index.chunks)
        or not np.isfinite(list(batch.candidate_scores.values())).all()
        or not np.isfinite(batch.evidence_scores).all()
    ):
        raise ValueError("The retrieval batch is incomplete or contains invalid scores.")
    specs = build_criteria(requirements)
    expected = {spec["criterion_id"] for spec in specs}
    if len(expected) != len(specs):
        raise ValueError("Each retrieval criterion must have a distinct stable ID.")
    if (batch.criterion_scores or expanded) and set(batch.criterion_scores) != expected:
        raise ValueError("Requirement retrieval scores must include every expected criterion ID.")
    for scores in batch.criterion_scores.values():
        if set(scores) != set(index.candidates) or not np.isfinite(list(scores.values())).all():
            raise ValueError("Requirement retrieval scores are incomplete or invalid.")
        if not all(0 <= score <= 1 for score in scores.values()):
            raise ValueError("Requirement retrieval ranks must lie between zero and one.")
    if expanded:
        context = batch.retrieval_audit.get("context_scores")
        if not isinstance(context, dict):
            raise ValueError("Expanded retrieval must record its resume-context scores.")
        if context or index.vectors is not None:
            if (
                set(context) != set(index.candidates)
                or not np.isfinite(list(context.values())).all()
            ):
                raise ValueError("Resume-context scores are incomplete or invalid.")


def _criterion_query(spec):
    """Search one capability without importing other criteria into that query."""
    if spec["kind"] == "total_duration":
        query = "The candidate has a history of professional employment."
    elif spec.get("standard"):
        query = "The resume describes this capability: " + spec["standard"]["capability"]
    elif spec.get("meaning"):
        query = "Candidate experience: " + " ".join(
            c["quote"] for c in spec["meaning"]["conditions"]
        )
    elif spec["kind"] == "role":
        query = (
            "The candidate has experience working as a "
            + spec["alternatives"][0].replace("_", " ")
            + "."
        )
    else:
        query = "The candidate has experience using " + " or ".join(spec["alternatives"]) + "."
    if len(query) > MAX_CRITERION_QUERY_CHARS:
        raise ValueError(
            "The complete criterion exceeds the retrieval query budget; shorten its capability without dropping conditions."
        )
    return query


def hybrid_scores(index, batch):
    from .retrieval import tied_ranks

    dense = tied_ranks({cid: v[0] for cid, v in batch.candidate_scores.items()})
    lexical = tied_ranks({cid: v[1] for cid, v in batch.candidate_scores.items()})
    k = index.policy.rrf_k
    result = {}
    for cid, values in batch.candidate_scores.items():
        d = (k + 1) / (k + dense[cid])
        b = (k + 1) / (k + lexical[cid]) if values[1] > 0 else 0.0
        result[cid] = (
            b if index.backend == "bm25" else d if index.backend == "dense" else (d + b) / 2
        )
    return result


def _context_query(requirements):
    """Rank against current meaning, including preferences and evidence scope."""
    parts = []
    if requirements.semantic_brief.strip():
        parts.append("Current request: " + requirements.semantic_brief.strip())
    specs = sorted(
        build_criteria(requirements),
        key=lambda s: (not s["mandatory"], s["kind"], sorted(s["alternatives"])),
    )
    for spec in specs:
        description = spec.get("standard", {}).get("capability")
        if not description:
            description = (
                "total employment"
                if spec["kind"] == "total_duration"
                else " or ".join(sorted(a.replace("_", " ") for a in spec["alternatives"]))
            )
        item = ("Required: " if spec["mandatory"] else "Optional preference: ") + description
        if "minimum_years" in spec:
            item += f"; minimum {spec['minimum_years']:g} years"
        scope = spec.get("meaning", spec.get("standard", {})).get("evidence_scope")
        if scope:
            item += "; evidence scope: " + scope.replace("_", " ")
        elif spec["kind"] == "skill_duration":
            item += "; employment involving the stated skill"
        parts.append(item + ".")
    return "\n".join(parts)


def _context_pairs(index, query):
    """Every complete scoring source; display citations retain original offsets."""
    from .retrieval import redact_identity

    pairs, owners = [], []
    for cid, document in index.documents.items():
        source = redact_identity(document["text"], index.candidates[cid])
        if not source.strip():
            raise ValueError("A resume has no tokens available for context scoring.")
        pairs.append((query, source))
        owners.append(cid)
    return pairs, owners


def expand_retrieval(index, batch: RetrievalBatch, requirements: Requirements, *, progress=None):
    requirements = index._validate_requirements(requirements)
    validate_batch(index, batch, requirements)
    started = time.perf_counter()
    specs = build_criteria(requirements)
    queries = [_criterion_query(spec) for spec in specs]
    criterion_scores, leads = {}, {}
    for spec, query in zip(specs, queries, strict=True):
        result = index.retrieve(requirements, query_text=query)
        criterion_scores[spec["criterion_id"]] = hybrid_scores(index, result)
        leads[spec["criterion_id"]] = {
            cid: [
                index._evidence(cid, np.asarray(result.evidence_scores), limit=1)[0].model_dump(
                    mode="json"
                )
            ]
            for cid in index.candidates
        }
    context, token_counts = {}, {}
    neural_pairs, neural_seconds = 0, 0.0
    query = _context_query(requirements)
    if index.vectors is not None:
        from .retrieval import _reranker

        pairs, owners = _context_pairs(index, query)
        neural_pairs = len(pairs)
        neural_started = time.perf_counter()
        if progress:
            progress(
                {
                    "stage": "initial_screen",
                    "completed": 0,
                    "total": len(pairs),
                    "message": "Preparing the local ranking model and checking complete resume inputs…",
                }
            )
        # Cached model initialization and inference are serialized across sessions
        # to avoid duplicate GPU loads or concurrent batch memory spikes.
        with RANKER_LOCK:
            model = _reranker()
            values, lengths = score_complete_pairs(model, pairs, progress=progress)
        neural_seconds = time.perf_counter() - neural_started
        context = dict(zip(owners, values.tolist(), strict=True))
        token_counts = dict(zip(owners, lengths, strict=True))
    audit = {
        "strategy": "all_corpus_learned_relevance" if context else "lexical_diagnostic",
        "local_scored_candidate_count": len(index.candidates),
        "local_scored_chunk_count": len(index.chunks),
        "neural_pair_count": neural_pairs,
        "neural_reranking_seconds": round(neural_seconds, 4),
        "candidate_assessment_calls": 0,
        "qualification_assessed": False,
        "coverage_scope": "Every supplied resume was ranked for relevance. Qualification is unassessed until detailed screening.",
        "criterion_leads": leads,
        "query_count": 1 + len(specs) + bool(context),
        "criterion_ids": [spec["criterion_id"] for spec in specs],
        "criteria": [spec["criterion"] for spec in specs],
        "criterion_labels": {spec["criterion_id"]: spec["criterion"] for spec in specs},
        "criterion_query_character_limit": MAX_CRITERION_QUERY_CHARS,
        "criterion_queries": dict(zip((s["criterion_id"] for s in specs), queries, strict=True)),
        "context_query": query,
        "context_scores": context,
        "ranker_model": RERANKER_MODEL if context else None,
        "ranker_revision": RERANKER_REVISION if context else None,
        "ranker_device": str(model.device) if context else None,
        "context_token_counts": token_counts,
        "context_token_limit": MAX_PAIR_TOKENS,
        "context_representation": "Complete identity-masked resume text. No source truncation, highest-window aggregation or metadata qualification filter. Exact original passages supply citations separately.",
        "score_semantics": "Uncalibrated learned relevance points: 100 times sigmoid of the raw model logit. These are not qualification probabilities or percentages of requirements met. Raw logits determine ordering.",
        "duration_verification": "Date thresholds and overlap calculations are deferred to detailed screening.",
        "heuristic_exclusions": False,
        "scope": "Only supplied resumes; no external candidate facts or generated resume text.",
    }
    return batch.model_copy(
        update={
            "criterion_scores": criterion_scores,
            "retrieval_audit": audit,
            "timings": {
                **batch.timings,
                "expansion_seconds": round(time.perf_counter() - started, 4),
            },
        }
    )


def rank_for_review(index, batch, requirements, *, top_k=10):
    requirements = index._validate_requirements(requirements)
    validate_batch(index, batch, requirements, expanded=True)
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    started = time.perf_counter()
    audit = batch.retrieval_audit
    context = audit["context_scores"]
    # Dense and lexical passage scores supply source leads. They cannot override
    # the learned candidate ordering. Lexical-only mode is an explicit diagnostic.
    ranking_values = context or hybrid_scores(index, batch)
    specs = build_criteria(requirements)
    matches = []
    for cid, candidate in index.candidates.items():
        points = (
            100 / (1 + np.exp(-np.clip(context[cid], -700, 700)))
            if context
            else 100 * ranking_values[cid]
        )
        components = {"learned_relevance" if context else "retrieval": float(points)}
        match = Match(
            candidate_id=cid,
            source_id=candidate.source_id,
            name=candidate.name,
            eligible=False,
            screening_status="needs_review",
            score=float(sum(components.values())),
            components=components,
            assessments=[],
            strengths=[],
            gaps=[],
            evidence=index._evidence(cid, np.asarray(batch.evidence_scores)),
            requirements_fingerprint=requirements_fingerprint(requirements),
            engine_fingerprint=index.engine_fingerprint,
            improvement_suggestions=[],
            retrieval_leads=[
                {
                    "criterion_id": spec["criterion_id"],
                    "criterion": spec["criterion"],
                    "mandatory": spec["mandatory"],
                    "status": "unverified",
                    "evidence": audit.get("criterion_leads", {})
                    .get(spec["criterion_id"], {})
                    .get(cid, []),
                }
                for spec in sorted(specs, key=lambda row: row["criterion_id"])
            ],
        )
        matches.append(match)
    matches.sort(key=lambda m: (-ranking_values[m.candidate_id], m.candidate_id))
    selected = matches[:top_k]
    return SearchResult(
        matches=selected,
        not_shortlisted_count=len(index.candidates) - len(selected),
        corpus_size=len(index.candidates),
        retrieved_count=len(index.candidates),
        backend=index.backend,
        timings={**batch.timings, "ranking_seconds": round(time.perf_counter() - started, 4)},
        retrieval_audit={
            **audit,
            "strategy": audit.get("strategy", "hybrid"),
            "reserve_candidate_ids": [m.candidate_id for m in matches[top_k:]],
            "ordering": "Raw pretrained relevance logits over all complete resumes; candidate IDs break exact ties. Every profile remains eligible for retrieval; qualification is not assessed here."
            if context
            else "Explicit lexical-only diagnostic ordering.",
            "provisional": True,
            "provisional_count": sum(m.screening_status == "needs_review" for m in selected),
            "unreviewed_is_not_rejected": True,
        },
    )
