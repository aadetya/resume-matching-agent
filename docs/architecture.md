# Agent Architecture

## Project Objective

I divide the work into three screening rounds. Initial screening selects source passages and a provisional shortlist. Detailed screening performs the candidate assessment. Recommendation compares the completed findings and explains the decision. There is no additional user-facing preparation round.

The [interactive diagram](architecture.html) and [SVG](architecture.svg) are generated from the compiled LangGraph. The browser and CLI both use `MatchingSession`; SQLite checkpoints preserve the conversation and human feedback loop.

## Workflow

| Round | Work | Output |
| --- | --- | --- |
| Initial screening | Interpret the request, freeze the evidence standards, retrieve and rank every profile locally. | Ten provisional candidates, relevant passage leads, unverified requirements and reserve IDs. |
| Detailed screening | Extract and audit work records for the shortlist; assess the complete source; calculate experience; audit the proposed findings. | First requirement judgments, work relationships, duration bounds, strengths, gaps and explicit uncertainty. |
| Recommendation | Verify the completed reviews, derive decision boundaries, compare the candidates and audit the memo. | Advance, hold or no-hire advice, justified priority groups, trade-offs and evidence that could change a decision. |

The planner selects a single evidence scope for each source-bound criterion. Internal depth and experience basis are derived from that choice, preventing contradictory combinations of redundant output fields. Invalid selectors and source quotations are returned with specific repair feedback; the one-correction budget and preservation of previous criteria remain in force.

Initial screening makes no candidate-specific OpenAI calls. Request interpretation and evidence-standard compilation still use the connected model. A pinned Qwen 4B reranker reads every complete, identity-masked resume against the current requirements. Its raw learned logits alone determine the ten-person pool. BGE and BM25 retrieve original source passages for inspection; their lexical and embedding scores cannot override candidate ordering. The current semantic brief, mandatory criteria, alternatives, preferences, thresholds and evidence scope reach the ranker. Historical titles and removed criteria do not. The [ranking validation](ranking_validation.md) records the model choice and its measured limits. Retrieval scores do not establish qualifications. The remaining 190 profiles are unassessed, not rejected.

Every fully formatted query/resume pair is checked against an 8,192-token limit before any inference begins. Oversized input stops the search rather than dropping source text. Inference batches also have a padded-token memory limit, and a process lock serializes access to the cached local model. The score shown to the user is `100 × sigmoid(raw relevance logit)`; it is uncalibrated and has no eligibility threshold. Sorting uses the raw logits so rounding or saturation cannot alter the selected set.

Detailed screening invokes a nested graph for each of at most ten candidates, with four workers. Its nodes are `extract_and_audit_work_history`, `assess_requirements` and `audit_evidence`. A fresh candidate normally requires four model calls: source extraction, inventory audit, assessment, and evidence audit. Work observations are produced with the assessment. Completed candidate counts are streamed to the interface.

Recommendation uses the reviewed fact ledger rather than reading the resumes again. Its two model steps draft and check a comparative memo. Code enforces complete candidate coverage, valid fact references, comparisons citing both candidates and priority groups containing exactly the advancing candidates. The semantic check rejects unsupported claims or new preferences. The memo is published only after those checks pass. Equal evidence can produce a tie; surplus years beyond a minimum do not become an unstated preference.

## State Design

| Field | Purpose |
| --- | --- |
| `messages`, `request`, `plan` | Conversation history and the selected action. |
| `requirements`, `requirements_version`, `requirements_history` | Current conditions, their source clauses and revisions. |
| `shortlist`, `search_result` | Candidate order, passage leads, relevance components and reserve IDs. |
| `detailed_reviews` | Source inventory, assessment draft, audited findings, disputes and execution receipts. |
| `decision_memo` | Cross-candidate comparisons, priority groups, cited reviewed facts and memo audit. |
| `comparison`, `questions`, `ranking_changes` | Requested follow-ups and changes in order. |
| `report`, `screening_report` | Current response and downloadable screening report. |
| `tool_events`, `node_events` | Executed steps, timing, model usage and transitions. |
| `round`, `deep_screen_version` | Completed stage and its requirement revision. |
| `engine_fingerprint`, `session_id` | Engine and checkpoint identities. |

A refinement searches the full corpus again and clears detailed findings, interview questions and recommendations tied to the old criteria. Unchanged criterion meanings and evidence standards are preserved. Comparing candidates reads available state and does not start assessment. Interview preparation requires an explicit request and a completed detailed review.

## Evidence and Failure Handling

The [contextual contract](contextual_screening.md) defines source grouping, semantic relationships and date calculations. Code checks source hashes and quotations, merges overlapping intervals, validates structures and preserves incomplete executions. A model interprets meaning; source references alone do not prove that interpretation is correct.

A failed candidate remains pending while other detailed reviews complete. Pending work cannot produce an advance decision. Corpus changes invalidate the transaction. An invalid recommendation memo leaves the completed detailed findings available. One bounded structural repair is allowed for malformed model output; there is no loop seeking a desired semantic label.

## Reproduction

The submission includes the source documents and index builder. Building the index extracts text, creates overlapping passages and computes local embeddings; it makes no OpenAI call and produces no candidate qualification assessment. The first model download needs internet. An evaluator can run initial screening with a fresh index and no candidate-assessment cache. The [evaluation notes](ranking_validation.md) record measured times and validation limits.

## Conditional routing

The normal search path is `START → parse_jd → extract_requirements → search_resumes → expand_search → rank_candidates → generate_report → human_feedback`. The feedback node calls LangGraph `interrupt`; the next message resumes through `Command` and returns to `parse_jd`.

The request router selects comparison, interview, file access, detailed screening, recommendation or completion. A recommendation request runs detailed screening first when the current requirement revision has no completed detailed round. After that prerequisite, `deep_screen` routes to `final_screen`. An explicit completion request runs `finish_review → END`. Clarifications and recoverable errors route to reporting while preserving the preceding committed search.

The three-node review subgraph is invoked from `deep_screen` for each shortlisted candidate. It is a compiled LangGraph called by the outer node, rather than an added edge between two separate state schemas. Local inventory caching is an implementation detail of detailed screening; it is not another user-facing round.

## Components and boundaries

| Component | Responsibility |
| --- | --- |
| `planner.py`, `criteria.py` | Interpret the request, preserve source clauses and compile shared standards. |
| `file_tools.py`, `corpus.py`, `chunking.py`, `metadata.py` | Extract files, validate membership, create overlapping passages and annotate entities. |
| `retrieval.py`, `candidate_retrieval.py`, `neural_ranking.py` | Retrieve passages and rank complete resumes locally. |
| `inventory.py`, `contextual.py`, `contextual_prompts.py` | Source inventory, criterion assessment, evidence audit and date compilation. |
| `review_integrity.py`, `policy.py`, `recommendation.py` | Verify reviewed evidence, derive decisions and audit the comparative memo. |
| `interview.py`, `agent_tools.py` | Explicit interview preparation and validated tool dispatch. |
| `app.py`, `ui_views.py`, `assets/` | Conversation, progress, candidate views and export presentation. |

Metadata skill aliases, role patterns and provisional date extraction remain parsing diagnostics. The connected ranker and reviewer do not use those fields as eligibility filters. The full source text reaches learned ranking and detailed assessment. Statistical NER improves entity annotation; it does not make the remaining metadata parser fully semantic.
