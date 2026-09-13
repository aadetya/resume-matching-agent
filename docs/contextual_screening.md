# Contextual Screening

## Project Objective

I use source-scoped work records to connect a candidate's responsibilities, technologies and dates across the document. A missed association or unsupported inference can affect both eligibility and duration, so the evidence and its remaining uncertainty stay visible.

The current implementation is [contextual.py](../src/screening_agent/contextual.py), with shared [evidence instructions](../src/screening_agent/contextual_prompts.py). It uses the connected model configured for the session, defaulting to `gpt-5.6-luna`. This architecture has no demonstrated state-of-the-art or broad accuracy advantage.

## Workflow

Initial screening uses local learned ranking across every complete profile and returns ten unverified leads. It does not invoke this candidate reviewer. The complete source assessment, work inventory and audit run during detailed screening. The [architecture notes](architecture.md) describe that distribution.

The detailed round applies one frozen evidence standard to each candidate. It assesses the requirements and collects useful work observations in one call, then audits both. Supported mandatory findings determine the reviewed order, followed by support proportion and retrieval relevance. The model does not generate the numeric relevance score. Recommendations compare the completed evidence in a separate memo; interview questions require their own request.

## Requirements and context

`semantic_brief` retains the complete current meaning beyond short labels. It distinguishes basic skill presence, applied depth, broad experience and explicitly continuous usage. Refinements preserve qualifiers that the user has not changed.

Each criterion retains exact user conditions and examples in `criterion_meanings`, linked to `requirement_sources`. Assumptions state interpretation choices. The compiler derives a standard from that meaning before seeing candidate text, recording capability, depth, experience basis, evidence scope and support boundaries. Unchanged meanings and standards are preserved exactly through refinements. The detailed assessment and audit receive the same stored standard. A model can still misclassify an example or interpret a condition incorrectly; provenance makes that decision inspectable.

A concrete implementation can support a broader capability without a literal keyword when the shared standard permits that interpretation. Adjacent technology alone does not prove the requested skill. Compiling a standard once prevents separate candidate calls from defining different written standards; it does not guarantee that the standard or each application is correct.

An explicit skills-list claim can support basic presence. It does not establish proficiency, personal implementation or years of use. A course can support requested coursework; employment requirements need their own work evidence.

Each assessment receives the complete extracted source as original non-empty lines, up to 50,000 characters. A larger source fails explicitly instead of producing an absence judgment from silently truncated text. Recognized names and contacts are redacted from model-facing lines. Other identifying material can remain in free text.

## Work and evidence relationships

A query-independent extraction inventories employment, projects, skills, education, training and other source context. Every original line must belong to at least one record. A separate source-only audit checks complete entries, date fidelity and grouping without judging eligibility. Both responses are cached for reuse across requests. The assessment receives these frozen records and cannot replace them.

Structured work records identify those source units. Each record contains a title and source line IDs. Employment records also contain their available interval dates; other dated annotations remain in cited text. Its required `end_kind` distinguishes an ended calendar date, ongoing work and an unknown endpoint. Calendar and ongoing endpoints retain their source token; null is reserved for unknown. Ongoing work maps to the snapshot without replacing its original token.

Every criterion records status, relationship, evidence depth, reason and source references. A duration criterion also records `unit_decisions`: include, exclude or uncertain for every employment unit. Each decision has its own source references and reason. Missing decisions fail validation; included units must have positive associations. An unrelated job therefore receives an explicit exclusion rather than disappearing from the inventory. `unit_ids` can cite supporting, contrary or contextual work. Those references grant no duration credit. Separate `duration_associations` contain only proposed positive links, with a work ID, requested experience basis, atomic assertion and supporting lines. Their `temporal_scope` distinguishes a whole role, separately dated usage, a quantity within a role and unknown timing. `usage_start_date`, `usage_end_date` and `usage_end_kind` belong to the usage association and do not replace the employment endpoints. The usage endpoint can be calendar, ongoing or unknown independently of its parent role.

A project's description and its following technology list can jointly establish resume-claimed application. A different project's stack cannot supply that connection. Attribution, negation, training and shorter usage claims remain part of the interpretation.

The model selects line IDs; code restores quotations, original character offsets and document hashes. Adjacent lines can be joined. Nonadjacent excerpts remain separate. Unknown or foreign line IDs, repeated criteria and invalid work references are rejected. Exact quotation establishes provenance, not semantic correctness.

## Dates and experience

| Evidence basis | Interpretation |
| --- | --- |
| Total employment | Merge accepted dated employment intervals. |
| Skill-related employment | Count work where the skill describes the role or a substantive ongoing responsibility. |
| Explicit usage | Require a stated usage period or a claim covering the whole role. |
| Shorter usage inside a longer role | Cap that role's contribution at the stated quantity. |
| Undated project | Retain application evidence without borrowing nearby employment dates. |
| Missing or ambiguous duration | Preserve uncertainty; missing information does not become zero. |

A dated skill-specific role can support related employment without a separate “three years” sentence. It does not verify uninterrupted use. A once-only task does not associate a skill with the whole role.

Only employment work records currently contribute to numeric duration. A separately dated usage interval must fit the source-supported employment bounds. The containment check respects day, month or year precision before month-granularity credit is calculated; unknown parent endpoints are not replaced with invented dates. Quantified partial usage requires its own source quantity. Unknown temporal scope grants no interval credit. Source dates must match the cited context either literally or through equivalent calendar normalization. The compiler accepts supported year, month and full-date forms, then calculates month-granularity coverage.

Explicit start and ended months are included. January 2020 through December 2023 therefore contributes forty-eight months. Present stops at the `2026-09-01` snapshot boundary. Year-only starts use the following January; year-only ends use the start of the stated year. These are conservative calendar conventions, not exact elapsed-day or full-time-equivalent estimates.

Numeric and supported number-word usage quantities are validated against source text and converted to whole months. Overlapping employment counts once. When the placement of quantified partial usage is unknown but its source limits are complete, the result retains minimum and maximum bounds. A positive work association with missing dates or year-only precision leaves overall coverage incomplete; accepted measurable intervals remain a lower bound and do not establish a global maximum.

Published duration prose is constructed from the calculated bounds, accepted association assertions, source dates and caps. It does not reuse a model's proposed aggregate explanation. Audit notes are labelled `Source assessment`, `Unresolved coverage` or `Rejected duration link`. The complete proposal remains in `pre_audit_assessment`.

## Detailed review graph

The outer graph invokes a compiled candidate-specific LangGraph:

| Node | Work |
| --- | --- |
| `extract_and_audit_work_history` | Extract and separately audit source work records within detailed screening. |
| `assess_requirements` | Assess every requirement and useful work context against the frozen records. |
| `audit_evidence` | Check the draft criteria, positive duration associations and observations against the complete source. |

`AssessmentResponse` contains criterion findings and up to four source-cited work observations. Observations must add useful context; a repeated skill list or an irrelevant date limitation is not sufficient. No questions or recommendations are generated. Both the assessment draft and audited result remain available. A support disagreement remains uncertain, while agreed support can retain a corrected explanation. The same model performs these separately prompted tasks and can share errors.

Audit requests contain typed target fields, work records, source-line IDs and redacted source text. Original evidence dictionaries, administrative IDs and source paths remain local. An invalid audit structure can receive one correction attempt containing the invalid response and validation error. Valid semantic judgments are to be preserved; the correction does not ask for a preferred candidate result.

## Final audit and recommendation

Non-duration `criterion_evidence_support` targets return a typed final status, relationship and evidence depth, with a reason and exact source lines. Their `entailed` field is null. The checker states its own judgment explicitly. If it differs from the draft support status, the published result is uncertain and retains both judgments with the frozen standard. Code never obtains a new status by reversing a Boolean rejection.

The audit uses the complete source and the draft factual explanation, allowing it to distinguish a false narrative from an agreed support decision. Its citations and explanation remain available beside the original draft. Agreed support permits a narrative or label correction; disagreement preserves uncertainty. Existing work IDs are attached only when their combined lines cover all audited lines. Original source provenance remains available when no complete work mapping exists.

Duration coverage returns a Boolean check and a null typed verdict. It identifies only positive association targets. Full work records remain context and do not all propose credit. Each positive association receives its own Boolean source-support check.

Separate endpoint checks verify that cited positive work records retain the available start date, endpoint kind and endpoint token. They judge extraction completeness rather than whether dates are numerically measurable. A genuinely unknown source date can pass that check; omitting a supplied ongoing marker cannot.

These duration checks receive source dates and usage claims, without computed totals or threshold verdicts. Structured minima are omitted from their criterion definitions, while the complete semantic brief remains available. Code alone applies date conversion, overlap, numeric caps and thresholds.

Incomplete coverage retains independently accepted associations as a lower bound. A sufficient lower bound from a complete source inventory can support a threshold. A complete upper bound below the threshold is not demonstrated regardless of experience basis. Incomplete coverage cannot establish a global maximum or a negative duration verdict; `maximum_months` becomes null. A positive association with unmeasurable dates remains unknown rather than becoming zero. With no accepted measurable link, duration remains unknown unless the complete source explicitly supports zero. The interface labels a known incomplete lower bound explicitly.

The source inventory and per-employment decisions remain fixed evidence inputs. An incomplete employment inventory cannot establish duration: its numeric value remains unknown. Non-employment annotation completeness is reported separately. With a complete inventory, uncertain decisions or incomplete date coverage preserve accepted lower bounds and leave the upper bound open. Source audit can remove unsupported associations or mark their decisions uncertain, but it cannot request a replacement duration proposal. The reviewer must inspect unresolved work rather than rely on another rewrite.

Up to four detailed observations receive separate claim-level checks in the same audit response. Unsupported observations are removed and retained with rejection reasons. A resume's reported outcome remains an unverified source claim after citation checking.

`audited_result_hash` binds the filtered assessment, and `audited_criteria_hash` binds its compiled findings. Source, requirements and engine identity are checked before recommendation. Recommendation compares the reviewed ledger, records decision contingencies and checks each memo claim against cited reviewed facts. It does not rerun the candidate assessment. Interview questions are generated only when requested.

## Caching and failures

Assessment caches are keyed by source packet, criteria, semantic brief, stage, model settings, relevant code and prompts. The neutral inventory and its audit exclude requirement state from their cache identity, so an unchanged source can be reused across different requests. These add two model calls for an uncached source, before the criterion assessment. Final-audit identity also includes the exact proposed assessment. Cached responses undergo integrity, schema and source validation before reuse.

A changed model or engine creates a different cache and session identity. A cached assessment does not remove the evidence audit; each cached audit is bound to its exact input and source identity.

Candidate-specific execution failures leave completed detailed results available alongside pending records. Detailed failures remain pending for review and cannot receive an advance recommendation. Failure records preserve available raw responses and completed-call usage. The entire source corpus is rechecked before committing the stage; source changes preserve the preceding review. Current bounds are forty criteria, sixty-four work records and fifty thousand source characters.

## Research basis

[DocRED](https://aclanthology.org/P19-1074/) and [DREEAM](https://aclanthology.org/2023.eacl-main.145/) motivate retaining evidence for relationships that cross sentence boundaries. This project uses structured model responses and local checks; it does not reproduce either trained extraction system.

[Recommend-Revise](https://aclanthology.org/2022.acl-long.432/) and [Revisiting DocRED](https://aclanthology.org/2022.emnlp-main.580/) document omitted relationships and incomplete annotation. They motivate checking missed support as well as proposed positives.

[Chain-of-Verification](https://aclanthology.org/2024.findings-acl.212/) motivates a separate verification task. [ALCE](https://aclanthology.org/2023.emnlp-main.398/) distinguishes citation support from citation relevance. These sources justify design choices, not claims that this implementation inherits their measured performance.

## Validation and limits

The [evaluation notes](ranking_validation.md) separate current workflow checks from historical semantic measurements. In the preceding reused-reserve architecture, strict agreement declined from 188/200 before audit to 185/200 after audit. Those results do not validate the current source-inventory and dispute contract, and they do not establish an aggregate semantic-accuracy benefit from another audit call.

Complete source delivery does not guarantee complete understanding. The ten-candidate shortlist can miss profiles, source audit can reproduce an error, and exact citations can accompany an incorrect inference. Human review is still needed for recommendations.
