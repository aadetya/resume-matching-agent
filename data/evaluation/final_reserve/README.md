# Source reserve for the semantic audit

I prepared 50 source profiles and reviewed each complete source against one of three requests before running the matching engine. The requests cover backend Python and SQL work, browser interfaces built with React and TypeScript, and Python and SQL data engineering. Each request also asks for three years of total employment when dates are available. All 50 profiles contain dated employment, giving 200 criterion judgments.

The profiles come from the synthetic People dataset. They contain no generated additions to the selected source facts. Names are replaced with `Candidate R001` through `Candidate R050`; biographies, demographic fields and administrative source IDs are omitted from the resume text. Original source IDs remain in the manifest and sanitized source records for provenance.

## Selection and review

The three [requests](queries.json) and [selection protocol](selection_protocol.json) were written before selection. Membership uses employment-role strata and a deterministic source-ID ordering. Skills, computed experience, retrieval results and model judgments did not choose the profiles.

The [selection manifest](selection_manifest.json) records 1,485 exclusion-artifact hashes containing 332 unique source IDs. Private artifact paths are hashed and the reviewer is identified by annotation role in the publication copy; [publication provenance](publication.json) links that copy to the original unchanged freeze. The selected profiles have no source-ID overlap with those artifacts, including the active 200 profiles, the earlier 50-profile holdout and the local architecture-study snapshots. This establishes separation from the recorded local sources. It does not establish that a provider profile was absent from model training or every other historical copy.

I read all 50 canonical sources before supplying statuses, reasons and exact quotations. Employment totals use the dated entries identified during that review, followed by a separate check of the calendar-month arithmetic. Ended ranges include both named months. `Present` stops at the 2026-09-01 snapshot and excludes the incomplete September month. Overlapping employment counts once; total employment does not establish time using a particular skill.

These are single-agent, source-based judgments. They have no independent human review. Of the 200 criteria, 119 have one allowed status and 81 retain multiple allowed statuses because the source or documentary-insufficiency boundary is ambiguous. Agreement with those sets needs to be reported separately from agreement on unambiguous labels.

## Files and reproduction

| File | Purpose |
| --- | --- |
| [judgments.json](judgments.json) | Runner-compatible cases, complete canonical text and sealed criterion judgments |
| [freeze.json](freeze.json) | Source, request and label hashes recorded before model evaluation |
| [mapping_provenance.json](mapping_provenance.json) | Identity mapping to the runner schema, input hashes and source-ID join |
| [corpus_manifest.json](corpus_manifest.json) | Candidate IDs, original source IDs, file formats and document hashes |
| [review_packet.jsonl](review_packet.jsonl) | Sanitized source packet used for review |
| [source_profiles.jsonl](source_profiles.jsonl) | Sanitized source records with administrative provenance |
| [review_protocol.json](review_protocol.json) | Annotation rules, snapshot convention and scope |
| [visual_review.json](visual_review.json) | Extraction, geometry and visual checks for the rendered documents |

The runner can use the published judgment file directly:

```bash
uv run --env-file .env python scripts/run_contextual_evaluation.py \
  --fixture data/evaluation/final_reserve/judgments.json \
  --run-id final-reserve-v1 \
  --output reports/semantic_audit/final-reserve.json
```

This command makes connected model calls. The fixture preparation and review made none. The runner receives the canonical TXT content stored in each case; that text matches `text/Rnnn.txt` byte for byte. The original source ID is resolved through the administrative manifest and is absent from the model-facing resume text.

The 50 rendered documents comprise 25 PDFs, 15 DOCX files and 10 TXT files. All 43 pages from the PDF and DOCX exports were visually checked. Every source line survived extraction from all 50 exports after whitespace normalization; the document hashes and page bounds also passed their checks. Page furniture and extraction whitespace mean the binary exports are not expected to extract into byte-identical TXT.

## What this reserve measures

Each profile has one assigned request. The reserve can measure criterion status, source attribution, duration bounds and missed employment associations for those 50 pairs. It cannot produce complete-corpus recall, nDCG or candidate-ranking quality from the unjudged pairs.

The requests were fixed before source selection and the labels were sealed before model evaluation. The profiles are still synthetic, role-stratified examples with author-reviewed labels. A result on this reserve does not establish human hiring validity or broad performance on real resumes. The earlier holdout remains a development dataset under its original rubric; its grades are not interchangeable with these criterion judgments.

The first evaluation informed subsequent development. This collection is now reused development material, not an untouched test set. Its one-request-per-profile judgments remain useful for source and criterion diagnostics; independently reviewed ranking labels would require additional annotation.
