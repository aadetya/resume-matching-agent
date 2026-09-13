# Tool Reference

## Tool boundary

The registry exposes the four filesystem tools from Milestone 1, RAG search from Milestone 2 and the three additional assignment tools. Pydantic validates every argument object and rejects extra fields.

I keep this boundary explicit so the planner can request an operation while extraction, source checks and path permissions remain application responsibilities.

## Available tools

| Tool | Arguments | Returned result |
| --- | --- | --- |
| `read_file` | `filepath: str` | Text, metadata, extraction segments and warnings. |
| `list_files` | `directory: str`, optional `extension: str` | Sorted direct-child file records. |
| `write_file` | `filepath: str`, `content: str` | Created/overwritten status and metadata. |
| `search_in_file` | `filepath: str`, `keyword: str` | Match counts, excerpts, offsets and source context. |
| `extract_requirements` | `jd: str` | Validated `Requirements` with shared evidence standards. |
| `rag_search` | `requirements: Requirements`, `top_k: int = 10` | Provisional `SearchResult` with unverified source leads. |
| `compare_candidates` | `candidate_ids: list[str]` | Current requirements, candidate rows and interpretation. |
| `generate_interview_questions` | `candidate_id: str` | Candidate-specific questions with purpose, follow-up and source evidence. |

The requested search size ranges from one through ten matches. Comparison requires two through ten distinct IDs from the current corpus. Comparison and interview generation need active requirements. Job text is limited to 100,000 characters; written report content to 2,000,000.

## Result envelope

```json
{
  "ok": true,
  "result": {},
  "event": {
    "event_id": "<ID>",
    "tool": "extract_requirements",
    "arguments": {"jd": "61 characters"},
    "status": "success",
    "elapsed_ms": 0.0,
    "result_summary": {"fields": ["title", "role", "must_have"]}
  }
}
```

The example is shortened. A failed call returns `{"ok": false, "error": "<message>", "event": {...}}`, with error status and elapsed time. Long job descriptions and written content are represented by their character counts in event arguments.

## Requirement and candidate schemas

`Requirements` contains:

```text
title              str
role               str | null
must_have          list[list[str]]
nice_to_have       list[str]
minimum_years      float, 0–60
skill_years        dict[str, float]
source_text        str
semantic_brief     str, at most 6,000 characters
unresolved         list[str]
capability_skills  list[str]
requirement_sources list[RequirementSource]
criterion_meanings list[CriterionMeaning]
evidence_standards list[EvidenceStandard]
standards_signature str
```

Groups use AND across lists and OR within a list. `role` is separate from skills. The model-facing `required_skills` and `alternative_skill_groups` are normalized into these public groups. Unfamiliar roles and skills remain valid text; aliases are normalization hints.

Each `EvidenceStandard` contains `criterion_id`, `capability`, `requested_depth`, `experience_basis`, `evidence_scope`, `sufficient_evidence`, `equivalence_boundary` and `uncertainty_boundary`. The compiler uses each criterion's bound conditions, examples and assumptions without candidate sources. Unchanged meanings and standards remain exact through refinements. Direct `extract_requirements` and `rag_search` calls use the same compilation contract. Tool events record its signature, cache status and model usage.

`SearchResult` contains:

```text
matches                list[Match]
corpus_size             int
retrieved_count         int
not_shortlisted_count   int
retrieval_audit         dict
timings                 dict[str, float]
backend                 str
```

The graph uses a `RetrievalBatch` for passage retrieval and all-corpus learned ranking. A direct `rag_search` call compiles requirements, retrieves passages, scores complete resumes and selects up to ten candidates. It does not assess qualifications. `retrieval_audit` records corpus coverage, the ranking input, raw context scores, per-criterion source leads, the pinned ranker revision, token counts, timings and reserve IDs. No paid pre-shortlist assessment pool or early qualification exclusion is used.

`Match` contains candidate identity, `score`, `eligible`, `screening_status`, score components, strengths, gaps, evidence, retrieval leads and detailed criterion assessments, and recommendation fields. Recommendation values are `pending`, `hire`, `hold` or `no_hire`. Fingerprints identify the requirements and engine behind the result.

A criterion assessment has `criterion`, `mandatory`, `status`, `reason` and `evidence`. Status is `supported`, `uncertain` or `not_demonstrated`. Temporal records and full audit responses are retained in the session's review records. Detailed records include the assessment `draft`, work observations and audited `final_assessment`, the frozen source inventory, endpoint and association checks, status disputes and agreed-support corrections. The contract is described in [contextual screening](contextual_screening.md).

Comparison returns `requirements`, `candidates` and `interpretation`. Candidate rows use the current evidence ledger; a candidate without a completed review remains unassessed. The comparison itself makes no candidate model call.

Each interview result contains `candidate_id`, `criterion_id`, `criterion`, `question`, `purpose`, `follow_up`, `strong_answer`, `evidence`, `model` and `provider_model`. The request supplies the current stable criterion IDs, frozen evidence standards, mandatory flags, alternatives, experience basis and thresholds. Code validates that these match the current requirements before generation. Total-employment questions cover employment across occupations; a deeper optional probe does not add an eligibility condition. The model selects allowed evidence IDs; code restores the original evidence records. Missing source evidence permits an explicit request for an example, without fabricating one.

`Evidence` has `candidate_id`, optional `source_id`, `source_path`, `start`, `end`, `quote`, `document_sha256` and `kind`. Offsets are half-open positions in extracted text: `text[start:end] == quote`. Source IDs are administrative provenance, not matching features.

## Filesystem schemas and examples

`read_file` returns `ok`, `content`, `metadata`, `segments` and `warnings`. Metadata includes name, path, extension, byte size, modification time, symlink status, hash and text counts, plus format-specific extraction details.

`list_files` returns a list with name, path, extension, byte size, modification time, symlink status and readability. Extension filtering is case-insensitive.

`write_file` returns `ok`, `status` and `metadata`. Status is `created` or `overwritten`; UTF-8 content is written through an atomic replacement.

`search_in_file` counts every match while retaining at most 100. Results include character offsets, line/context information and source location when available. Keywords are limited to 256 characters.

```text
/tool list_files {"directory":"data/resumes","extension":"pdf"}
/tool read_file {"filepath":"data/resumes/P001.pdf"}
/tool search_in_file {"filepath":"data/resumes/P001.pdf","keyword":"Python"}
/tool write_file {"filepath":"reports/generated/note.md","content":"Review note."}
```

A job description can enter through a document:

```text
Read data/job_descriptions/backend_platform.txt
Use the job description I just opened to create the shortlist.
```

The second request uses the latest successful read. A failed read clears that document context. Reads and listings resolve inside the project root; writes resolve inside `reports/generated/`. The path is resolved before checking the boundary, including symlinks.

Standalone file-tool errors use `{"ok": false, "error": {"code": "...", "message": "..."}}`. The agent registry normalizes them into its envelope. Index ingestion and conversational reads share the same extraction module.

## Model context and local state

Conversations require the connected OpenAI planner and contextual reviewer. File utilities and local ranking with supplied criteria can run without a key. The default is `gpt-5.6-luna`. `OPENAI_MODEL` overrides `MATCHING_MODEL`, which overrides the default. Configure `OPENAI_API_KEY`. See [installation](../README.md#installation) for the local `.env` file and launch command.

`--backend semantic` is the default local retrieval path. `dense` selects dense passage retrieval; the complete-resume neural ranker still selects candidates. `bm25` is an explicit lexical-only ranking diagnostic. These options do not replace the connected planner or detailed reviewer. The index builder also offers `--ner-backend rules` and `--lexical-only` for extraction/index diagnostics; these options do not add another conversational workflow.

The planner receives current criteria, ordered shortlist IDs and display names, the candidate directory and up to eight prior user requests limited to 2,000 characters each. The latest request is supplied separately. Earlier assistant reports are excluded. A latest successful document read can contribute up to 100,000 characters when the request refers to it.

Original upstream IDs stay outside automatic planner context. Text pasted by the user is sent as request content. Contextual review sends complete extracted source lines up to 50,000 characters per candidate, with recognized identities redacted. Free-text redaction can miss identifying material. Evidence-standard compilation receives pending criteria and their bound conditions, examples and assumptions without candidate sources. A direct structured request without bound meanings can supply its semantic brief; an unrelated rewritten brief is excluded when meanings are already bound. Review and audit reuse those standards. Source-inventory extraction and its separate audit receive complete source lines without job criteria. These two responses are cached across requests. Criterion assessment receives the frozen work records, and each duration criterion must address every employment entry. Original evidence dictionaries remain local.

Standalone filesystem calls do not require an API key:

```python
from fs_tools import read_file

result = read_file("data/resumes/P001.pdf")
if result["ok"]:
    print(result["content"])
else:
    print(result["error"]["code"], result["error"]["message"])
```

Read limits are 20 MiB per file and 500 PDF pages. DOCX packages are checked for valid structure, at most 5,000 entries, 50 MiB of expanded content and excessive compression. Encrypted, corrupt or image-only PDF inputs produce explicit errors. TXT extraction checks binary content, byte-order marks and decoding confidence. These utilities share the index's parser.

## Python session interface

```python
from pathlib import Path
from matching_agent import MatchingSession

session = MatchingSession(Path.cwd(), session_id="react-screen")
try:
    state = session.send("Find React candidates with 3+ years overall experience")
    print(state["report"])
    print(session.snapshot()["requirements"])
finally:
    session.close()
```

`send` returns JSON-compatible state; `stream` yields completed graph updates; `snapshot` reads saved state. `call_tool(name, arguments)` submits an explicit registry operation.

SQLite checkpoints use `artifacts/sessions-<engine hash prefix>.sqlite`. The model, corpus, configuration and relevant code contribute to the identity. The CLI's `--database` or Python's `db_path` selects an explicit store. An incompatible checkpoint cannot resume silently.

The Gradio Report view prepares Markdown and JSON downloads under `runtime/exports/<session_id>/`. Filenames include the requirement revision and content identity. JSON contains the complete graph state; Markdown contains the readable screening report.
