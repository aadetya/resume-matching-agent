# People Profile Corpus

## Dataset contents

The active index uses 200 selected profiles from the external synthetic People dataset. The earlier fifty-profile evaluation collection remains development material. An additional fifty-profile reserve has source-based criterion judgments sealed before its first model evaluation and subsequently reused in development. The original 100 generated resumes remain as regression fixtures.

I retained these collections separately so the application corpus, earlier evaluation labels, additional source reserve and controlled extraction cases keep their own identities.

| Collection | PDF | DOCX | TXT | Total |
| --- | ---: | ---: | ---: | ---: |
| Active People profiles | 100 | 60 | 40 | 200 |
| Earlier People development profiles | 25 | 15 | 10 | 50 |
| Additional People source reserve | 25 | 15 | 10 | 50 |
| Original generated fixtures | 50 | 30 | 20 | 100 |

The [source card](https://huggingface.co/datasets/akzaidan/People/blob/82242bb397d893284643c29f85d5dae082e02ad2/README.md) describes generated profiles and declares MIT licensing. These are external synthetic records, not verified applicants. Their PDF and DOCX layouts are rendered by this project.

## Selection and expansion

The first selection covered frontend, backend/full-stack, data, DevOps, machine learning and adjacent operations roles. A later hundred-profile expansion added software roles and skill combinations after the original sample lacked Python-and-React co-mentions.

Twenty expansion places targeted that co-mention; the other eighty covered occupational groups. Selection excluded existing source IDs and duplicate normalized content before adding profiles. The first hundred active documents and fifty evaluation documents retained their identities.

The sample is selected for assignment coverage and demonstration. Its distribution is not representative of applicants, and the expansion was informed by an observed coverage gap. It does not establish an unbiased improvement in retrieval accuracy.

## File organization

| Path | Contents |
| --- | --- |
| `data/resumes/` | 200 active documents. |
| `data/corpus_manifest.json` | Active membership, local/source IDs, document hashes and snapshot. |
| `data/holdout/resumes/` | Earlier 50-profile evaluation collection, reused during development. |
| `data/holdout/corpus_manifest.json` | Earlier evaluation identities and hashes. |
| `data/holdout/queries.json`, `judgments.json` | Earlier frozen queries and agent-reviewed labels. |
| `data/people/active.jsonl`, `holdout.jsonl` | Selected sanitized records and administrative provenance. |
| `data/people/provenance.json`, `expansion.json` | Selection procedures and source identity. |
| `data/people/source/` | Upstream card and retrieval metadata. |
| `data/regression/data/resumes/` | Original 100 generated documents. |
| `data/contextual_diagnostics.json` | Forty constructed contextual development cases. |
| `data/evaluation/semantic_audit_cases.json` | Sixteen paired semantic audit diagnostics. |
| `data/evaluation/final_reserve/` | Fifty additional sources, rendered files, canonical text, requests, labels and freeze records. |

Active IDs run from P001 to P200, earlier evaluation IDs from E001 to E050, and additional reserve IDs from R001 to R050. Rendered text uses labels such as `Candidate P001`. Original source IDs stay in manifests and local provenance, outside indexed text and automatic planner context because some IDs expose source cohort labels.

## Retained fields and dates

The qualification allowlist retains skills, employment titles and descriptions, education, projects and certifications. Biographies, demographic/status fields, career interests and the source's precomputed experience total are omitted. Administrative selection fields do not enter matching text.

The renderer preserves supplied qualification wording, normalizes whitespace and date presentation, and adds headings. It does not infer a technology stack from an occupation. Source descriptions can still contain inconsistent or incomplete information.

Employment screening uses the fixed `2026-09-01` snapshot. The source publisher does not document a collection reference date. Present therefore follows a project convention, not verified current employment. The contextual compiler's inclusive calendar-month rules are described in [screening notes](contextual_screening.md#dates-and-experience).

Reversed or incomplete education dates remain uncorrected in the source records. Education does not add employment tenure. Overlapping qualifying employment is merged by the runtime's duration calculation.

## Evaluation scope

Earlier agent reviewers supplied 300 candidate-query judgments across six requests for the E001–E050 collection. These labels informed development and were not independently verified by a human. Their earlier rubric uses different duration and mandatory-gate conventions, so its grades are not interchangeable with current typed criterion judgments.

The additional reserve fixes three requests before role-stratified selection and assigns one request to each of fifty profiles. Source-only review supplies two hundred criterion judgments with exact excerpts and employment bounds. The selected source IDs have zero overlap with the 332 unique IDs found in the 1,485 recorded exclusion-artifact hashes. This is separation from those named artifacts, not a claim about model training or every possible historical copy.

The [reserve notes](../data/evaluation/final_reserve/README.md) describe source selection, rendering, labels and the freeze. Eighty-one of its two hundred judgments permit multiple statuses for ambiguous evidence. This single-agent annotation is not human ground truth. Fifty assigned pairs support source-evidence evaluation; they do not establish complete-corpus ranking metrics. The first-use run subsequently informed an endpoint extraction change, so later runs on these profiles are reused-reserve validation.

The original generated corpus contains controlled facts and known edge cases. The forty-case contextual fixture and sixteen-case audit fixture are development material. Retained measurements and their limits are described in [evaluation notes](ranking_validation.md).

## Reproducibility

The retained source revision is `82242bb397d893284643c29f85d5dae082e02ad2`. Its Parquet file contains 178,078,918 bytes with SHA-256 `dd2d3b70642260b046f33034f3c19b608be5b72ad582005408d5256526490b42`. The full upstream file is outside the submission.

Recreate documents from the included sanitized records:

```bash
uv run python scripts/import_people.py --render-only
uv run python scripts/audit_people.py
```

To repeat selection, use a fresh staging copy and the pinned source:

```bash
uv run python scripts/import_people.py \
  --source /path/to/profiles.parquet --root /path/to/staging-copy

uv run python -m scripts.expand_people \
  --source /path/to/profiles.parquet --root /path/to/staging-copy
```

The fresh reserve is already rendered from its included sanitized records; its selection and review protocol are recorded separately and are not regenerated by the commands above.

The original selector refuses to replace an expanded corpus. The expansion appends the second hundred profiles to the first selection. Document metadata and archive timestamps are fixed, and manifests identify rendered hashes. Rebuild the index after recreating documents.

Recreate only the original synthetic fixtures with:

```bash
uv run python scripts/generate_dataset.py --root data/regression
```

Full source attribution appears in [data/ATTRIBUTION.md](../data/ATTRIBUTION.md).

The expansion manifest protects source documents, evaluation labels and corpus configuration. It records the original manifest hash and the count of excluded non-dataset artifacts. Historical application reports and their dependency lock are not required to reproduce or validate these sources.
