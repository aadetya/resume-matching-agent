# Resume Data

The active corpus contains 200 selected profiles from the external synthetic People dataset. The earlier fifty-profile evaluation collection remains development material. I retained the original 100 generated resumes as regression fixtures and prepared a separate fifty-profile source reserve for the semantic audit.

## Dataset contents

The active collection has 100 PDF, 60 DOCX and 40 TXT files. The earlier People evaluation collection and source reserve each have 25 PDF, 15 DOCX and 10 TXT files. The [dataset notes](../docs/dataset.md) describe selection, source identity, dates and limits.

`resumes/` holds the active documents, with membership and hashes in `corpus_manifest.json`. `holdout/resumes/` holds the separate fifty profiles and its own manifest. Earlier queries and agent-reviewed labels remain with that development collection. [Annotation provenance](../reports/manifest.json) links original and published metadata hashes; source text and judgment content are unchanged.

`evaluation/final_reserve/` holds fifty additional sources and their frozen judgments. Each profile has one of three preregistered requests. The [reserve notes](evaluation/final_reserve/README.md) explain its two hundred source-based criterion judgments, provenance and limits. `evaluation/semantic_audit_cases.json` contains sixteen separate paired development diagnostics.

`people/active.jsonl` and `people/holdout.jsonl` retain sanitized source records and administrative provenance. Rendered documents use labels such as `Candidate P001`; original source IDs remain outside matching text. `people/provenance.json` and `people/expansion.json` record both selections.

The original fixtures remain under `regression/data/resumes/`. Their controlled facts and edge cases remain part of the test and evaluation collections.

## Reproduction

From the repository root:

```bash
uv run python scripts/import_people.py --render-only
uv run python scripts/audit_people.py
uv run python scripts/generate_dataset.py --root data/regression
```

The last command recreates the original generated fixtures under their separate root. Full source selection requires the pinned Parquet file and a staging copy; see [reproducibility](../docs/dataset.md#reproducibility).

## Attribution and scope

[Attribution](ATTRIBUTION.md) records People provenance and the vocabulary inherited from the earlier milestone. The fixed employment snapshot is `2026-09-01`.

All source collections are synthetic. Earlier evaluation labels were agent-reviewed and reused during development. The source reserve labels were sealed before its final evaluation, but remain single-agent source judgments with explicit ambiguity. None establishes performance on real hiring decisions.

The [architecture notes](../docs/architecture.md) describe the current review contract. Earlier evaluations retain their original source and engine identities; the reserved profiles have already informed development and are not an untouched benchmark.
