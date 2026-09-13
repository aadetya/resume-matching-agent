# Dataset Attribution

## People source profiles

The active profiles, earlier evaluation collection and semantic-audit reserve are adapted from [akzaidan/People](https://huggingface.co/datasets/akzaidan/People). The publisher describes generated profiles and declares MIT licensing in the [retained dataset card](https://huggingface.co/datasets/akzaidan/People/blob/82242bb397d893284643c29f85d5dae082e02ad2/README.md). The release has no separate license file.

The unmodified card is retained in [people/source/README.md](people/source/README.md), with retrieval details in [source_metadata.json](people/source/source_metadata.json). The selected `profiles.parquet` contains 178,078,918 bytes and has SHA-256 `dd2d3b70642260b046f33034f3c19b608be5b72ad582005408d5256526490b42`.

I selected 200 active profiles in two batches, fifty earlier evaluation profiles and a separate fifty-profile semantic-audit reserve, retained allowlisted qualification fields, and rendered PDF, DOCX and TXT documents. Original source IDs remain in administrative records and manifests. The [import provenance](people/provenance.json), [expansion record](people/expansion.json) and [reserve selection manifest](evaluation/final_reserve/selection_manifest.json) document selection and excluded fields.

Document layouts, the sixteen paired audit diagnostics and agent-reviewed judgments are project additions. The included JSONL files contain selected sanitized records and administrative provenance; the full source file remains outside the submission.

## Earlier milestone vocabulary

The curated skill vocabulary in `config/skills.yaml` is adapted from the earlier resume RAG project, which used the [O*NET 31.0 Database](https://www.onetcenter.org/database.html). That project retained source tables and a download manifest.

The O*NET data is provided by the U.S. Department of Labor, Employment and Training Administration under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The [database license](https://www.onetcenter.org/license_db.html) describes attribution and adaptation requirements. O*NET® is a trademark of USDOL/ETA.

I adapted the vocabulary into skill identifiers, labels and aliases. USDOL/ETA has not approved, endorsed or tested these changes.

## Original fixtures

The earlier 100 generated resumes remain under `regression/data/`. They were constructed from five role families in `scripts/generate_dataset.py`. Names, employers, institutions, dates, descriptions and achievements are fictional; contact addresses use `example.test`.

The fixtures contain no copied O*NET tables or occupation/SOC metadata.
