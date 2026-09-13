# Shortlist ranking validation

The initial screen scores every complete resume with `Qwen/Qwen3-Reranker-4B`, pinned to revision `22e683669bc0f0bd69640a1354a6d0aebcfeede5`. Its learned relevance score determines which ten candidates reach detailed screening. BGE and BM25 still retrieve source passages for inspection. Their scores do not determine the candidate order.

I made this change after finding that the preceding selector could put explicit denials ahead of documented work. The replacement passes the constructed meaning checks and improves the aggregate regression result. It still misses strong profiles, and one separate People query regresses. The evidence below supports a better development default with known limits; it does not establish maximum hiring accuracy or superiority over every architecture.

## Root cause

The preceding selector combined 75% dense/lexical retrieval rank with 25% MiniLM context rank. Topical similarity therefore dominated even when the cross-encoder preferred the better evidence. MiniLM itself also failed several simple meaning comparisons. Taking the strongest score from overlapping 512-token pairs could reward one locally promising passage without requiring the same score to account for contradictory context elsewhere.

The original source survived extraction in the failures examined. For the backend development query, `CAND_BE_005` described Python services and SQL work but ranked thirteenth, while `CAND_BE_018` explicitly denied those skills and ranked sixth. A DevOps denial fixture ranked third while a positive fixture ranked fourteenth. Adjusting those individual candidates would have concealed the failure in the scoring method.

The software tests checked workflow and data integrity, but the release criteria did not require the ranker to distinguish a denial, someone else's work, or study from requested professional experience. Passing those software tests was insufficient evidence of retrieval quality.

## Research and model choice

[NevIR](https://arxiv.org/abs/2305.07614) demonstrates that retrieval models can perform poorly when a small negation reverses relevance. This motivated paired semantic checks alongside whole-corpus retrieval metrics. The [Qwen3 embedding and reranking paper](https://arxiv.org/abs/2506.05176) describes pretrained rerankers at several sizes, trained across domains and languages. Those checkpoints provide learned query-document interactions without fitting feature weights to this small resume collection.

[ConFit](https://arxiv.org/abs/2401.16349) and [ConFit v2](https://arxiv.org/abs/2502.12361) study resume-job matching through contrastive learning, augmentation and hard-negative selection. They support evaluating difficult near-matches and the candidate pool separately from simple keyword overlap. This project does not reproduce their training or claim their published accuracy. The current labels are too small and too closely connected to development to justify a new hiring model trained on them.

The [selection protocol](../reports/ranking_validation/experiments/protocol.json) fixed 24 comparisons before the first model run. Twelve concern engineering and twelve concern other occupations. Cases cover denial, attribution, study, capability scope, implicit absence, unrelated negation, positive qualification containing a negative word, mandatory conjunctions, alternatives, allowed coursework, cross-sentence evidence and aspiration. Every case has now been inspected during model selection. All 24 are development diagnostics; the `transfer` field identifies occupational transfer, not an untouched test set.

| Pretrained model | Correct meaning comparisons | Decision |
|---|---:|---|
| MiniLM L6 v2, previous component | 9/24 | Insufficient semantic discrimination |
| Jina reranker v2 | 18/24 | Fails the development gate |
| Jina reranker v3.5 | 19/24 | Fails denial, attribution, scope and alternatives |
| Mixedbread rerank large v2 | 20/24 | Fails denial, coursework, scope and an alternative |
| Qwen3 reranker 0.6B | 22/24 | Still fails a professional-work denial and web/mobile distinction |
| Qwen3 reranker 4B | 24/24 | Retained for the application |

These are authored comparisons, with no independent human review. A model can pass them and still make errors in longer resumes. All model revisions and scores are preserved in the [evaluation records](../reports/manifest.json). Mixedbread's first run applied a sigmoid in half precision and created artificial ties. That invalid measurement is marked by the `saturated` filename; the table uses its corrected raw-score run.

## External semantic checks

The external tests use fixed random samples of 128 rows with seed `20260912`. NevIR uses revision `6263585072ce3b435ed09658613553fbf4e74184` of [orionweller/NevIR](https://huggingface.co/datasets/orionweller/NevIR). The exclusion test uses revision `81bced5dc6d9b8820cfdeaed588ddc7e9066e027` of [ExcluIR-conversational](https://huggingface.co/datasets/thijmennijdam/ExcluIR-conversational). Its conversation format is validated before decoding the query, two documents and preference label.

| Model component | NevIR paired correctness | ExcluIR preference correctness |
|---|---:|---:|
| MiniLM L6 v2 | 17/128 (13.3%) | 81/128 (63.3%) |
| Qwen3 0.6B | 40/128 (31.3%) | 98/128 (76.6%) |
| Qwen3 4B | 58/128 (45.3%) | 99/128 (77.3%) |

NevIR requires both queries in a row to prefer the correct document; its random baseline is 25%. ExcluIR evaluates one preference per row. These metrics cannot be compared directly. They measure model components with a generic search instruction, not the old weighted selector or the complete screening application. The samples are smaller than the full benchmarks, and prior exposure in checkpoint training is unknown. No full BEIR evaluation was run.

## Full-corpus shortlist quality

The regression collection contains 100 authored resumes and 15 queries. The separate People development collection contains 50 profiles and six queries, reviewed by two agents with adjudication and no human review. The original date convention in its frozen judgments is retained. The additional fifty-profile reserve has criterion judgments for one assigned request per profile. It lacks the complete candidate-query ranking grades required for these calculations.

A positive pair has grade 2 or 3. The maximum selectable count accounts for the ten slots per query and queries with fewer than ten positives. Highest-grade recall counts grade-3 pairs. Set gain sums `2**grade - 1` over the selected ten and divides by the best possible ten; ordering within the shortlist does not affect this metric.

| Collection and measure | Previous selector | Current application, frozen criteria |
|---|---:|---:|
| Regression: positive pairs selected | 131/150 available slots | 136/150 |
| Regression: highest-grade pairs selected | 60/75 | 69/75 |
| Regression: mean set gain | 0.864 | 0.931 |
| People: positive pairs selected | 23/25 available slots | 21/25 |
| People: mean set gain | 0.904 | 0.908 |
| People: mean highest-grade recall across applicable queries | 0.750 | 0.688 |

The corrected regression queries are identical across this before/after comparison. The [previous regression receipt](../reports/ranking_validation/experiments/corrected-baseline-frozen.json), [current regression receipt](../reports/ranking_validation/regression-frozen.json) and [current People receipt](../reports/ranking_validation/holdout-frozen.json) contain every selected ID and source citation. No explicit-denial fixture appears in the corresponding current regression shortlists.

The People decline occurs in the broad software-development query. Two selected profiles have dated developer titles and sufficient total employment but sparse descriptions of duties. The frozen rubric marks them as passing mandatory conditions while assigning grade 1 for weaker evidence. Their selection displaces two richer grade-3 profiles. This is a ranking-quality loss under that rubric, rather than proof that those two profiles are ineligible. The bare-criteria machine-learning queries also miss six highest-grade pairs across two requests.

## Query interpretation and input sensitivity

The exploratory ranker scored the original natural requests directly and selected 75/75 highest-grade regression pairs. The application's frozen-criteria path selected 69/75. Those are different model inputs. The application includes mandatory criteria, optional preferences, duration thresholds and evidence scope; connected conversations also include the interpreted current brief and compiled capabilities. The prototype's 75/75 is therefore not the application's accuracy.

The old evaluation called some queries paraphrases even though their text omitted preferences present in the supplied requirements. It also bypassed the interpreter entirely. The revised generator preserves every condition in each paraphrase, distinguishes total employment from skill-specific duration, and labels all inspected variants as development material. [The revision record](../data/regression/data/ground_truth/query_revision.json) preserves the original snapshot and hashes; no resume or original query history was silently overwritten.

Conversation mode now sends the actual query through `MatchingSession`, including the planner and shared-standard compiler. Expected requirements enter only the evaluator after the graph returns. Exact field comparisons deliberately withhold ranking metrics when interpretation differs. The separate [interpretation review](../reports/ranking_validation/interpretation-review.json) checks meaning without candidate IDs, scores or results. These are additional model judgments, with correlated errors possible; they are not independent human approval.

The interpreter uses one evidence-scope choice and derives its internal depth and duration basis from that choice. A single structural correction can report invalid selectors or source quotations. This constrains the representation without deciding which candidate should pass.

In the saved five-flow connected run, three requests passed exact field checks. A frontend paraphrase used `testing` for `automated testing`; the blinded model review accepted the meaning, while the exact-check ranking metric remained withheld. The backend request failed because its proposed citations joined non-contiguous words. The failure remains in the [conversation record](../reports/ranking_validation/regression-conversation.json).

Two fault-injected [repair checks](../reports/ranking_validation/quote-repair.json) completed a single correction after receiving precise quotation errors. A [fresh backend conversation](../reports/ranking_validation/backend-conversation-retest.json) then passed exact interpretation checks and selected all five highest-grade fixtures. These are separate observations; the full five-flow batch was not rerun after that feedback change. The machine-learning paraphrase still selected four of five highest-grade fixtures despite faithful interpretation.

## Runtime and stage boundaries

Initial screening reads all 200 complete resumes and returns ten provisional candidates. It makes no candidate-specific OpenAI assessment calls. Each requirement/resume pair is checked against an 8,192-token budget using the same formatter used for inference. A document that exceeds that budget stops the pass with an explicit error; it is not silently truncated or skipped. Batches have a separate padded-token limit, and a process lock prevents concurrent reviews from multiplying local inference memory use.

Raw logits determine ordering. The displayed 0–100 transformation is an uncalibrated relevance indicator, not a probability of qualification. Sorting never uses rounded display points. Names and contact details are masked before scoring, while displayed excerpts retain exact original source offsets. Progress updates report the number of complete resumes actually scored.

The measured 200-profile prototype pass took 109.487 seconds on an Apple M2 Max with 32 GB RAM. That number excludes request interpretation and model loading. The frozen 100-profile application benchmark had a median local duration of 26.304 seconds. Resume lengths and query lengths differ between those collections, so the times do not scale simply with candidate count. A separate 4-bit MLX trial took about 105 seconds on the 200-profile query and did not justify an additional runtime. A reduced candidate pool was also rejected because it could remove strong profiles before learned ranking.

The application selects CUDA, then Apple MPS, then CPU according to availability. Model weights require about 8 GB of download storage. CPU uses full precision and has not been timed on all 200 profiles; the initial protocol's proposed full CPU timing remains uncompleted. The two-minute estimate applies to the measured Mac configuration and is not a portable latency guarantee.

Detailed screening owns full-source qualification assessment, audited work records, criterion findings and local date arithmetic. Recommendations compare the completed reviews and validate their claims against those findings. Interview questions remain an explicit separate action. The [interactive diagram](architecture.html) is generated from the two compiled graphs and shows these responsibilities.

## Workflow verification

The [connected workflow record](../reports/workflow/verification.json) covers a search for backend development, Python, four years of overall employment, required SQL and preferred Docker. It records 200 locally scored profiles, ten detailed reviews, complete source coverage, zero pending reviews, a populated three-candidate comparison and five explicitly requested interview questions.

Detailed screening used 44 model calls with no candidate-cache hits. Recommendation required two calls, and all 24 memo claims passed its audit. Exact source checks verified 845 distinct quotation spans with no mismatch. These checks establish execution and provenance for this review; they do not measure independent hiring accuracy.

The recommendation schema restricts candidate, fact and audit-target IDs to the current ledger. This prevents a memo from citing nonexistent evidence. A separate semantic check tests whether each claim follows from the referenced facts. A valid ID alone does not establish support. Invalid memo output leaves the completed detailed findings available.

## Semantic assessment evidence

Earlier semantic runs used a preceding review contract. On the first fifty-profile reserve use, strict agreement was 166/200 before audit and 163/200 after audit. After development changes on those sources, the reused-reserve run reached 188/200 and 185/200 respectively. The [first-use result](../reports/semantic_evaluation/final-reserve-summary.json) and [reused-reserve result](../reports/semantic_evaluation/release-reserve-summary.json) retain per-criterion failures and their source/model identities.

These results show why adding an audit call does not automatically improve accuracy. They are historical measurements, not an accuracy evaluation of the current inventory, shared-standard and dispute contracts. The current implementation has source-integrity and workflow checks, but a new independently annotated semantic evaluation remains necessary. Both assessment and audit use the same model and can share interpretation errors.

## Testing and reproduction

```bash
uv run python scripts/benchmark_ranking_meaning.py
uv run python scripts/benchmark_selection.py --collection regression --mode frozen
uv run python scripts/benchmark_selection.py --collection holdout --mode frozen
uv run --env-file .env python scripts/benchmark_selection.py \
  --collection regression --mode conversation
uv run --env-file .env python scripts/review_interpretations.py \
  reports/ranking_validation/regression-conversation.json \
  --output reports/ranking_validation/interpretation-review.json
uv run --env-file .env python scripts/benchmark_stages.py --output reports/my-stage-run
uv run pytest -m "not live and not model"
uv run pytest -m model
```

The first three commands use local models. Conversation, interpretation review and stage benchmarks use the configured OpenAI key. A full regression conversation run contains fifteen requests; the retained focused run names the five requests it executed. Each benchmark records its source and label hashes and distinguishes frozen criteria from model interpretation.

The [report manifest](../reports/manifest.json) identifies retained measurement files and their original hashes. Reports retain scores, selected candidate IDs, source citations, interpretation failures and model identities. Repeated passage inventories and full conversation exports are omitted. Historical model comparisons remain measurements of their original versions; the current benchmark commands rerun the active implementation.

## Remaining limits

A pretrained relevance score cannot guarantee that every strongest candidate enters ten slots. The People regression and sensitivity to query representation remain open quality issues. The application makes no claim of state-of-the-art hiring accuracy.

The statistical NER model and overlapping chunks improve the extraction/retrieval foundation, but metadata still contains aliases and date-pattern diagnostics. Those fields do not determine connected eligibility. PDF/DOCX/TXT corruption, binary extraction, uncertain encodings, overlapping dates and unsupported claims remain covered by regression tests. Image-only input needs external OCR.

A stronger generalization claim needs previously unseen job/resume pairs, independently reviewed labels, fixed model inputs, repeated conversations and an acceptance criterion defined before testing. The existing fifty-profile source reserve lacks complete ranking judgments and has already informed development. It cannot serve as an untouched ranking benchmark without overstating the evidence.

## Current verification

The [local verification record](../reports/verification/checks.json) contains 454 passing offline tests and two passing local-model integration tests. Six provider tests were not run. The clean submission index rebuilt to 200 profiles and 612 passages; the People audit checked all 250 active/development documents with no failures. The locked dependency dry run, source/label preservation, static checks and no-key browser behavior passed.

These checks used the installed Python 3.12 environment. Dependency resolution was checked with a frozen dry run, rather than a new installation on another machine. The model tests preceded the final missing-key message fix; ranking and extraction were unchanged afterward. No OpenAI calls were made during this verification.
