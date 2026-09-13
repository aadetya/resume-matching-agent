# Semantic Audit Fixture

I wrote sixteen new source profiles and their expectations before inspecting any output from the audit implementation. The eight pairs use identical requirements within each pair. Existing diagnostic structure and definitions informed the format; prior development failures informed the topics.

The [fixture](semantic_audit_cases.json) contains twenty-four criterion expectations and nine required positive duration associations. Its original SHA256 is `7f17670afa01749e38cbfef24531ee4865c4f246f4e841fe4d88f513ff304784`. [Annotation provenance](../../reports/manifest.json) records the current file hash and unchanged judgment content. These are targeted, author-reviewed synthetic examples. They are not independent human judgments or a representative hiring benchmark.

## Source judgments

| Pair | Shared request and expected distinction |
| --- | --- |
| SCA001–002 | Broad database implementation accepts both personally authored SQL and substantive ORM query/schema work. Handwritten statements are not required. |
| SCA003–004 | Explicit handwritten SQL authorship accepts the direct claim and rejects the source that expressly denies it. |
| SCA005–006 | Personal Python application is established by an owned project and its scoped stack. A skills list and course alone leave application unestablished. |
| SCA007–008 | Conflicting claims about the same Java contribution remain uncertain. Corroborating personal responsibility supports the other profile. |
| SCA009–010 | Both separate Python jobs must be recovered: twelve plus twenty-four months gives thirty-six months. Reordering and paraphrasing do not change the result. |
| SCA011–012 | The same thirty-six-month Python job survives different later context. Unrelated employment, study and undated projects add no employment credit. |
| SCA013–014 | Explicit Go usage is capped at six or eighteen months inside a sixty-month job. Only the eighteen-month claim meets twelve months. |
| SCA015–016 | An undated Python task establishes application with unknown usage duration. A dated two-month task establishes a duration below twelve months. |

SCA005 allows either `uncertain` or `not_demonstrated` because the source establishes familiarity without the requested application. SCA007 requires uncertainty because its attribution statements conflict. Neither case permits a supported application finding.

All quoted fragments occur in their source text. Requirements validate against the public contract, criterion names match the current criterion constructor, paired requirements are equal, and the nine association amounts pass separate calendar-month checks. These local checks validate the fixture's construction; they do not measure model performance.

## Evaluation boundary

The fixture keeps the existing status, relationship, duration and pair fields. `positive_associations` and `excluded_duration_source_fragments` add source facts for association checks. The evaluator checks required association completeness and excluded-context credit in addition to status and arithmetic.

Compare the assessment draft and audited result separately. Historical runner keys named `initial` and `deep` refer to these two candidate-review outputs; neither is the application’s provisional retrieval screen. Count corrected findings, newly introduced errors and unresolved findings without treating every changed result as a correction. Freeze the file before evaluation and retain disagreements.

## Earlier reserve labels

The earlier fifty-profile collection has 300 candidate-query judgments from two agent reviewers with adjudication. Its metadata states that reviewers did not inspect rankings before adjudication. There was no human annotator. The project later reused these labels during development.

That rubric excludes the end month from employment duration, converts unknown mandatory evidence into a failed documentary gate, and prohibits inferring SQL from PostgreSQL alone. The contextual workflow uses inclusive ended months and can assess a broader requested database capability. The older grade and gate labels therefore cannot directly score the new criterion and duration contracts.

Those labels remain useful under their frozen historical rubric. A current semantic evaluation needs separately recorded source judgments under the current request. Reannotation would still be agent review of development-exposed synthetic material. It would not create an untouched or human-validated reserve.
