---
license: mit
language:
- en
size_categories:
- 100K<n<1M
task_categories:
- text-ranking
tags:
- synthetic
- jobs
- recruiting
- resumes
- learning-to-rank
pretty_name: Synthetic US Candidate Profiles
configs:
- config_name: default
  data_files: profiles.parquet
---

# Synthetic US Candidate Profiles

248,522 synthetic job-seeker profiles covering 803 occupations across the United States,
generated with DeepSeek V4 Flash. Built to train a job-ranking cross-encoder, where real
candidate data cannot be used.

**No real people.** Every profile is generated. No resumes were scraped, no user records
were used, and nothing here maps to a real individual. Names, emails, phone numbers and
URLs are excluded by construction and checked for on every row.

## Quick start

```python
from datasets import load_dataset
import json

ds = load_dataset("<your-org>/<your-dataset>", split="train")
doc = json.loads(ds[0]["profile_json"])

print(doc["user"]["career_interests"])   # ['Recommendation Systems Engineer', ...]
print(doc["user"]["skills"][:5])         # ['Learning-to-rank', 'Feature stores', ...]
print(doc["experience"][0]["title"])     # 'Senior Machine Learning Engineer'
```

## Schema

Seven flat columns for filtering without parsing JSON, plus the full record.

| column | type | description |
|---|---|---|
| `profile_id` | string | unique identifier; prefix marks the generation batch (see Provenance) |
| `source` | string | always `synthetic` |
| `cohort` | string | generation cohort (13 values, see below) |
| `job_family` | string | one of 14 families |
| `seniority` | string | `intern` … `executive` |
| `country` | string | always `United States` |
| `seed_role` | string | the occupation the profile was seeded from (803 values) |
| `profile_json` | string | complete profile as a JSON object |

`profile_json` is a single application-user document with exactly eight top-level keys:

| key | contents |
|---|---|
| `user` | age, ethnicity, legal_status, sponsorship_needed, city, state, country, desired_salary, experience_years, skills, languages, bio, career_interests |
| `work_location_preferences` | where they want to work — `remote` / `country` / `state` / `city` entries, containment-checked and capped at 20 |
| `experience` | company, title, start_date, end_date, description, city, state, country, is_internship |
| `education` | school, degree, field_of_study, start_date, end_date, is_current, gpa, description |
| `projects` | name, description, start_date, end_date |
| `certifications` | name, description, date |
| `awards`, `publications` | always empty — no source data |

Dates are `MM/YYYY` strings; a null `end_date` on an experience entry means the role is
current. `state` is a **two-letter code** (`"IL"`) and `country` is **ISO-2** (`"US"`).

Four fields are computed or sampled rather than generated, and should be read that way:

| field | how it is produced | why |
|---|---|---|
| `user.experience_years` | computed from experience date spans | LLMs are unreliable at date arithmetic, and this value feeds a scoring feature directly |
| `user.age` | derived from the earliest education/experience year | not generated, not real; a plausible value consistent with the work history |
| `user.ethnicity` | **sampled** from a census-like distribution, seeded by `profile_id` | present for coverage only — it is not inferred from the profile and carries no relationship to its content (see Limitations) |
| `user.legal_status`, `sponsorship_needed` | sampled from a fixed distribution | visa status is not reliably inferable from a résumé, and guessing it in a hiring context would be both inaccurate and wrong |

## Composition

**Cohorts** — profiles are deliberately not all well-formed. Real signups are often sparse,
mid-career-change, or hard to classify.

| cohort | n | | cohort | n |
|---|---|---|---|---|
| standard | 49,682 | | over_qualified | 14,945 |
| entry_level | 32,267 | | visa_constrained | 12,460 |
| career_changer | 24,785 | | return_to_work | 12,444 |
| sparse_profile | 22,402 | | bootcamp_self_taught | 7,511 |
| senior_specialist | 22,261 | | military_transition | 7,446 |
| student_intern | 19,955 | | gig_contract | 4,999 |
| manager_lead | 17,365 | | | |

**Seniority** — senior 29.7%, entry 21.2%, mid 20.4%, junior 8.3%, intern 4.9%,
manager 4.4%, staff 3.7%, principal 3.5%, director 2.6%, executive 1.4%.

**Job families** (14) — other 19.1%, operations 17.9%, software_engineering 15.7%,
healthcare 10.4%, research 6.3%, hardware_engineering 6.0%, finance 4.9%, education 3.9%,
design 3.7%, marketing 3.3%, legal 3.2%, sales 2.6%, data_science 2.1%,
product_management 1.1%.

**Occupations** — 803 seed roles, each with 308–311 profiles. Coverage is even by
construction rather than popularity-weighted, so welders, paralegals and nurses are
represented on the same footing as software engineers.

**Geography** — 50 states and territories. California 23.0%, Texas 11.0%, Florida 6.0%,
Colorado 4.0%, North Carolina 3.6%, Arizona 3.6%, Virginia 2.7%, Illinois 2.6%.

**Content** — median 13 skills (p10 8, p90 17), 3 experience entries (p10 1, p90 5),
10 years experience (p10 2, p90 20), age 33 (p10 25, p90 43), target salary $85,000
(p10 $48,000, p90 $145,000). 87.5% have a bio, 94.9% education, 79.6% certifications,
77.7% projects. 47.9% list Remote among their work-location preferences; 5.1% need
sponsorship. Languages: English 99.9%, Spanish 19.9%, then a long tail.

**Sampled demographics** — ethnicity: White 57.5%, Hispanic/Latino 19.0%, Black 13.0%,
Asian 6.3%, Other 1.7%, Native American 1.3%, Middle Eastern 1.2%. Legal status:
Citizen 94.4%, Work Visa 2.7%, Student Visa 2.4%, Permanent Resident 0.5%. Both are drawn
from fixed distributions, independent of profile content.

## How it was built

Roles were enumerated systematically rather than sampled, so coverage is even rather than
popularity-weighted. A vocabulary bank was generated once per occupation (803 roles →
~33,700 distinct skill, tool, specialization and certification terms), then every profile
was seeded with a role, cohort, city, seniority, employer archetype, education path,
career shape and a random subset of that role's vocabulary before generation at
temperature 1.0. Seeding is positional rather than random, so each attribute is visited in
proportion within every role instead of by luck.

Generation produces a richer internal record — used by the validators and the
work-location assigner — which is converted to the stored document before writing; the
internal shape is never persisted. Every profile was schema-validated at write time and
again at merge, near-duplicates were dropped by per-role SimHash over the rendered text,
and anything matching a PII pattern was discarded.

The 148,527 profiles in the newer batch cost **$121.85** and about 4.5 hours at 96-way
concurrency (measured, at off-peak API rates). 40 rows were dropped at merge: 37
near-duplicates, 2 schema failures, 1 possible-PII match.

**Provenance.** The corpus was built in two batches. `syn_`-prefixed IDs (99,995 rows) come
from an earlier run that predates explicit seniority seeding; `syn2_`-prefixed IDs
(148,527 rows) come from the newer run. The two are identical in format and were validated
identically — the only difference is the seniority mix, which is flatter in the older batch
(0.3% staff and executive, versus 5.9% and 2.1% in the newer one). Filter on the prefix if
you need the more balanced subset.

## Intended use

Training and evaluating candidate–job relevance models — cross-encoder rerankers,
bi-encoder retrieval, matching heuristics. Useful anywhere real candidate data is
unavailable or shouldn't be used.