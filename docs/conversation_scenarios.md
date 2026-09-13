# Conversation Scenarios

## Setup

Start a connected session with the current Luna model and the active index. These are review scenarios and expected workflow properties, not claims that every model response or candidate ordering is predetermined.

The checks use visible state so changes in interpretation can be inspected without requiring a fixed candidate order.

## 1. Search and compare

```text
Find candidates with React and 3+ years overall experience.
Compare the top 3 candidates.
```

Check that React is mandatory, three years refers to total employment, and the comparison uses the selected candidates and current criteria. Inspect source citations and uncertainty rather than expecting particular candidate IDs.

## 2. Refine a requirement

```text
Make TypeScript a must-have.
Make React optional.
Explain the ranking changes.
```

The new requirement version should preserve unrelated conditions, move React out of mandatory groups and record shortlist membership or position changes. Inspect unchanged criterion meanings and standards for exact preservation. Candidate-specific failures should list pending IDs with provisional completed results; source corruption should retain the preceding complete state.

## 3. Mandatory alternatives and preferences

```text
Require Python and either PostgreSQL or MySQL. Docker is preferred.
Now make Docker mandatory.
```

Inspect AND between groups, OR within the database group and the preference-to-mandatory change. The model should not treat independently required skills as alternatives.

## 4. Experience followed by role

```text
Find someone with 3+ years of experience.
The candidate must also be a software developer.
Remove the role requirement.
```

The second request should preserve three years of total experience and add a role without inventing a technology stack. The third removes the role and keeps the experience threshold.

## 5. Skill presence and duration

```text
need someone with 3+ years python and sql
Show the evidence for the top candidate's Python and SQL experience.
```

Inspect separate basic skill and skill-duration findings. A skills-list claim can support presence while duration remains unknown. Check accepted work associations, calendar dates and caps. A role-related period is not verified uninterrupted use.

## 6. Detailed review and recommendation

```text
Run the second screening round.
Generate final recommendations.
```

Initial screening should contain unverified passage leads and no qualification judgments. Detailed screening should produce the first full assessment, source inventory, per-employment decisions and audited findings. Any disagreement between the assessment draft and audit remains visible and uncertain. Recommendation should add comparisons, justified priority groups and what would change each decision. Inspect whether the requested depth stays consistent across candidates.

## 7. Interview, export and resume

```text
Generate interview questions for the top candidate.
Save the report to reports/generated/screening.md
```

Questions should refer to the current evidence and appear only after the request. Download Markdown and JSON, note the review ID, restart the app and resume. Model or code changes can make the old checkpoint incompatible; old exports remain records of their own engine.

## 8. File input and invalid references

```text
Read data/job_descriptions/backend_platform.txt
Use the job description I just opened to create the shortlist.
Compare P_DOES_NOT_EXIST with the top candidate.
```

The document read should provide source context for the next request. An unknown ID must produce a clarification or error rather than a fabricated candidate. A failed file read must not leave an older document silently active.

## Evaluation scope

Local tests check contracts, source integrity, state transitions and failure preservation. A live scenario also needs manual inspection of evidence and criteria. Neither a successful sequence nor consistent output proves semantic correctness. The [architecture notes](architecture.md) describe the current contracts, and the [evaluation notes](ranking_validation.md) distinguish workflow verification from semantic accuracy.
