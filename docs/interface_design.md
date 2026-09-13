# Review Interface

## Design

I use the conversation for requests and concise results, with separate views for candidate records, comparison, evidence, recommendations, interview questions and reports. The desktop layout places chat beside the review workspace. Narrow screens use separate Chat and Results views.

Status colours have written labels. Candidate scores are retrieval relevance values, while duration findings retain their own evidence basis and units. A score is not a hiring probability.

## Requests and progress

The interface acknowledges a request before computation and disables conflicting actions on the same review. Progress comes from completed LangGraph nodes through `MatchingSession.stream()`; the interface does not invent a completion percentage.

The chat response identifies the action and result counts. Detailed records remain in the results views. Activity shows graph steps, tool outcomes and elapsed time, with technical details available separately.

## Evidence and comparison

The requirements panel exposes the shared evidence standards under “How these requirements will be assessed.” It shows exact request conditions, examples, interpretation assumptions, evidence scope and the shared support boundaries for the current revision.

Initial candidate evidence shows relevant source passages with an Unverified label. Detailed screening replaces that view with requirement findings, experience calculations and audited work context. Recommendation has a comparative memo, justified priority groups and decision contingencies.

Initial cards show retrieval points and source leads. Reviewed cards promote the number of supported requirements and the candidate's strengths or gaps. Decision cards lead with the action and its reason. The retrieval score stays available for inspection without competing with the reviewed findings.

The candidate record separates Requirements, Work and results, Employment history, Projects and Review notes. Requirements open first; the other sections expand on request. Employment appears as separate dated entries in recent-first order, with the original quotations under each entry. Projects stay separate from employment. Missing dates remain explicit rather than being guessed.

Detailed review can reorder the shortlist by reviewed support. Compare shows the current first three candidates even when no comparison action has been run. An explicit comparison preserves the requested selection and uses the current findings. Comparison uses real tables that scroll horizontally on narrow screens. Source quotations are escaped for display and retain their line breaks.

Incomplete duration coverage is labelled as a lower bound. Missing duration remains unresolved. The application retains changed findings and audit notes so the reviewer can inspect the reason for uncertainty. The source inventory, explicit employment decisions and audit disputes remain inspectable. A status disagreement stays uncertain; agreed support can retain a corrected explanation. Pending execution is shown separately from a qualification gap.

## Recommendations and interviews

Recommendations begin with the checked summary and priority groups, followed by expandable comparisons and decision groups. Each group shows reasons, blockers and next actions. Candidate links return to the supporting evidence. A next action asks for review or clarification of the unresolved point.

Interview preparation requires its own request. Questions include purpose, follow-up, assessment guidance and source evidence. The guide receives the same frozen criterion meaning and experience basis as screening, so an overall-employment requirement stays separate from software-only or skill-specific years. Detailed screening does not generate questions automatically.

A successful requirement change clears stale comparisons and questions. Candidate-specific model failures preserve completed results as provisional and list pending IDs. Source corruption preserves the preceding complete state.

## Reports and saved reviews

The Report view shows Markdown and offers report and complete-state JSON downloads after a search. Filenames include requirement revision and content identity. Starting a new review clears the current download links.

SQLite holds the full graph state. Short interface conversations are stored under `runtime/conversations/`. Saved review IDs restore compatible sessions; model or engine changes create a different review identity.

## Setup and validation

The header reports whether `OPENAI_API_KEY` is configured. A missing key produces setup instructions; a configured key does not itself prove the API is reachable. See [running without a key](../README.md#running-without-an-openai-key).

The [conversation scenarios](conversation_scenarios.md) cover eight review flows. Local tests check stage presentation, comparison fallback, source escaping, report exports, refinement and failure preservation. The [workflow verification](../reports/workflow/verification.json) records a connected 200-profile review, ten completed detailed reviews, a populated comparison and five explicitly requested interview questions. Its source checks establish provenance for that run, not independent semantic accuracy. Measured times and remaining weaknesses are in [evaluation](ranking_validation.md).
