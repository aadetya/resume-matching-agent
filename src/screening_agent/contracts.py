"""Validated boundaries between conversation, retrieval, and screening."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


EvidenceDepth = Literal["presence", "application", "role", "employment", "explicit_usage"]
ExperienceBasis = Literal[
    "not_applicable", "total_employment", "skill_related_employment", "explicit_usage"
]
EvidenceScope = Literal[
    "source_claim",
    "personal_application",
    "role_or_equivalent_work",
    "occupational_employment",
    "all_employment",
    "skill_related_employment",
    "explicit_usage",
]


class RequirementSource(StrictModel):
    source_id: str
    kind: Literal["user_request", "job_document", "structured_request"]
    text: str


class RequirementClause(StrictModel):
    source_id: str
    quote: str = Field(min_length=1, max_length=6000)


class CriterionMeaning(StrictModel):
    criterion_id: str
    conditions: list[RequirementClause] = Field(min_length=1)
    examples: list[RequirementClause] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    requested_depth: EvidenceDepth
    experience_basis: ExperienceBasis
    evidence_scope: EvidenceScope


class EvidenceStandard(StrictModel):
    """Request-specific meaning, compiled before any candidate is inspected."""

    criterion_id: str
    capability: str = Field(min_length=3, max_length=500)
    requested_depth: EvidenceDepth
    experience_basis: ExperienceBasis
    evidence_scope: EvidenceScope
    sufficient_evidence: str = Field(min_length=10, max_length=900)
    equivalence_boundary: str = Field(min_length=10, max_length=700)
    uncertainty_boundary: str = Field(min_length=10, max_length=700)


class Requirements(StrictModel):
    title: str = "Candidate search"
    role: str | None = Field(default=None, max_length=160)
    must_have: list[list[str]] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    minimum_years: float = Field(default=0, ge=0, le=60)
    skill_years: dict[str, float] = Field(default_factory=dict)
    source_text: str = ""
    semantic_brief: str = Field(default="", max_length=6000)
    unresolved: list[str] = Field(default_factory=list)
    capability_skills: list[str] = Field(default_factory=list)
    evidence_standards: list[EvidenceStandard] = Field(default_factory=list)
    standards_signature: str = ""
    requirement_sources: list[RequirementSource] = Field(default_factory=list)
    criterion_meanings: list[CriterionMeaning] = Field(default_factory=list)

    @field_validator("role")
    @classmethod
    def clean_role(cls, value):
        if value is None:
            return None
        value = " ".join(value.split()).strip().casefold()
        if not value:
            raise ValueError("A role must contain text, or be null.")
        return value


class Evidence(StrictModel):
    candidate_id: str
    source_id: str | None = None
    source_path: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    quote: str
    document_sha256: str
    kind: str = "retrieval"


class Candidate(StrictModel):
    candidate_id: str
    source_id: str | None = None
    name: str
    source_path: str
    skills: list[str]
    experience_years: float | None = None
    skill_years: dict[str, float] = Field(default_factory=dict)
    experience_months: int | None = Field(default=None, ge=0)
    claimed_skill_months: dict[str, int] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    entities: list[dict] = Field(default_factory=list)


class CriterionAssessment(StrictModel):
    criterion: str
    mandatory: bool
    status: Literal["supported", "not_demonstrated", "uncertain"]
    reason: str
    evidence: list[Evidence] = Field(default_factory=list)


class Match(StrictModel):
    candidate_id: str
    source_id: str | None = None
    name: str
    score: float = Field(ge=0, le=100)
    eligible: bool
    screening_status: Literal["supported", "needs_review", "does_not_meet"] = "supported"
    components: dict[str, float] = Field(default_factory=dict)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    recommendation: Literal["pending", "hire", "no_hire", "hold"] = "pending"
    recommendation_reason: str = ""
    recommendation_actions: list[str] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)
    requirements_fingerprint: str = ""
    engine_fingerprint: str = ""
    assessments: list[CriterionAssessment] = Field(default_factory=list)
    reviewed_assessments: list[CriterionAssessment] = Field(default_factory=list)
    retrieval_leads: list[dict] = Field(default_factory=list)


class RetrievalBatch(StrictModel):
    requirements_fingerprint: str
    engine_fingerprint: str
    candidate_scores: dict[str, tuple[float, float]]
    evidence_scores: list[float]
    corpus_size: int
    retrieved_count: int
    timings: dict[str, float] = Field(default_factory=dict)
    backend: str
    criterion_scores: dict[str, dict[str, float]] = Field(default_factory=dict)
    retrieval_audit: dict = Field(default_factory=dict)


class SearchResult(StrictModel):
    matches: list[Match]
    corpus_size: int
    retrieved_count: int
    timings: dict[str, float] = Field(default_factory=dict)
    backend: str
    not_shortlisted_count: int = 0
    retrieval_audit: dict = Field(default_factory=dict)


class FileArguments(StrictModel):
    filepath: str | None = None
    directory: str | None = None
    keyword: str | None = None
    content: str | None = None
    extension: str | None = None


class Plan(StrictModel):
    action: Literal[
        "search",
        "refine",
        "compare",
        "explain",
        "questions",
        "deep_screen",
        "finalize",
        "approve",
        "help",
        "file_tool",
    ]
    requirements: Requirements | None = None
    changed_criterion_ids: list[str] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)
    top_n: int = Field(default=3, ge=1, le=10)
    explanation: str = ""
    file_tool_name: Literal["read_file", "list_files", "write_file", "search_in_file"] | None = None
    file_arguments: FileArguments | None = None
