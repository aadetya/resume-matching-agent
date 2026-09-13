"""Declared screening policy. Ranking points are not hiring probabilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import Field, model_validator

from .contracts import Match, Requirements, StrictModel

ENGINE_VERSION = "open-evidence-review-v3"


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def requirements_fingerprint(requirements: Requirements) -> str:
    values = requirements.model_dump(exclude={"title", "source_text", "unresolved"})
    values["semantic_brief"] = requirements.semantic_brief or requirements.source_text
    values["must_have"] = sorted(sorted(group) for group in requirements.must_have)
    values["nice_to_have"] = sorted(requirements.nice_to_have)
    return fingerprint(values)


class ScreeningPolicy(StrictModel):
    version: str = "criteria-evidence-v2"
    rrf_k: float = Field(default=60, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def supported_version(self):
        if self.version != "criteria-evidence-v2":
            raise ValueError("Unsupported screening policy version")
        return self


def load_policy(root: Path) -> ScreeningPolicy:
    path = Path(root) / "config/screening_policy.yaml"
    return (
        ScreeningPolicy.model_validate(yaml.safe_load(path.read_text()))
        if path.exists()
        else ScreeningPolicy()
    )


def recommend(match: Match, requirements: Requirements | None) -> Match:
    """Apply one evidence rule after deep review; rank and score are irrelevant."""
    result = match.model_copy(deep=True)
    ledger = result.reviewed_assessments or result.assessments
    unresolved = [row for row in ledger if row.mandatory and row.status != "supported"]
    has_criteria = (
        any(r.mandatory for r in ledger)
        if requirements is None
        else bool(
            requirements.role
            or requirements.must_have
            or requirements.minimum_years
            or requirements.skill_years
        )
    )
    supported = [row for row in ledger if row.mandatory and row.status == "supported"]
    labels = "; ".join(row.criterion for row in unresolved)
    if result.screening_status == "does_not_meet":
        result.recommendation = "no_hire"
        result.recommendation_reason = (
            "The recorded evidence fails a mandatory requirement: "
            + (labels or "; ".join(result.gaps))
            + "."
        )
        result.recommendation_actions = [
            "Reconsider only if the source evidence is corrected or the job requirement changes."
        ]
    elif result.gaps or unresolved or not has_criteria or not ledger:
        result.recommendation = "hold"
        result.recommendation_reason = (
            f"{len(supported)} of {len(supported) + len(unresolved)} mandatory criteria are supported. "
            + (
                "Resolve the reviewed evidence for: " + labels + "."
                if unresolved
                else "Define mandatory job criteria before making a screening decision."
            )
        )
        result.recommendation_actions = [
            "Review the cited evidence for "
            + row.criterion
            + "; clarify the unresolved point before deciding."
            for row in unresolved
        ] or ["Specify the mandatory job criteria."]
        if not result.improvement_suggestions:
            result.improvement_suggestions = [
                "Specify the job requirements and verify the candidate's own dated work in an interview."
            ]
    else:
        result.recommendation = "hire"
        result.recommendation_reason = (
            f"All {len(supported)} mandatory criteria are supported by the reviewed evidence: "
            + "; ".join(row.criterion for row in supported)
            + ". Advance to interview verification."
        )
        result.recommendation_actions = [
            "Invite the candidate to an interview to verify the cited work.",
            "Prepare the interview guide when you are ready to assess personal ownership and technical decisions.",
        ]
    return result
