"""Shared source-integrity checks and recommendation application for completed reviews."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .contracts import Match, Requirements
from .model_config import generation_options
from .policy import fingerprint, requirements_fingerprint

REVIEW_VERSION = "evidence-only-review-v4"


class ReviewIntegrity:
    def __init__(self, *, client=None, model: str | None = None):
        self.client = client
        self.model = model
        self.mode = "openai" if client is not None else "local"
        self.signature = fingerprint(
            {
                "version": REVIEW_VERSION,
                "mode": self.mode,
                "model": model,
                "generation_options": generation_options(model) if model else {},
                "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
        )

    def verify(
        self, reviews: list[dict], index, matches: list[Match], requirements: Requirements
    ) -> None:
        # Recheck the complete physical corpus before using cached review facts.
        # A changed unreviewed source or identity manifest also invalidates selection.
        index._check_sources()
        if len(reviews) != len(matches) or {r["candidate_id"] for r in reviews} != {
            m.candidate_id for m in matches
        }:
            raise ValueError("Final screening requires a complete current detailed review.")
        for review in reviews:
            body = {k: v for k, v in review.items() if k != "integrity"}
            packet = review["packet"]
            if (
                fingerprint(body) != review["integrity"]
                or review["reviewer_signature"] != self.signature
            ):
                raise ValueError(
                    "Detailed-review content or model configuration changed; review again."
                )
            if (
                packet["requirements_fingerprint"] != requirements_fingerprint(requirements)
                or packet["engine_fingerprint"] != index.engine_fingerprint
            ):
                raise ValueError("Detailed review belongs to different criteria or engine.")
            record = index.documents[review["candidate_id"]]
            if packet["source_sha256"] != record["sha256"]:
                raise ValueError("Detailed-review source changed; rebuild and review again.")
            for ev in [p["evidence"] for p in packet["passages"]] + [
                e
                for row in [
                    *review["criteria"],
                    *review["observations"],
                    *review.get("claim_checks", []),
                    *review.get("rejected_observations", []),
                ]
                for e in row["evidence"]
            ]:
                if (
                    ev["candidate_id"] != review["candidate_id"]
                    or ev["source_path"] != record["source_path"]
                    or ev["source_id"] != record.get("source_id")
                    or ev["document_sha256"] != record["sha256"]
                    or record["text"][ev["start"] : ev["end"]] != ev["quote"]
                ):
                    raise ValueError("Detailed review contains an invalid source citation.")

    def apply_recommendations(
        self, reviews: list[dict], matches: list[dict], requirements: Requirements | None = None
    ) -> list[dict]:
        from .policy import recommend

        by_id = {r["candidate_id"]: r for r in reviews}
        output = []
        for match in self.apply_evidence(reviews, matches, requirements):
            review = by_id[match["candidate_id"]]
            match = recommend(Match.model_validate(match), requirements).model_dump(mode="json")
            gaps = [
                c for c in review["changes"] if c["mandatory"] and c["review_status"] != "supported"
            ]
            if gaps and match["recommendation"] == "hire":
                match["recommendation"] = "hold"
                match["recommendation_reason"] = (
                    "Detailed evidence review needs human verification: "
                    + "; ".join(c["reason"] for c in gaps)
                )
            if not review["packet"]["coverage"]["complete"] and match["recommendation"] == "hire":
                match["recommendation"] = "hold"
                match["recommendation_reason"] = (
                    "Some resume text was outside the review budget. Verify the unreviewed evidence before advancing."
                )
                match["recommendation_actions"] = [
                    "Review the source passages outside the assessed coverage before advancing."
                ]
            output.append(match)
        return output
