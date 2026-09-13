"""Agent-visible tools with validated arguments and confined file access."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field, ValidationError

from .contracts import Requirements, StrictModel
from .criteria import RequirementCompiler
from .planner import PlanningError, validate_requirements


class ReadArgs(StrictModel):
    filepath: str


class ListArgs(StrictModel):
    directory: str
    extension: str | None = None


class WriteArgs(ReadArgs):
    content: str = Field(max_length=2_000_000)


class SearchFileArgs(ReadArgs):
    keyword: str = Field(min_length=1)


class RequirementArgs(StrictModel):
    jd: str = Field(min_length=1, max_length=100_000)


class RagArgs(StrictModel):
    requirements: Requirements
    top_k: int = Field(default=10, ge=1, le=10)


class ComparisonArgs(StrictModel):
    candidate_ids: list[str] = Field(min_length=2, max_length=10)


class QuestionArgs(StrictModel):
    candidate_id: str


TOOL_SCHEMAS = {
    "read_file": ReadArgs,
    "list_files": ListArgs,
    "write_file": WriteArgs,
    "search_in_file": SearchFileArgs,
    "extract_requirements": RequirementArgs,
    "rag_search": RagArgs,
    "compare_candidates": ComparisonArgs,
    "generate_interview_questions": QuestionArgs,
}
TOOL_DESCRIPTIONS = {
    "read_file": "Extract text and metadata from a PDF, DOCX or TXT file inside the project.",
    "list_files": "List files with optional extension filtering inside the project.",
    "write_file": "Write a UTF-8 report under reports/generated; existing source files cannot be changed.",
    "search_in_file": "Find a case-insensitive keyword and its source context in a supported document.",
    "extract_requirements": "Extract and validate mandatory OR groups, preferred skills and experience thresholds.",
    "rag_search": "Rank every resume for relevance and return up to ten provisional candidates with unverified source passages. Qualification assessment runs separately during detailed screening.",
    "compare_candidates": "Compare explicit candidate IDs using the current requirements.",
    "generate_interview_questions": "Generate candidate-specific questions tied to evidence and gaps.",
}


def detect_file_request(text: str, report: str = "") -> dict | None:
    """Recognize explicit file commands without a model call in an active session."""
    if text.startswith("/tool "):
        parts = text.split(maxsplit=2)
        if len(parts) != 3:
            raise PlanningError("Use /tool tool_name followed by a JSON argument object.")
        try:
            arguments = json.loads(parts[2])
        except json.JSONDecodeError as exc:
            raise PlanningError("Tool arguments must be a valid JSON object.") from exc
        if not isinstance(arguments, dict):
            raise PlanningError("Tool arguments must be a JSON object.")
        return {"name": parts[1], "arguments": arguments}
    found = re.fullmatch(
        r"\s*(?:list|show)\s+(?:the\s+)?files(?:\s+in)?\s+[\"']?([^\"']+?)[\"']?\s*", text, re.I
    )
    if found:
        return {"name": "list_files", "arguments": {"directory": found.group(1).strip()}}
    found = re.fullmatch(r"\s*(?:read|open)\s+[\"']?(.+\.(?:pdf|docx|txt))[\"']?\s*", text, re.I)
    if found:
        return {"name": "read_file", "arguments": {"filepath": found.group(1)}}
    found = re.fullmatch(
        r"\s*search\s+['\"]?(.+?)['\"]?\s+in\s+['\"]?(.+\.(?:pdf|docx|txt))['\"]?\s*", text, re.I
    )
    if found:
        return {
            "name": "search_in_file",
            "arguments": {"keyword": found.group(1), "filepath": found.group(2)},
        }
    found = re.fullmatch(r"\s*(?:save|export)\s+(?:the\s+)?report(?:\s+to\s+(.+))?\s*", text, re.I)
    if found:
        if not report:
            raise PlanningError("Generate a screening report before saving it.")
        target = found.group(1) or "reports/generated/screening_report.md"
        return {"name": "write_file", "arguments": {"filepath": target.strip(), "content": report}}
    return None


class ToolRegistry:
    def __init__(self, root: str | Path, index, planner):
        self.root = Path(root).resolve()
        self.index = index
        self.planner = planner
        self.requirement_compiler = RequirementCompiler(
            self.root, client=getattr(planner, "client", None), model=getattr(planner, "model", "")
        )

    @property
    def definitions(self) -> list[dict]:
        return [
            {
                "name": name,
                "description": TOOL_DESCRIPTIONS[name],
                "parameters": schema.model_json_schema(),
            }
            for name, schema in TOOL_SCHEMAS.items()
        ]

    def _path(self, value: str, *, write: bool = False) -> str:
        raw = Path(value).expanduser()
        target = (raw if raw.is_absolute() else self.root / raw).resolve()
        allowed = self.root / "reports/generated" if write else self.root
        if not target.is_relative_to(allowed):
            raise PlanningError(
                "Writes are limited to reports/generated."
                if write
                else "The requested path is outside the project workspace."
            )
        return str(target)

    def invoke(
        self,
        name: str,
        arguments: dict,
        *,
        requirements: Requirements | None = None,
        proposal: Requirements | None = None,
        retrieval_only: bool = False,
        review: dict | None = None,
        context_matches: list[dict] | None = None,
        changed_criterion_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        event = {
            "event_id": uuid4().hex,
            "tool": name,
            "arguments": {
                key: (
                    f"{len(value)} characters"
                    if key in {"content", "jd"} and isinstance(value, str)
                    else value
                )
                for key, value in arguments.items()
            },
        }
        try:
            if name not in TOOL_SCHEMAS:
                raise PlanningError(f"Unknown tool: {name}.")
            args = TOOL_SCHEMAS[name].model_validate(arguments)
            if name in {"read_file", "list_files", "write_file", "search_in_file"}:
                import fs_tools

                values = args.model_dump()
                key = "directory" if name == "list_files" else "filepath"
                values[key] = self._path(values[key], write=name == "write_file")
                try:
                    result = getattr(fs_tools, name)(**values)
                except fs_tools.FileToolError as exc:
                    raise PlanningError(f"{exc.code}: {exc.message}") from exc
                if isinstance(result, dict) and (
                    result.get("ok") is False or result.get("status") == "error"
                ):
                    raise PlanningError(str(result.get("error", "File tool failed.")))
            elif name == "extract_requirements":
                if proposal is None:
                    plan = self.planner.plan(
                        args.jd,
                        requirements,
                        candidates=self.index.candidates,
                        shortlist=[],
                        history=[],
                    )
                    proposal = plan.requirements
                    changed_criterion_ids = plan.changed_criterion_ids
                if proposal is None:
                    raise PlanningError("No requirements were extracted from the job description.")
                compiled, metadata = self.requirement_compiler.compile_with_metadata(
                    validate_requirements(proposal, self.planner.vocabulary),
                    previous=requirements,
                    changed_criterion_ids=changed_criterion_ids,
                )
                event.update(metadata)
                result = compiled.model_dump(mode="json")
            elif name == "rag_search":
                # Direct tool calls receive the same request contract as graph calls.
                compiled, metadata = self.requirement_compiler.compile_with_metadata(
                    validate_requirements(args.requirements, self.planner.vocabulary)
                )
                args.requirements = compiled
                event.update(metadata)
                if not retrieval_only:
                    batch = self.index.retrieve(args.requirements)
                    batch = self.index.expand_retrieval(batch, args.requirements)
                    operation = self.index.rank(batch, args.requirements, top_k=args.top_k)
                else:
                    operation = self.index.retrieve(args.requirements)
                result = operation.model_dump(mode="json")
            elif name == "compare_candidates":
                if requirements is None:
                    raise PlanningError("Run a search before comparing candidates.")
                self._candidate_ids(args.candidate_ids)
                result = self.deep_reviewer.compare(
                    self.index, args.candidate_ids, requirements, context_matches
                )
            else:
                if requirements is None:
                    raise PlanningError("Run a search before generating interview questions.")
                self._candidate_ids([args.candidate_id])
                from .interview import generate

                if review is None:
                    raise PlanningError(
                        "Run detailed screening for this candidate before preparing the interview guide."
                    )

                result, usage = generate(
                    self.index,
                    args.candidate_id,
                    requirements,
                    client=getattr(self.planner, "client", None),
                    model=getattr(self.planner, "model", None),
                    review=review,
                )
                event["model"] = getattr(self.planner, "model", None)
                event["usage"] = usage
            event.update(
                status="success", elapsed_ms=round((time.perf_counter() - started) * 1000, 2)
            )
            if isinstance(result, list):
                event["result_summary"] = {"item_count": len(result)}
                if event.get("usage"):
                    event["result_summary"].update(
                        model=event.get("model"),
                        model_input_tokens=event["usage"].get("input_tokens", 0),
                        model_output_tokens=event["usage"].get("output_tokens", 0),
                    )
            elif name == "rag_search":
                event["result_summary"] = {
                    key: result[key] for key in ("corpus_size", "retrieved_count", "backend")
                }
                if "matches" in result:
                    event["result_summary"]["shortlist_count"] = len(result["matches"])
                event["result_summary"]["stage"] = (
                    "retrieval" if retrieval_only else "search_and_rank"
                )
            elif isinstance(result, dict):
                event["result_summary"] = {"fields": list(result)}
            return {"ok": True, "result": result, "event": event}
        except (PlanningError, ValidationError, OSError, RuntimeError, ValueError) as exc:
            event.update(
                status="error",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                error=str(exc),
            )
            return {"ok": False, "error": str(exc), "event": event}

    def _candidate_ids(self, ids: list[str]) -> None:
        if len(ids) != len(set(ids)):
            raise PlanningError("Candidate IDs must be distinct.")
        if set(ids) - self.index.candidates.keys():
            raise PlanningError("One or more candidate IDs do not exist in the index.")
