"""Validated source identity and membership, independent of evaluation labels."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

RESUME_EXTENSIONS = frozenset({".pdf", ".docx", ".txt"})
MANIFEST_PATH = "data/corpus_manifest.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths(root: Path, *, recursive: bool = False) -> list[Path]:
    directory = root / "data/resumes"
    paths = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(path for path in paths if path.suffix.lower() in RESUME_EXTENSIONS)


@dataclass(frozen=True)
class SourceCatalog:
    snapshot_date: date
    files: tuple[dict[str, str], ...]
    provenance: dict[str, str | int]


def _field(record: dict, key: str) -> str:
    value = record.get(key)
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"Corpus manifest requires a nonempty {key} without control characters.")
    return value


def load_catalog(root: Path) -> SourceCatalog | None:
    """Version 2 is an explicit allowlist; version 1 retains legacy discovery.

    Only source provenance fields are read. Scoring judgments have no runtime
    contract here and are never loaded from ground-truth or evaluation files.
    """
    manifest_path = root / MANIFEST_PATH
    if not manifest_path.exists():
        return None
    if manifest_path.is_symlink():
        raise ValueError("The corpus manifest cannot be a symlink.")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("The corpus manifest must be an object.")
    version = raw.get("schema_version", 1)
    if version == 1:
        return None
    if version != 2:
        raise ValueError("Unsupported corpus manifest schema_version; expected 1 or 2.")
    snapshot = _field(raw, "snapshot_date")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", snapshot):
        raise ValueError("Corpus snapshot_date must be an ISO date (YYYY-MM-DD).")
    snapshot_date = date.fromisoformat(snapshot)
    dataset_id = _field(raw, "dataset_id")
    rows = raw.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("The version 2 corpus manifest requires a nonempty files list.")
    entries = []
    identities: dict[str, set[str]] = {key: set() for key in ("candidate_id", "source_id", "path")}
    source_directory = root / "data/resumes"
    if source_directory.is_symlink() or not source_directory.resolve().is_relative_to(root):
        raise ValueError("The resume directory cannot be a symlink.")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Corpus manifest files must contain objects.")
        entry = {
            key: _field(row, key)
            for key in ("candidate_id", "source_id", "display_name", "path", "split", "sha256")
        }
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", entry["candidate_id"]):
            raise ValueError("Corpus candidate_id must be a stable identifier, not a file path.")
        if entry["split"] != "production":
            raise ValueError("Only production files may enter this runtime corpus manifest.")
        relative = Path(entry["path"])
        path = root / relative
        if (
            relative.is_absolute()
            or relative.as_posix() != entry["path"]
            or relative.parent != Path("data/resumes")
            or path.suffix.lower() not in RESUME_EXTENSIONS
            or path.is_symlink()
            or path.resolve().parent != source_directory.resolve()
        ):
            raise ValueError("Manifest sources must be direct, nonsymlink files in data/resumes.")
        if not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]):
            raise ValueError("Corpus source sha256 must be a lowercase SHA-256 digest.")
        for key, seen in identities.items():
            if entry[key] in seen:
                raise ValueError(f"Duplicate corpus {key}: {entry[key]}")
            seen.add(entry[key])
        if not path.is_file() or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Corpus source hash mismatch or missing file: {entry['path']}")
        entries.append(entry)
    actual = {path.relative_to(root).as_posix() for path in source_paths(root, recursive=True)}
    if actual != identities["path"]:
        raise ValueError(
            "Corpus manifest membership differs from data/resumes; unlisted files are not indexed."
        )
    provenance: dict[str, str | int] = {
        "path": MANIFEST_PATH,
        "sha256": file_sha256(manifest_path),
        "schema_version": 2,
        "dataset_id": dataset_id,
    }
    if "dataset_split" in raw:
        provenance["dataset_split"] = _field(raw, "dataset_split")
    return SourceCatalog(snapshot_date, tuple(entries), provenance)
