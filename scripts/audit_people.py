"""Check source preservation, date arithmetic and split isolation for People."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path

from import_people import FIELDS, FORBIDDEN, ROOT, SNAPSHOT, blocks, date_audit, dump, sha

from screening_agent.file_tools import read_file
from screening_agent.metadata import MetadataExtractor


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def audit(root: Path) -> dict:
    extractor = MetadataExtractor(root, backend="spacy", snapshot_date=date.fromisoformat(SNAPSHOT))
    records, ids, content_hashes, source_ids = [], {}, {}, {}
    for split in ("active", "holdout"):
        rows = [
            json.loads(line)
            for line in (root / f"data/people/{split}.jsonl").read_text().splitlines()
        ]
        manifest_path = root / (
            "data/corpus_manifest.json"
            if split == "active"
            else "data/holdout/corpus_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        if manifest["snapshot_date"] != SNAPSHOT:
            raise ValueError("The reviewed source snapshot differs from the manifest cutoff")
        files = {row["candidate_id"]: row for row in manifest["files"]}
        ids[split], content_hashes[split], source_ids[split] = set(), set(), set()
        for row in rows:
            cid, profile = row["candidate_id"], row["profile"]
            entry = files[cid]
            path = root / entry["path"]
            read = read_file(str(path))
            if not read["ok"]:
                raise ValueError(f"Unreadable export {cid}: {read['error']}")
            text = read["content"]
            # Page footer is generated layout material, never candidate evidence.
            normalized = normalize(
                re.sub(
                    r"(?m)^\s*\d+\s*$", "", text.replace("Synthetic profile | People dataset", "")
                )
            )
            expected = normalize("\n".join(value for _, value in blocks(row)))
            keys = set(profile["user"])
            for section in ("experience", "education", "projects", "certifications"):
                for item in profile[section]:
                    keys.update(item)
                    if set(item) != set(FIELDS[section]):
                        raise ValueError(f"Unexpected fields in {cid}/{section}")
            candidate = extractor.extract(
                cid,
                text,
                entry["path"],
                entry["sha256"],
                source_id=row["source_id"],
                display_name=row["display_name"],
            )
            raw_oracle = date_audit(profile)
            excluded_training = [
                entry
                for entry in profile["experience"]
                if "bootcamp" in str(entry.get("title", "")).lower().split()
                and {"student", "participant"} & set(str(entry.get("title", "")).lower().split())
            ]
            oracle = date_audit(
                {
                    **profile,
                    "experience": [
                        entry for entry in profile["experience"] if entry not in excluded_training
                    ],
                }
            )
            checks = {
                "source_sha256": sha(path) == entry["sha256"],
                "source_id_retained": entry["source_id"] == row["source_id"],
                "source_id_absent_from_screening_text": row["source_id"] not in text,
                "administrative_labels_absent_from_screening_text": not bool(
                    re.search(
                        r"(?i)\b(?:visa_constrained|career_changer|sparse_profile|seed_role|source_id)\b|\bcohort\s*:",
                        text,
                    )
                ),
                "fields_allowlisted": not bool(keys & FORBIDDEN),
                "export_text_complete": normalized == expected,
                "tenure_matches_independent_calendar_oracle": candidate.experience_years
                == oracle["experience_years"],
                "citation_spans_exact": all(
                    text[e.start : e.end] == e.quote for e in candidate.evidence
                ),
                "neutral_display_name": candidate.name == row["display_name"],
            }
            if not checks["export_text_complete"]:
                # Store small differences for debugging, never silently accept truncated content.
                mismatch = next(
                    (
                        i
                        for i, (a, b) in enumerate(zip(expected, normalized, strict=False))
                        if a != b
                    ),
                    min(len(expected), len(normalized)),
                )
                differences = {
                    "offset": mismatch,
                    "expected": expected[max(0, mismatch - 60) : mismatch + 160],
                    "extracted": normalized[max(0, mismatch - 60) : mismatch + 160],
                }
            else:
                differences = None
            records.append(
                {
                    "candidate_id": cid,
                    "split": split,
                    "format": path.suffix[1:],
                    "checks": checks,
                    "expected_years": oracle["experience_years"],
                    "upstream_experience_list_years": raw_oracle["experience_years"],
                    "excluded_classroom_entries": [entry["title"] for entry in excluded_training],
                    "extracted_years": candidate.experience_years,
                    "date_warnings": oracle["warnings"],
                    "extraction_warnings": candidate.warnings,
                    "text_difference": differences,
                }
            )
            ids[split].add(cid)
            source_ids[split].add(row["source_id"])
            content_hashes[split].add(row["content_hash"])
    original = json.loads((root / "data/regression/data/corpus_manifest.json").read_text())
    preserved = all(
        sha(root / "data/regression" / entry["path"]) == entry["sha256"]
        for entry in original["files"]
    )
    result = {
        "scope": f"All {len(records)} exported profiles; fields, file content, citations and date arithmetic. Does not establish hiring accuracy or authentic source-document layout diversity.",
        "counts": dict(Counter(row["split"] for row in records)),
        "formats": dict(Counter(row["format"] for row in records)),
        "checks_passed": {
            key: sum(row["checks"][key] for row in records) for key in records[0]["checks"]
        },
        "isolation": {
            "candidate_ids_disjoint": not bool(ids["active"] & ids["holdout"]),
            "source_ids_disjoint": not bool(source_ids["active"] & source_ids["holdout"]),
            "normalized_profile_hashes_disjoint": not bool(
                content_hashes["active"] & content_hashes["holdout"]
            ),
            "original100_files_preserved": preserved,
        },
        "failures": [row for row in records if not all(row["checks"].values())],
        "profiles": records,
    }
    dump(root / "reports/people_import_audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = audit(args.root)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"profiles"}}, indent=2
        )
    )
    raise SystemExit(bool(result["failures"]) or not all(result["isolation"].values()))
