"""Append a reproducible coverage sample without replacing existing profiles.

This is dataset selection, not qualification or evaluation. Source mentions
determine sampling coverage; the application independently assesses evidence.
Run against a staging copy and audit it before switching the running index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from scripts.import_people import (
    ROOT,
    SOURCE_SHA,
    blocks,
    content_key,
    date_audit,
    dump,
    render_docx,
    render_pdf,
    sanitize,
    sha,
)

SEED = "people-expansion-v1-20260911"
ROLE_GROUPS = {
    "software": [
        "graduate software engineer",
        "junior software engineer",
        "software engineering intern",
        "tooling engineer",
        "fintech engineer",
        "payments engineer",
    ],
    "full_stack": [
        "full stack developer",
        "web developer",
        "e-commerce engineer",
        "typescript developer",
        "node js developer",
    ],
    "backend": ["backend engineer", "python developer", "api engineer"],
    "frontend": ["frontend engineer", "react developer", "react native developer"],
    "data": ["data engineer", "big data engineer", "streaming data engineer"],
    "devops": ["devops engineer", "site reliability engineer", "mlops engineer"],
    "ml": ["machine learning engineer"],
}
QUOTAS = {
    "python_react": 20,
    "software": 20,
    "full_stack": 20,
    "backend": 10,
    "frontend": 10,
    "data": 8,
    "devops": 7,
    "ml": 5,
}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mentions(profile: dict) -> set[str]:
    text = json.dumps(profile, ensure_ascii=False)
    return {
        skill
        for skill in ("python", "react", "sql", "typescript", "docker", "aws", "kubernetes")
        if re.search(r"\b" + skill + r"\b", text, re.I)
    }


def coverage(records: list[dict]) -> dict:
    found = [mentions(row["profile"]) for row in records]
    return {
        "profiles": len(records),
        "python": sum("python" in s for s in found),
        "react": sum("react" in s for s in found),
        "python_and_react": sum({"python", "react"} <= s for s in found),
        "python_react_sql": sum({"python", "react", "sql"} <= s for s in found),
        "react_typescript": sum({"react", "typescript"} <= s for s in found),
        "python_sql": sum({"python", "sql"} <= s for s in found),
        "docker_kubernetes": sum({"docker", "kubernetes"} <= s for s in found),
    }


def choose(pool: list[dict], existing: list[dict], quotas: dict[str, int]) -> list[dict]:
    """Balance source roles, career cohorts and seniority with deterministic ties."""
    used_ids = {r["source_id"] for r in existing}
    used_content = {r["content_hash"] for r in existing}
    picked = []
    for group, count in quotas.items():
        candidates = [
            r
            for r in pool
            if (
                {"python", "react"} <= mentions(r["profile"])
                and r["family"] in {"software", "full_stack", "frontend", "backend"}
                if group == "python_react"
                else r["family"] == group
            )
        ]
        role_counts, cohort_counts, seniority_counts = Counter(), Counter(), Counter()
        for _ in range(count):
            available = [
                r
                for r in candidates
                if r["source_id"] not in used_ids and r["content_hash"] not in used_content
            ]
            if not available:
                raise ValueError(f"Insufficient distinct profiles for sampling stratum {group}")
            row = min(
                available,
                key=lambda r: (
                    role_counts[r["seed_role"]],
                    cohort_counts[r["cohort"]],
                    seniority_counts[r["seniority"]],
                    r["selection_key"],
                ),
            )
            picked.append({**row, "selection_stratum": group})
            used_ids.add(row["source_id"])
            used_content.add(row["content_hash"])
            role_counts[row["seed_role"]] += 1
            cohort_counts[row["cohort"]] += 1
            seniority_counts[row["seniority"]] += 1
    return picked


def select(source: Path, root: Path) -> tuple[list[dict], dict]:
    import pyarrow.parquet as pq

    if sha(source) != SOURCE_SHA:
        raise ValueError("Source checksum differs from the reviewed People release")
    active = rows(root / "data/people/active.jsonl")
    heldout = rows(root / "data/people/holdout.jsonl")
    if len(active) != 100 or len(heldout) != 50:
        raise ValueError(
            "This extension requires the preserved 100 active and 50 evaluation profiles"
        )
    by_role = {role: family for family, roles in ROLE_GROUPS.items() for role in roles}
    pool, rejected = [], Counter()
    for batch in pq.ParquetFile(source).iter_batches(batch_size=1024):
        for raw in batch.to_pylist():
            family = by_role.get(raw.get("seed_role"))
            if family is None:
                continue
            try:
                if raw.get("source") != "synthetic" or not re.fullmatch(
                    r"syn2?_[a-z_]+_\d+", raw["profile_id"]
                ):
                    raise ValueError("Unexpected identity")
                original = json.loads(raw["profile_json"])
                profile = sanitize(original)
                if not profile["experience"]:
                    raise ValueError("Missing employment history")
            except (ValueError, KeyError, TypeError):
                rejected["invalid schema, identity or empty employment"] += 1
                continue
            pool.append(
                {
                    "source_id": raw["profile_id"],
                    "family": family,
                    "cohort": raw["cohort"],
                    "seniority": raw["seniority"],
                    "seed_role": raw["seed_role"],
                    "selection_key": hashlib.sha256(
                        (SEED + raw["profile_id"]).encode()
                    ).hexdigest(),
                    "profile": profile,
                    "content_hash": content_key(profile),
                    "date_audit": date_audit(profile),
                    "source_reported_experience_years": original["user"].get("experience_years"),
                }
            )
    chosen = choose(pool, active + heldout, QUOTAS)
    # Assign IDs only after selection, independent of the order in the Parquet file.
    chosen.sort(key=lambda r: r["selection_key"])
    for n, row in enumerate(chosen, 101):
        row.update(candidate_id=f"P{n:03d}", display_name=f"Candidate P{n:03d}", split="active")
    report = {
        "schema_version": 1,
        "selection_seed": SEED,
        "source_sha256": SOURCE_SHA,
        "protocol": "Append 100 distinct source profiles. Reserve 20 sampling places for Python and React co-mentions; distribute 80 across broader occupational strata. Balance source titles, career cohorts and seniority within each stratum with seeded tie breaks. Do not use runtime retrieval, scoring, extraction or evaluation judgments.",
        "coverage_interpretation": "Source-field mentions describe sample coverage; they do not establish qualification or personal skill ownership.",
        "quotas": QUOTAS,
        "eligible_source_roles": ROLE_GROUPS,
        "pool_sizes": dict(Counter(r["family"] for r in pool)),
        "exclusions": dict(rejected),
        "before": coverage(active),
        "added": coverage(chosen),
        "after": coverage(active + chosen),
        "added_families": dict(Counter(r["family"] for r in chosen)),
        "added_cohorts": dict(Counter(r["cohort"] for r in chosen)),
        "added_seniority": dict(Counter(r["seniority"] for r in chosen)),
        "original_active_records_sha256": sha(root / "data/people/active.jsonl"),
        "original_manifest": json.loads((root / "data/corpus_manifest.json").read_text()),
        "protected_files": {
            p.relative_to(root).as_posix(): sha(p)
            for folder in ("data/holdout", "data/regression/data", "data/regression/config")
            for p in sorted((root / folder).rglob("*"))
            if p.is_file()
        },
        "holdout_records_sha256": sha(root / "data/people/holdout.jsonl"),
        "selected": [
            {
                k: r[k]
                for k in (
                    "candidate_id",
                    "source_id",
                    "content_hash",
                    "selection_stratum",
                    "seed_role",
                )
            }
            for r in chosen
        ],
    }
    return chosen, report


def append(root: Path, additions: list[dict], report: dict) -> None:
    active_file = root / "data/people/active.jsonl"
    if sha(active_file) != report["original_active_records_sha256"]:
        raise ValueError("Active records changed after selection")
    manifest = json.loads((root / "data/corpus_manifest.json").read_text())
    if manifest != report["original_manifest"]:
        raise ValueError("Manifest changed after selection")
    for entry in manifest["files"]:
        if sha(root / entry["path"]) != entry["sha256"]:
            raise ValueError("An existing resume changed before append")
    planned = []
    for offset, row in enumerate(additions, len(manifest["files"])):
        ext = "pdf" if offset % 10 < 5 else "docx" if offset % 10 < 8 else "txt"
        path = root / f"data/resumes/{row['candidate_id']}.{ext}"
        canonical = root / f"data/people/text/{row['candidate_id']}.txt"
        if path.exists() or canonical.exists():
            raise ValueError(f"Refusing to overwrite {row['candidate_id']}")
        planned.append((offset, row, ext, path, canonical))
    for offset, row, ext, path, canonical in planned:
        content = blocks(row)
        text = "\n".join(value for _, value in content) + "\n"
        if ext == "pdf":
            render_pdf(path, content, offset % 3)
        elif ext == "docx":
            render_docx(path, content, offset % 3, keep_certifications_together=True)
        else:
            path.write_text(text, encoding="utf-8")
        canonical.write_text(text, encoding="utf-8")
        manifest["files"].append(
            {
                "candidate_id": row["candidate_id"],
                "source_id": row["source_id"],
                "display_name": row["display_name"],
                "path": path.relative_to(root).as_posix(),
                "sha256": sha(path),
                "split": "production",
                "layout": offset % 3,
                "canonical_text_sha256": sha(canonical),
            }
        )
    with active_file.open("a", encoding="utf-8") as stream:
        for row in additions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    for entry in manifest["files"][-len(additions) :]:
        if entry["path"].endswith(".docx"):
            entry["docx_keep_certifications_together"] = True
    manifest["candidate_count"] = len(manifest["files"])
    manifest["formats"] = dict(Counter(Path(r["path"]).suffix[1:] for r in manifest["files"]))
    dump(root / "data/corpus_manifest.json", manifest)
    dump(root / "data/people/expansion.json", report)
    summary_path = root / "data/people/render_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["active"] = {k: manifest[k] for k in ("candidate_count", "formats")}
    dump(summary_path, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--select-only",
        type=Path,
        help="Write a review packet without rendering or modifying the corpus",
    )
    args = parser.parse_args()
    additions, report = select(args.source, args.root)
    if args.select_only:
        dump(args.select_only, {"additions": additions, "report": report})
    else:
        append(args.root, additions, report)
    print(
        json.dumps({k: report[k] for k in ("before", "added", "after", "added_families")}, indent=2)
    )
