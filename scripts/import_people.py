"""Import a fixed, sanitized sample of the externally generated People corpus.

Selection never imports the application's extractor, scorer or relevance labels.
The source file remains outside the repository; only sanitized selections ship.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
DATASET = "akzaidan/People"
SOURCE_SHA = "dd2d3b70642260b046f33034f3c19b608be5b72ad582005408d5256526490b42"
SNAPSHOT = "2026-09-01"
SEED = "people-screening-v1-20260911"
ROLES = {
    "frontend": ["frontend engineer", "react developer"],
    "backend": ["backend engineer", "python developer", "full stack developer"],
    "data": ["data engineer", "big data engineer", "streaming data engineer"],
    "devops": ["devops engineer"],
    "ml": ["machine learning engineer"],
    "adjacent": ["operations analyst"],
}
QUOTAS = {
    "active": {"frontend": 22, "backend": 22, "data": 16, "devops": 16, "ml": 14, "adjacent": 10},
    "holdout": {"frontend": 10, "backend": 10, "data": 8, "devops": 8, "ml": 7, "adjacent": 7},
}
# An allowlist, rather than deleting a few known keys from a changing source schema.
FIELDS = {
    # Bios repeat sampled immigration/status details in this release. Exclude
    # the entire field uniformly rather than selectively rewriting sentences.
    "user": ("skills",),
    "experience": ("company", "title", "start_date", "end_date", "description", "is_internship"),
    "education": ("school", "degree", "field_of_study", "start_date", "end_date", "description"),
    "projects": ("name", "description", "start_date", "end_date"),
    "certifications": ("name", "description", "date"),
}
FORBIDDEN = {
    "age",
    "ethnicity",
    "legal_status",
    "sponsorship_needed",
    "desired_salary",
    "gender",
    "country",
    "city",
    "state",
    "languages",
    "career_interests",
    "experience_years",
    "work_location_preferences",
    "bio",
}


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean(value):
    if isinstance(value, str):
        # Keep wording, punctuation and qualifications. Normalize only whitespace.
        return re.sub(r"[ \t]+", " ", value.replace("\r\n", "\n")).strip()
    if isinstance(value, list):
        return [clean(item) for item in value if item not in (None, "")]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise ValueError("Unexpected nested value in an allowlisted scalar field")


def sanitize(profile: dict) -> dict:
    if not isinstance(profile.get("user"), dict):
        raise ValueError("Profile user must be an object")
    result = {}
    for section, fields in FIELDS.items():
        if section == "user":
            result[section] = {key: clean(profile[section].get(key)) for key in fields}
        else:
            values = profile.get(section) or []
            if not isinstance(values, list) or any(not isinstance(row, dict) for row in values):
                raise ValueError(f"Invalid {section} collection")
            result[section] = [{key: clean(row.get(key)) for key in fields} for row in values]
    if not isinstance(result["user"].get("skills"), list):
        result["user"]["skills"] = []
    return result


def month(value: str | None, *, current: bool = False) -> int | None:
    if value is None:
        return 2026 * 12 + 9 if current else None
    match = re.fullmatch(r"(0[1-9]|1[0-2])/((?:19|20)\d{2})", str(value))
    if not match:
        return None
    return int(match[2]) * 12 + int(match[1])


def date_text(value: str | None, *, current: bool = False) -> str:
    if value is None:
        return "Present" if current else "date unavailable"
    parsed = month(value)
    if parsed is None:
        return f"unparsed date {value}"
    year, zero_month = divmod(parsed - 1, 12)
    return f"{year:04d}-{zero_month + 1:02d}"


def date_audit(profile: dict) -> dict:
    """Independent arithmetic oracle: enumerate occupied months, do not merge spans."""
    occupied = set()
    warnings, intervals = [], []
    ceiling = month(None, current=True)
    for index, entry in enumerate(profile["experience"]):
        start = month(entry.get("start_date"))
        end = month(entry.get("end_date"), current=True)
        if start is None or end is None or start >= end:
            warnings.append({"entry": index, "reason": "missing, invalid, or reversed dates"})
            continue
        stop = min(end, ceiling)
        if end > ceiling:
            warnings.append({"entry": index, "reason": "end clipped at evaluation snapshot"})
        if start >= stop:
            warnings.append(
                {"entry": index, "reason": "employment starts after evaluation snapshot"}
            )
            continue
        occupied.update(range(start, stop))
        intervals.append({"entry": index, "start_month": start, "end_month": stop})
    return {
        "snapshot_date": SNAPSHOT,
        "intervals": intervals,
        "union_months": len(occupied),
        "experience_years": round(len(occupied) / 12, 2) if intervals else None,
        "warnings": warnings,
        "method": "Enumerated unique calendar months; end month excluded; concurrent jobs count once.",
    }


def content_key(profile: dict) -> str:
    # Identity-independent duplicate guard includes complete work and education text.
    value = json.dumps(profile, sort_keys=True, ensure_ascii=False).casefold()
    return hashlib.sha256(re.sub(r"\W+", " ", value).encode()).hexdigest()


def select(source: Path, root: Path) -> dict:
    import pyarrow.parquet as pq

    if (root / "data/people/expansion.json").exists():
        raise ValueError(
            "Refusing to replace an expanded corpus with the original 100-profile selection; use a fresh staging copy for historical reproduction."
        )
    actual = sha(source)
    if actual != SOURCE_SHA:
        raise ValueError(
            "This importer is pinned to a different source checksum; review a new release before selecting it."
        )
    by_role = {role: family for family, roles in ROLES.items() for role in roles}
    pools = defaultdict(list)
    inspected, rejected = 0, Counter()
    parquet = pq.ParquetFile(source)
    for batch in parquet.iter_batches(batch_size=512):
        for row in batch.to_pylist():
            family = by_role.get(row.get("seed_role"))
            if not family:
                continue
            inspected += 1
            try:
                if row.get("source") != "synthetic" or not re.fullmatch(
                    r"syn2?_[a-z_]+_\d+", row["profile_id"]
                ):
                    raise ValueError("Unexpected source identity")
                original = json.loads(row["profile_json"])
                profile = sanitize(original)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                rejected["invalid schema or identity"] += 1
                continue
            if not profile["experience"]:
                rejected["no employment entries"] += 1
                continue
            key = hashlib.sha256((SEED + row["profile_id"]).encode()).hexdigest()
            pools[family].append(
                {
                    "source_id": row["profile_id"],
                    "family": family,
                    "cohort": row["cohort"],
                    "seniority": row["seniority"],
                    "seed_role": row["seed_role"],
                    "selection_key": key,
                    "profile": profile,
                    "content_hash": content_key(profile),
                    "date_audit": date_audit(profile),
                    "source_reported_experience_years": original["user"].get("experience_years"),
                }
            )
    used_ids, used_content = set(), set()
    results = {}
    # Holdout first: its allocation cannot be adjusted after examining app rankings.
    for split in ("holdout", "active"):
        selected = []
        for family, count in QUOTAS[split].items():
            buckets = defaultdict(list)
            for row in sorted(pools[family], key=lambda item: item["selection_key"]):
                # Round-robin career cohorts, with seeded order and varied seniority inside each.
                buckets[row["cohort"]].append(row)
            cohort_order = sorted(
                buckets, key=lambda c: hashlib.sha256((SEED + family + c).encode()).hexdigest()
            )
            family_rows = []
            while len(family_rows) < count:
                progressed = False
                for cohort in cohort_order:
                    while buckets[cohort]:
                        row = buckets[cohort].pop(0)
                        if row["source_id"] in used_ids or row["content_hash"] in used_content:
                            continue
                        used_ids.add(row["source_id"])
                        used_content.add(row["content_hash"])
                        family_rows.append(row)
                        progressed = True
                        break
                    if len(family_rows) == count:
                        break
                if not progressed:
                    raise ValueError(f"Insufficient distinct profiles for {split}/{family}")
            selected.extend(family_rows)
        selected.sort(key=lambda item: item["selection_key"])
        prefix = "P" if split == "active" else "E"
        for n, row in enumerate(selected, 1):
            row["candidate_id"] = f"{prefix}{n:03d}"
            row["display_name"] = f"Candidate {prefix}{n:03d}"
            row["split"] = split
        results[split] = selected
        path = root / f"data/people/{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Administrative selection strata stay outside rendered/indexed profiles.
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
            encoding="utf-8",
        )
    provenance = {
        "schema_version": 1,
        "dataset_id": DATASET,
        "source_url": f"https://huggingface.co/datasets/{DATASET}",
        "source_file": "profiles.parquet",
        "source_sha256": actual,
        "source_bytes": source.stat().st_size,
        "source_row_count": parquet.metadata.num_rows,
        "license": "MIT (declared by publisher)",
        "snapshot_date": SNAPSHOT,
        "snapshot_basis": "Project evaluation cutoff; publisher does not document a collection reference date. Present is evaluated as of this cutoff, not assumed verified current employment.",
        "selection_seed": SEED,
        "selection_protocol": "Source checksum pinned; exact occupational strata; seeded round-robin career cohorts; holdout selected first; no scorer/extractor/labels used.",
        "eligible_source_roles": ROLES,
        "quotas": QUOTAS,
        "inspected_relevant_rows": inspected,
        "schema_exclusions": dict(rejected),
        "pool_sizes": {family: len(rows) for family, rows in pools.items()},
        "removed_fields": sorted(FORBIDDEN),
        "retained_profile_fields": FIELDS,
        "counts": {
            split: dict(Counter(row["family"] for row in rows)) for split, rows in results.items()
        },
        "cohorts": {
            split: dict(Counter(row["cohort"] for row in rows)) for split, rows in results.items()
        },
        "seniority": {
            split: dict(Counter(row["seniority"] for row in rows))
            for split, rows in results.items()
        },
    }
    dump(root / "data/people/provenance.json", provenance)
    return provenance


def blocks(row: dict) -> list[tuple[str, str]]:
    """Faithful text transformation. No generated achievements or inferred skills."""
    p = row["profile"]
    # Source IDs encode generator cohorts (including visa status); retain them
    # in the administrative manifest, never in searchable text or model input.
    result = [("title", row["display_name"]), ("note", "Synthetic candidate profile")]
    result.append(("section", "EMPLOYMENT HISTORY"))
    for entry in p["experience"]:
        role = entry.get("title") or "Title not supplied"
        company = entry.get("company") or "Employer not supplied"
        dates = f"{date_text(entry.get('start_date'))} to {date_text(entry.get('end_date'), current=True)}"
        result.append(("job", f"{role} | {company} | {dates}"))
        if entry.get("description"):
            result.append(("body", entry["description"]))
    if p["user"]["skills"]:
        result.extend([("section", "SKILLS"), ("body", "; ".join(p["user"]["skills"]))])
    if p["education"]:
        result.append(("section", "EDUCATION"))
        for entry in p["education"]:
            values = [
                entry.get(key) for key in ("degree", "field_of_study", "school") if entry.get(key)
            ]
            result.append(("body", " | ".join(values)))
            if entry.get("start_date") or entry.get("end_date"):
                result.append(
                    (
                        "body",
                        f"{date_text(entry.get('start_date'))} to {date_text(entry.get('end_date'))}",
                    )
                )
            if entry.get("description"):
                result.append(("body", entry["description"]))
    for section, label in [("projects", "PROJECTS"), ("certifications", "CERTIFICATIONS")]:
        if p[section]:
            result.append(("section", label))
        for entry in p[section]:
            if entry.get("name"):
                result.append(("subheading", entry["name"]))
            if entry.get("description"):
                result.append(("body", entry["description"]))
            if section == "projects" and (entry.get("start_date") or entry.get("end_date")):
                result.append(
                    (
                        "body",
                        f"{date_text(entry.get('start_date'))} to {date_text(entry.get('end_date'))}",
                    )
                )
            elif entry.get("date"):
                result.append(("body", date_text(entry["date"])))
    return [(kind, text) for kind, text in result if text]


def render_pdf(path: Path, content: list[tuple[str, str]], layout: int) -> None:
    import reportlab
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate

    if "PeopleVera" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(
            TTFont("PeopleVera", str(Path(reportlab.__file__).parent / "fonts/Vera.ttf"))
        )
    base = ParagraphStyle(
        "body",
        fontName="PeopleVera",
        fontSize=10,
        leading=13,
        spaceAfter=4,
        textColor=colors.HexColor("#283a36"),
    )
    styles = {
        "body": base,
        "title": ParagraphStyle("title", parent=base, fontSize=21, leading=26, spaceAfter=8),
        "note": ParagraphStyle(
            "note", parent=base, fontSize=8, leading=11, textColor=colors.HexColor("#60746c")
        ),
        "section": ParagraphStyle(
            "section",
            parent=base,
            fontSize=10,
            leading=14,
            spaceBefore=9,
            spaceAfter=5,
            keepWithNext=True,
            textColor=colors.HexColor("#315e49"),
        ),
        "job": ParagraphStyle("job", parent=base, spaceBefore=5, keepWithNext=True),
        "subheading": ParagraphStyle("subheading", parent=base, keepWithNext=True),
    }
    flow, certificates, in_certifications = [], [], False
    for kind, text in content:
        if kind == "section":
            in_certifications = text == "CERTIFICATIONS"
        paragraph = Paragraph(escape(text).replace("\n", "<br/>"), styles[kind])
        if in_certifications:
            certificates.append(paragraph)
        else:
            flow.append(paragraph)
    if certificates:
        # The imported profiles have short certification sections. Keep the
        # heading and complete entries on one page, including their dates.
        flow.append(KeepTogether(certificates))
    doc = SimpleDocTemplate(
        str(path),
        pagesize=(595.28, 841.89),
        leftMargin=48 + layout * 4,
        rightMargin=48 + layout * 4,
        topMargin=40,
        bottomMargin=42,
        title=content[0][1],
        author="Resume Matching Agent",
        invariant=1,
    )

    def footer(canvas, document):
        canvas.setFont("PeopleVera", 8)
        canvas.setFillColor(colors.HexColor("#60746c"))
        canvas.drawString(48, 24, "Synthetic profile | People dataset")
        canvas.drawRightString(547, 24, str(document.page))

    doc.build(flow, onFirstPage=footer, onLaterPages=footer)


def render_docx(
    path: Path,
    content: list[tuple[str, str]],
    layout: int,
    *,
    keep_certifications_together: bool = False,
) -> None:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor

    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    section.top_margin = section.bottom_margin = Inches(0.65)
    section.left_margin = section.right_margin = Inches(0.7 + layout * 0.05)
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = "Arial", Pt(10)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.1
    title = doc.styles["Title"]
    title.font.name, title.font.size, title.font.color.rgb = "Arial", Pt(21), RGBColor(0, 0, 0)
    for border in title.element.xpath(".//w:pBdr"):
        border.getparent().remove(border)
    in_certifications = False
    for number, (kind, text) in enumerate(content):
        if kind == "section":
            in_certifications = text == "CERTIFICATIONS"
        style = "Title" if kind == "title" else "Heading 1" if kind == "section" else "Normal"
        paragraph = doc.add_paragraph(text, style=style)
        if kind == "section":
            paragraph.paragraph_format.space_before = Pt(10)
            paragraph.paragraph_format.space_after = Pt(5)
            for run in paragraph.runs:
                run.font.size, run.font.color.rgb = Pt(10), RGBColor.from_string("315E49")
        if kind in {"job", "subheading", "section"}:
            paragraph.paragraph_format.keep_with_next = True
        if keep_certifications_together and in_certifications and number < len(content) - 1:
            paragraph.paragraph_format.keep_with_next = True
        if kind == "note":
            for run in paragraph.runs:
                run.font.size = Pt(8)
        if kind == "job":
            for run in paragraph.runs:
                run.bold = True
    doc.core_properties.title = content[0][1]
    doc.core_properties.author = "Resume Matching Agent"
    doc.core_properties.created = doc.core_properties.modified = datetime(2026, 9, 11)
    doc.save(path)
    source = path.read_bytes()
    with (
        zipfile.ZipFile(io.BytesIO(source)) as old,
        zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as new,
    ):
        for name in sorted(old.namelist()):
            info = zipfile.ZipInfo(name, (2026, 9, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            new.writestr(info, old.read(name))


def preserve_regression(root: Path) -> None:
    destination = root / "data/regression"
    if destination.exists():
        return
    original = json.loads((root / "data/corpus_manifest.json").read_text())
    if original.get("schema_version") != 1 or original.get("candidate_count") != 100:
        raise ValueError("Original regression corpus was not identified; nothing was replaced.")
    (destination / "data").mkdir(parents=True)
    for name in ["resumes", "ground_truth", "evaluation", "corpus_manifest.json"]:
        source, target = root / "data" / name, destination / "data" / name
        if source.is_dir():
            shutil.copytree(source, target)
        elif source.exists():
            shutil.copy2(source, target)
    shutil.copytree(root / "config", destination / "config")
    for entry in original["files"]:
        if sha(destination / entry["path"]) != entry["sha256"]:
            raise ValueError("Regression preservation hash mismatch")


def render(root: Path) -> dict:
    preserve_regression(root)
    summary = {}
    for split in ("active", "holdout"):
        source = root / f"data/people/{split}.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        parent = root / "data" if split == "active" else root / "data/holdout"
        manifest_path = parent / "corpus_manifest.json"
        prior_files = (
            {
                entry["candidate_id"]: entry
                for entry in json.loads(manifest_path.read_text())["files"]
            }
            if manifest_path.exists()
            else {}
        )
        staging = parent / "resumes.staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        files = []
        for n, row in enumerate(rows):
            extension = "pdf" if n % 10 < 5 else "docx" if n % 10 < 8 else "txt"
            path = staging / f"{row['candidate_id']}.{extension}"
            content = blocks(row)
            layout = n % 3
            if extension == "pdf":
                render_pdf(path, content, layout)
            elif extension == "docx":
                render_docx(
                    path,
                    content,
                    layout,
                    keep_certifications_together=prior_files.get(row["candidate_id"], {}).get(
                        "docx_keep_certifications_together", False
                    ),
                )
            else:
                path.write_text("\n".join(text for _, text in content) + "\n", encoding="utf-8")
            canonical = root / f"data/people/text/{row['candidate_id']}.txt"
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_text("\n".join(text for _, text in content) + "\n", encoding="utf-8")
            relative = (
                f"data/resumes/{path.name}"
                if split == "active"
                else f"data/holdout/resumes/{path.name}"
            )
            files.append(
                {
                    "candidate_id": row["candidate_id"],
                    "source_id": row["source_id"],
                    "display_name": row["display_name"],
                    "path": relative,
                    "sha256": sha(path),
                    "split": "production" if split == "active" else "holdout",
                    "layout": layout,
                    "canonical_text_sha256": sha(canonical),
                }
            )
            if extension == "docx" and prior_files.get(row["candidate_id"], {}).get(
                "docx_keep_certifications_together"
            ):
                files[-1]["docx_keep_certifications_together"] = True
        manifest = {
            "schema_version": 2,
            "dataset_id": DATASET,
            "dataset_split": split,
            "snapshot_date": SNAPSHOT,
            "source_sha256": SOURCE_SHA,
            "candidate_count": len(files),
            "formats": dict(Counter(Path(entry["path"]).suffix[1:] for entry in files)),
            "files": files,
        }
        destination = parent / "resumes"
        old = parent / "resumes.previous"
        if old.exists():
            shutil.rmtree(old)
        if destination.exists():
            destination.rename(old)
        staging.rename(destination)
        dump(parent / "corpus_manifest.json", manifest)
        if old.exists():
            shutil.rmtree(old)
        summary[split] = {key: manifest[key] for key in ["candidate_count", "formats"]}
    # Historical labels must not masquerade as labels for the new default corpus.
    for name in ("ground_truth",):
        obsolete = root / "data" / name
        if obsolete.exists():
            shutil.rmtree(obsolete)
    dump(root / "data/people/render_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if not args.render_only:
        if args.source is None:
            parser.error("--source is required for selection")
        result = select(args.source, args.root)
        print(
            json.dumps({"counts": result["counts"], "pool_sizes": result["pool_sizes"]}, indent=2)
        )
    if not args.select_only:
        print(json.dumps(render(args.root), indent=2))
