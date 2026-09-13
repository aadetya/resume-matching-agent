"""Create 100 fictional resumes with separately stored evaluation labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from docx import Document
from docx.shared import Pt
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260910
FAMILIES = {
    "FE": {
        "title": "Frontend Engineer",
        "must": ["react", "javascript"],
        "preferred": ["typescript", "testing", "git"],
        "display": ["React", "JavaScript", "TypeScript", "automated testing", "Git"],
        "work": [
            "Built accessible account screens and reusable React components for a customer portal.",
            "Reduced repeated browser work by splitting large JavaScript bundles and caching static assets.",
            "Used TypeScript contracts and automated testing to catch failures before each release.",
            "Reviewed Git changes and documented keyboard navigation behavior with designers.",
        ],
        "adjacent": "Designed wireframes and static HTML pages; production application development was outside this role.",
    },
    "BE": {
        "title": "Backend Engineer",
        "must": ["python", "sql"],
        "preferred": ["fastapi", "postgresql", "docker"],
        "display": ["Python", "SQL", "FastAPI", "PostgreSQL", "Docker"],
        "work": [
            "Implemented Python services that validate incoming orders and recover from duplicate requests.",
            "Investigated slow SQL queries and added indexes after reading execution plans.",
            "Published FastAPI endpoints with explicit error responses and contract tests.",
            "Used PostgreSQL transactions and Docker development environments for repeatable releases.",
        ],
        "adjacent": "Prepared manual spreadsheet exports and reconciled order totals with the operations team.",
    },
    "DE": {
        "title": "Data Engineer",
        "must": ["sql", "spark"],
        "preferred": ["python", "airflow", "kafka"],
        "display": ["SQL", "Apache Spark", "Python", "Airflow", "Kafka"],
        "work": [
            "Built SQL transformations with explicit grain, null checks, and late-arriving data handling.",
            "Used Apache Spark to compact partitioned event data and avoid repeated full-table scans.",
            "Scheduled Python validation steps in Airflow with retries limited to idempotent tasks.",
            "Tracked Kafka consumer lag and replayed failed event batches from known offsets.",
        ],
        "adjacent": "Maintained weekly report spreadsheets and checked imported totals with business analysts.",
    },
    "DO": {
        "title": "DevOps Engineer",
        "must": ["aws", "kubernetes"],
        "preferred": ["terraform", "linux", "monitoring"],
        "display": ["AWS", "Kubernetes", "Terraform", "Linux", "observability"],
        "work": [
            "Operated AWS services with budget checks, least-privilege access, and tested recovery procedures.",
            "Diagnosed Kubernetes scheduling failures and configured readiness checks before deployment.",
            "Reviewed Terraform plans and Linux service limits during infrastructure changes.",
            "Built observability dashboards around error budgets and recurring incident symptoms.",
        ],
        "adjacent": "Coordinated release meetings and maintained a manual inventory of team laptops.",
    },
    "ML": {
        "title": "Machine Learning Engineer",
        "must": ["python", "pytorch"],
        "preferred": ["docker", "statistics", "monitoring"],
        "display": ["Python", "PyTorch", "Docker", "statistics", "observability"],
        "work": [
            "Built Python training pipelines with split checks and reproducible feature transformations.",
            "Trained PyTorch models and inspected errors by class before changing model capacity.",
            "Packaged inference with Docker and compared statistics across held-out data slices.",
            "Added observability for input shifts and checked failed predictions with domain reviewers.",
        ],
        "adjacent": "Labeled research examples and maintained annotation guidelines with a small research team.",
    },
}
FIRST = [
    "John",
    "Jane",
    "Maya",
    "Arjun",
    "Sofia",
    "Rohan",
    "Elena",
    "Kabir",
    "Priya",
    "Nikhil",
    "Nora",
    "Dev",
    "Lina",
    "Omar",
    "Asha",
    "Ishan",
    "Zoya",
    "Neel",
    "Sara",
    "Vikram",
]
LAST = {"FE": "Morgan", "BE": "Shah", "DE": "Patel", "DO": "Reed", "ML": "Sen"}
EMPLOYERS = [
    "Northstar Fiction Labs",
    "Willow Sample Systems",
    "Mosaic Example Works",
    "Cedar Fiction Studio",
    "Atlas Sample Services",
]
HEADINGS = [
    "EXPERIENCE",
    "CAREER NOTES",
    "EMPLOYMENT HISTORY",
    "WHERE I WORKED",
    "SELECTED DELIVERY",
    "PROFESSIONAL EXPERIENCE",
    "WORK RECORD",
    "RECENT ENGAGEMENTS",
    "CAREER HISTORY",
    "",
]


def _dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _profile(family: str, number: int) -> dict:
    spec = FAMILIES[family]
    cid = f"CAND_{family}_{number:03d}"
    years = 8 - (number - 1) % 7
    # Fifteen profiles have direct production evidence, including three juniors.
    direct = number <= 15
    if number in {16, 17, 18, 19}:
        years = 5
    if number == 20:
        years = 4
    preferred_count = 3 if number <= 5 else 2 if number <= 10 else 1
    skills = (
        list(spec["must"]) + spec["preferred"][:preferred_count] if direct or number == 20 else []
    )
    dated_skills = skills[:]
    blocks: list[tuple[str, str]] = []
    name = FIRST[number - 1] + " " + LAST[family]
    contact = f"{cid.lower()}@example.test"
    role_title = spec["title"]
    intro = f"{role_title} working on maintainable systems, explicit checks, and documented operational decisions."
    paragraphs = list(spec["work"][:2])
    if preferred_count:
        # Preferred tools only enter the resume when the truth labels include them.
        preferred = ", ".join(spec["display"][2 : 2 + preferred_count])
        paragraphs.append(
            f"Used {preferred} in delivery work, with peer review and documented failure cases."
        )
    paragraphs.append(
        f"Owned a {['billing', 'support', 'inventory', 'subscription', 'analytics'][number % 5]} workflow; checked outputs against recorded examples before release."
    )
    if direct:
        start, end = 2026 - years, 2026
        style = number % 4
        dates = (
            f"Jan {start} - Jan {end}"
            if style == 0
            else f"January {start} to January {end}"
            if style == 1
            else f"{start}-01 to {end}-01"
            if style == 2
            else f"{start} – {end}"
        )
        blocks.append((f"{role_title} | {EMPLOYERS[number % 5]} | {dates}", "\n".join(paragraphs)))
        if number % 5 == 0:
            # Concurrent engagement must not double-count total or skill experience.
            blocks.append(
                ("Consultant | Cedar Fiction Studio | Jan 2024 - Jan 2025", paragraphs[0])
            )
    elif number == 16:
        skills = list(spec["must"]) + spec["preferred"][:1]
        blocks.append(
            ("Operations Analyst | Willow Sample Systems | Jan 2021 - Jan 2026", spec["adjacent"])
        )
        intro = "Skills listed below have been studied independently. The dated role describes operations work."
    elif number == 17:
        skills = list(spec["must"])
        blocks.append(
            ("Research Assistant | Mosaic Example Works | Jan 2021 - Jan 2026", spec["adjacent"])
        )
        intro = f"Completed an introductory course covering {' and '.join(spec['display'][:2])}; no production ownership is claimed."
    elif number == 18:
        blocks.append(
            ("Operations Analyst | Atlas Sample Services | Jan 2021 - Jan 2026", spec["adjacent"])
        )
        intro = f"No experience with {spec['display'][0]}. No experience with {spec['display'][1]}. Seeking a supported career change."
    elif number == 19:
        skills = [spec["must"][1]]
        dated_skills = skills[:]
        blocks.append(
            (
                f"{role_title} | Northstar Fiction Labs | Jan 2021 - Jan 2026",
                f"Used {spec['display'][1]} for a small internal workflow. {spec['adjacent']}",
            )
        )
        intro = "A focused resume with one directly supported technical skill."
    else:
        # A gap between two two-year engagements is four years, not six.
        blocks = [
            (f"{role_title} | Mosaic Example Works | Jan 2019 - Jan 2021", "\n".join(paragraphs)),
            (f"{role_title} | Cedar Fiction Studio | Jan 2023 - Jan 2025", "\n".join(paragraphs)),
        ]
        skills = list(spec["must"]) + spec["preferred"][:preferred_count]
        dated_skills = skills[:]
    display_skills = [
        display
        for canonical, display in zip(
            spec["must"] + spec["preferred"], spec["display"], strict=True
        )
        if canonical in skills
    ]
    lines = [name, contact, "", "PROFILE", intro, "", HEADINGS[(number - 1) % 10]]
    for header, body in blocks:
        lines.extend([header, body, ""])
    lines += [
        "TECHNOLOGIES",
        ", ".join(display_skills) or "Technical experience is not claimed.",
        "",
        "EDUCATION",
        "Bachelor of Science | Fictional Institute of Technology",
    ]
    if family == "ML" and (direct or number in {19, 20}):
        # The role itself explicitly names this broader competency.
        skills.append("machine_learning")
        dated_skills.append("machine_learning")
    return {
        "candidate_id": cid,
        "name": name,
        "family": family,
        "number": number,
        "lines": lines,
        "skills": sorted(skills),
        "experience_years": years,
        "skill_years": {skill: float(years) for skill in dated_skills},
        "direct": direct or number == 20,
        "layout": (number - 1) % 10,
    }


def _pdf(path: Path, lines: list[str], layout: int) -> None:
    font = Path(rl_config.TTFSearchPath[0]) / "Vera.ttf"
    if not font.exists():
        import reportlab

        font = Path(reportlab.__file__).parent / "fonts/Vera.ttf"
    if "ResumeVera" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("ResumeVera", str(font)))
    style = ParagraphStyle(
        "body",
        fontName="ResumeVera",
        fontSize=9,
        leading=13,
        spaceAfter=4,
        textColor=colors.HexColor("#243447"),
    )
    heading = ParagraphStyle("heading", parent=style, fontSize=12, leading=16, spaceAfter=8)
    flow = []
    for index, line in enumerate(lines):
        if not line:
            flow.append(Spacer(1, 5))
            continue
        paragraph = Paragraph(escape(line).replace("\n", "<br/>"), heading if index == 0 else style)
        if layout in {1, 4, 7} and " | " in line:
            table = Table([[paragraph]], colWidths=[6.65 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef4f7")),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            flow.append(table)
        else:
            flow.append(paragraph)
    doc = SimpleDocTemplate(
        str(path),
        leftMargin=48 + layout % 3 * 4,
        rightMargin=48,
        topMargin=40,
        bottomMargin=40,
        title="Fictional screening benchmark resume",
        author="Resume Matching Agent",
        invariant=1,
    )
    doc.build(flow)


def _docx(path: Path, lines: list[str], layout: int) -> None:
    doc = Document()
    doc.styles["Normal"].font.size = Pt(10)
    doc.core_properties.created = datetime(2026, 9, 10)
    doc.core_properties.modified = datetime(2026, 9, 10)
    for index, line in enumerate(lines):
        if layout % 2 and " | " in line:
            table = doc.add_table(rows=1, cols=1)
            table.cell(0, 0).text = line
        else:
            doc.add_paragraph(line, style="Title" if index == 0 else "Normal")
    doc.save(path)
    # Archive timestamps otherwise change each generation despite identical content.
    import io
    import zipfile

    source = path.read_bytes()
    with (
        zipfile.ZipFile(io.BytesIO(source)) as old,
        zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as new,
    ):
        for name in sorted(old.namelist()):
            info = zipfile.ZipInfo(name, (2026, 9, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            new.writestr(info, old.read(name))


def generate(root: Path) -> dict:
    data = root / "data"
    manifest_path = data / "corpus_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("schema_version") == 2:
        raise ValueError(
            "Refusing to overwrite an imported corpus. Use a separate regression root."
        )
    (data / "resumes").mkdir(parents=True, exist_ok=True)
    profiles = [_profile(family, number) for family in FAMILIES for number in range(1, 21)]
    files = []
    for index, profile in enumerate(profiles):
        extension = "pdf" if index % 10 < 5 else "docx" if index % 10 < 8 else "txt"
        path = data / "resumes" / f"{profile['candidate_id']}.{extension}"
        if extension == "pdf":
            _pdf(path, profile["lines"], profile["layout"])
        elif extension == "docx":
            _docx(path, profile["lines"], profile["layout"])
        else:
            path.write_text("\n".join(profile["lines"]), encoding="utf-8")
        files.append(
            {
                "candidate_id": profile["candidate_id"],
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "layout": profile["layout"],
            }
        )
    truths = [
        {key: value for key, value in profile.items() if key not in {"lines"}}
        for profile in profiles
    ]
    _dump(data / "ground_truth/candidates.json", truths)
    jobs = build_queries(profiles)
    _dump(data / "ground_truth/queries.json", jobs)
    manifest = {
        "schema_version": 1,
        "seed": SEED,
        "candidate_count": len(profiles),
        "query_count": len(jobs),
        "formats": {"pdf": 50, "docx": 30, "txt": 20},
        "files": files,
        "description": "All people, employers, institutions, dates and achievements are fictional. Ground truth is excluded from runtime indexing.",
    }
    _dump(data / "corpus_manifest.json", manifest)
    return manifest


def build_queries(profiles: list[dict]) -> list[dict]:
    """Generate source-faithful query variants; all are development material."""
    from screening_agent.roles import role_mentions

    jobs = []
    for family, spec in FAMILIES.items():
        for variant in ("development", "paraphrase", "skill_duration"):
            requirements = {
                "title": spec["title"],
                "must_have": [[skill] for skill in spec["must"]],
                "nice_to_have": [name.casefold() for name in spec["display"][2:]],
                "role": role_mentions(spec["title"])[0][2],
                "minimum_years": 0 if variant == "skill_duration" else 3,
                "skill_years": {spec["must"][0]: 3} if variant == "skill_duration" else {},
            }
            query = (
                f"Find {spec['title']} candidates with {spec['display'][0]} and {spec['display'][1]}, and at least 3 years of overall employment. "
                f"Prefer {', '.join(spec['display'][2:])}."
            )
            if variant == "paraphrase":
                query = (
                    f"Please shortlist people who have done {spec['title']} work with both {spec['display'][0]} and {spec['display'][1]}. "
                    f"They need at least three years of overall employment. Experience with {', '.join(spec['display'][2:])} would be useful but is optional."
                )
            elif variant == "skill_duration":
                query = (
                    f"Find {spec['title']} candidates with {spec['display'][0]} and {spec['display'][1]}. "
                    f"Require 3 years in employment involving {spec['display'][0]}; overall employment alone is insufficient. "
                    f"Prefer {', '.join(spec['display'][2:])}."
                )
            requirements["source_text"] = query
            grades, mandatory = {}, {}
            for profile in profiles:
                passes = (
                    all(skill in profile["skills"] for skill in spec["must"])
                    and profile["experience_years"] >= 3
                )
                if variant == "skill_duration":
                    passes = passes and profile["skill_years"].get(spec["must"][0], 0) >= 3
                mandatory[profile["candidate_id"]] = passes
                grade = 0
                if profile["family"] == family:
                    if not passes:
                        grade = 1 if profile["skills"] else 0
                    elif profile["direct"]:
                        grade = (
                            3
                            if all(skill in profile["skills"] for skill in spec["preferred"])
                            else 2
                        )
                    else:
                        grade = 1  # Skill-list/course-only profiles remain weaker than production evidence.
                grades[profile["candidate_id"]] = grade
            jobs.append(
                {
                    "job_id": f"{family}_{variant}",
                    "split": "development",
                    "scenario": variant,
                    "label_version": 2,
                    "label_scope": "Generator-authored rubric: direct qualifying work with all stated preferences receives grade 3. No independent human review.",
                    "query": query,
                    "requirements": requirements,
                    "grades": grades,
                    "mandatory": mandatory,
                }
            )
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data/regression")
    args = parser.parse_args()
    result = generate(args.root)
    print(
        f"Generated {result['candidate_count']} fictional resumes and {result['query_count']} labeled queries."
    )
