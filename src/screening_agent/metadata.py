"""Source spans, statistical entities, and conservative dated work evidence."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import yaml

from .contracts import Candidate, Evidence

AS_OF_DATE = date(2026, 9, 1)
EXTRACTION_VERSION = "spans-ner-intervals-v6"
MONTHS = {
    name.lower(): index
    for index, name in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
    )
}
DATE_PART = r"(?:[A-Za-z]{3,9}\s+\d{4}|\d{4}-\d{2}|(?:19|20)\d{2}|Present|Current|Now)"
DATE_RANGE = re.compile(
    rf"(?P<start>{DATE_PART})\s*(?:–|—|\s-\s|\bto\b)\s*(?P<end>{DATE_PART})", re.I
)
ROLE = re.compile(
    r"\b(?:engineer|developer|analyst|scientist|architect|consultant|administrator|designer|manager|specialist|assistant|intern)\b",
    re.I,
)
SECTION = re.compile(
    r"(?mi)^[ \t]*(?P<heading>EMPLOYMENT HISTORY|EMPLOYMENT|PROFESSIONAL EXPERIENCE|WORK EXPERIENCE|WORK HISTORY|"
    r"EDUCATION|SKILLS|TECHNOLOGIES|PROJECTS|PERSONAL PROJECTS|ACADEMIC PROJECTS|CERTIFICATIONS|"
    r"ACADEMIC BACKGROUND|PROFILE|SUMMARY|CAREER OBJECTIVE|CAREER TARGET|OBJECTIVE)[ \t]*:?[ \t]*$"
)
WORK_SECTIONS = frozenset(
    {
        "employment history",
        "employment",
        "professional experience",
        "work experience",
        "work history",
    }
)
PROFILE_SECTIONS = frozenset(
    {"profile", "summary", "career objective", "career target", "objective"}
)
NEGATED = re.compile(
    r"(?:\bno|\bwithout|\bnot|\bnever|\black(?:ing|s)?|\bnot yet)\s+(?:\w+\s+){0,3}$", re.I
)
COURSE_CUE = re.compile(
    r"\b(?:courses?|coursework|tutorials?|workshops?|bootcamps?|classroom|studied|training\s+(?:course|program|session))\b|\b(?:completed|attended|took|enrolled\s+in)\b[^.;\n]*\btraining\b",
    re.I,
)
NUMBER_WORDS = {
    name: index
    for index, name in enumerate(
        [
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
        ],
        1,
    )
}
DURATION_CLAIM = re.compile(
    r"(?P<n>\d+(?:\.\d+)?|"
    + "|".join(NUMBER_WORDS)
    + r")\s*[- ]\s*(?P<unit>years?|months?|weeks?|days?)\b",
    re.I,
)
OWN_WORK = re.compile(
    r"\b(?:built|developed|implemented|designed|wrote|used|using|applied|maintained|migrated|deployed|"
    r"tested|automated|optimized|improved|delivered|created|integrated|engineered|operated|configured|"
    r"administered|debugged|refactored|programmed|trained|fine.tuned|evaluated|managed|led|owned|reduced|tuned|investigated)\b",
    re.I,
)
OTHER_OR_ASPIRATIONAL = re.compile(
    r"\b(?:hoped?|hope|want(?:ed)?|intend(?:ed)?|aspir(?:e|ed|ing)|would|could|"
    r"learn(?:ing)?|observed|shadowed|partnered|collaborated|liaised)\b|"
    r"\b(?:other|another|external|partner|their)\s+(?:\w+\s+){0,2}team\b|"
    r"\b(?:colleagues?|teammates?|coworkers?)\b|\bplan(?:s|ned|ning)?\s+to\b",
    re.I,
)


def owned_work(clause: str) -> bool:
    """A bounded assertion rule; ambiguous participation remains uncredited."""
    return bool(OWN_WORK.search(clause)) and not (
        COURSE_CUE.search(clause) or OTHER_OR_ASPIRATIONAL.search(clause)
    )


class SkillExtractor:
    def __init__(self, root: Path):
        self.spec = yaml.safe_load((Path(root) / "config/skills.yaml").read_text())
        self.patterns: dict[str, re.Pattern] = {}
        for key, spec in self.spec.items():
            aliases = sorted(
                {key, spec["display_name"], *spec.get("aliases", [])}, key=len, reverse=True
            )
            self.patterns[key] = re.compile(
                "(?:"
                + "|".join(
                    (r"(?<![\w.\-–—])" if len(value) <= 3 else r"(?<!\w)")
                    + re.escape(value)
                    + r"(?!\w)"
                    for value in aliases
                )
                + ")",
                re.I,
            )

    def spans(self, text: str) -> list[dict]:
        spans = []
        for key, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                if key == "react" and re.match(r"[\s-]+native\b", text[match.end() :], re.I):
                    # React Native is a separate mobile framework. A later
                    # independent React/React.js mention can still qualify.
                    continue
                # The nearest clause controls a negation; the preceding sentence does not.
                prefix = re.split(r"[.;\n]", text[max(0, match.start() - 55) : match.start()])[-1]
                suffix = text[match.end() :]
                denied_after = re.match(
                    r"\s+(?:(?:was|is|were|has|have|had|been|used|experience)\s+){0,3}(?:not|never)\b",
                    suffix,
                    re.I,
                )
                if not NEGATED.search(prefix) and not denied_after:
                    spans.append(
                        {
                            "text": match.group(),
                            "label": "SKILL",
                            "canonical": key,
                            "start": match.start(),
                            "end": match.end(),
                            "method": "taxonomy",
                        }
                    )
        return sorted(spans, key=lambda value: (value["start"], value["end"]))

    def match(self, text: str) -> list[str]:
        return sorted({item["canonical"] for item in self.spans(text)})

    def display(self, skill: str) -> str:
        return self.spec.get(skill, {}).get("display_name", skill)


def _month(value: str, snapshot_date: date = AS_OF_DATE) -> int:
    value = value.strip()
    if value.lower() in {"present", "current", "now"}:
        return snapshot_date.year * 12 + snapshot_date.month
    if re.fullmatch(r"\d{4}-\d{2}", value):
        year, month = map(int, value.split("-"))
        if not 1 <= month <= 12:
            raise ValueError("invalid month")
        return year * 12 + month
    if value.isdigit():
        return int(value) * 12 + 1
    name, year = value.split()
    return int(year) * 12 + MONTHS[name[:3].lower()]


def union_months(intervals: list[tuple[int, int]]) -> int:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def _skill_role_credits(
    text: str, skills: SkillExtractor, role_months: int
) -> tuple[dict[str, int], dict[str, int], list[str]]:
    """A role's dates cannot override an explicit shorter or course-only claim."""
    mentions: dict[str, list[int | None]] = {}
    warnings = []
    # The dated header describes a position, not an assertion of skill usage.
    body = text.partition("\n")[2]
    clauses = re.split(r"\n+|(?<=[.!?;])\s+|,?\s+(?:then|but|and later)\s+", body, flags=re.I)
    for clause in clauses:
        if not owned_work(clause):
            continue
        spans = skills.spans(clause)
        claims = list(DURATION_CLAIM.finditer(clause))
        for index, span in enumerate(spans):
            right = spans[index + 1]["start"] if index + 1 < len(spans) else len(clause)
            left = spans[index - 1]["end"] if index else 0
            linked = [claim for claim in claims if span["end"] <= claim.start() < right]
            if not linked:
                linked = [
                    claim
                    for claim in claims
                    if left <= claim.start()
                    and claim.end() <= span["start"]
                    and re.fullmatch(
                        r"\s*(?:(?:of|with|in|using)\s+)?(?:experience\s+)?(?:(?:with|in|using)\s+)?",
                        clause[claim.end() : span["start"]],
                        re.I,
                    )
                ]
            cap = None
            if linked:
                values = []
                for claim in linked:
                    value = claim["n"].lower()
                    quantity = float(NUMBER_WORDS[value]) if value in NUMBER_WORDS else float(value)
                    unit = claim["unit"].lower()
                    # Weeks/days are rounded down so a partial month never passes a whole-month threshold.
                    values.append(
                        int(
                            quantity
                            * (
                                12
                                if unit.startswith("year")
                                else 1
                                if unit.startswith("month")
                                else 7 / 31
                                if unit.startswith("week")
                                else 1 / 31
                            )
                        )
                    )
                cap = min(values)
            if cap is None and re.search(
                r"\bthroughout\s+(?:(?:the|this|my|entire)\s+){0,2}role\b", clause, re.I
            ):
                cap = role_months
            mentions.setdefault(span["canonical"], []).append(cap)
    credits, explicit_credits = {}, {}
    for skill, caps in mentions.items():
        explicit = [value for value in caps if value is not None]
        credits[skill] = min(role_months, min(explicit)) if explicit else role_months
        if explicit:
            explicit_credits[skill] = min(role_months, min(explicit))
        if explicit and min(explicit) < role_months:
            warnings.append(
                f"{skills.display(skill)} experience is limited to an explicit {min(explicit)}-month claim within the dated role; the full role duration was not credited."
            )
    return credits, explicit_credits, warnings


def _credited_union(records: list[tuple[int, int, int]]) -> int:
    """Merge overlaps without inventing dates for a shorter skill-duration claim.

    Within overlapping roles, the lower bound is the greater of the fully
    supported intervals and the longest partial claim. Disjoint roles add.
    """
    groups: list[list[tuple[int, int, int]]] = []
    end = -1
    for record in sorted(records):
        if groups and record[0] < end:
            groups[-1].append(record)
            end = max(end, record[1])
        else:
            groups.append([record])
            end = record[1]
    return sum(
        max(
            union_months(
                [(start, stop) for start, stop, credit in group if credit >= stop - start]
            ),
            max((credit for _, _, credit in group), default=0),
        )
        for group in groups
    )


def work_blocks(text: str, *, snapshot_date: date = AS_OF_DATE) -> tuple[list[dict], list[str]]:
    """Accept dated job headers, including nontechnical jobs in explicit work sections.

    Education and project sections never supply employment tenure. A profile
    permits only an explicit dated job header, never a target-role description.
    Arithmetic is bounded by the snapshot; source excerpts retain original dates.
    """
    ranges = list(DATE_RANGE.finditer(text))
    sections = list(SECTION.finditer(text))
    blocks, warnings = [], []
    for index, match in enumerate(ranges):
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        line_end = len(text) if line_end < 0 else line_end
        previous_start = text.rfind("\n", 0, max(0, line_start - 1)) + 1
        header = text[previous_start:line_end]
        dated_title = text[line_start:line_end].split("|", 1)[0]
        if re.search(r"\bbootcamp\b", dated_title, re.I) and re.search(
            r"\b(?:student|participant)\b", dated_title, re.I
        ):
            warnings.append(
                f"Classroom participation is not employment tenure: {dated_title.strip()}"
            )
            continue
        preceding_sections = [section for section in sections if section.end() <= match.start()]
        section = preceding_sections[-1]["heading"].casefold() if preceding_sections else None
        if section is not None and section not in WORK_SECTIONS:
            job_header = text[line_start:line_end]
            job_title = job_header.split("|", 1)[0]
            if not (
                section in PROFILE_SECTIONS
                and "|" in job_header
                and ROLE.search(job_title)
                and not re.search(
                    r"\b(?:aspiring|seeking|desired|target|objective|wants?)\b", job_title, re.I
                )
            ):
                continue
        title = DATE_RANGE.sub("", header).strip(" \n\t|:-–—")
        # Generic titles are accepted only within a declared employment section.
        # Historical fixtures with unconventional headings still require a role word.
        if not ROLE.search(header) and not (
            section in WORK_SECTIONS
            and re.search(r"[A-Za-z]{2}", title)
            and not SECTION.fullmatch(title)
        ):
            continue
        end = ranges[index + 1].start() if index + 1 < len(ranges) else len(text)
        # Stop before the next role's entire header or an explicit later section.
        if index + 1 < len(ranges):
            end = text.rfind("\n", 0, end) + 1
        later_heading = SECTION.search(text, line_end, end)
        if later_heading:
            end = later_heading.start()
        try:
            start_month, end_month = (
                _month(match["start"], snapshot_date),
                _month(match["end"], snapshot_date),
            )
        except (ValueError, KeyError):
            warnings.append(f"Unrecognized date range: {match.group()}")
            continue
        ceiling = snapshot_date.year * 12 + snapshot_date.month
        if start_month >= end_month or start_month >= ceiling:
            warnings.append(f"Invalid or future employment range: {match.group()}")
            continue
        if end_month > ceiling:
            end_month = ceiling
            warnings.append(
                f"Employment range ends after snapshot {snapshot_date.isoformat()}; credited only through the snapshot: {match.group()}"
            )
        blocks.append(
            {
                "start": line_start,
                "end": end,
                "interval": (start_month, end_month),
                "text": text[line_start:end],
            }
        )
    return blocks, warnings


class MetadataExtractor:
    def __init__(self, root: Path, *, backend: str = "spacy", snapshot_date: date = AS_OF_DATE):
        if backend not in {"spacy", "rules"}:
            raise ValueError("NER backend must be spacy or rules")
        self.backend = backend
        if not isinstance(snapshot_date, date):
            raise ValueError("snapshot_date must be a date")
        self.snapshot_date = snapshot_date
        self.skills = SkillExtractor(root)
        self.nlp = None
        if backend == "spacy":
            import spacy

            try:
                self.nlp = spacy.load(
                    "en_core_web_sm", disable=["parser", "lemmatizer", "tagger", "attribute_ruler"]
                )
            except OSError as exc:
                raise RuntimeError(
                    "The statistical NER model en_core_web_sm is missing. Run uv sync."
                ) from exc

    def extract(
        self,
        candidate_id: str,
        text: str,
        source_path: str,
        sha256: str,
        *,
        source_id: str | None = None,
        display_name: str | None = None,
    ) -> Candidate:
        skill_spans = self.skills.spans(text)
        entities = list(skill_spans)
        warnings = []
        if self.nlp:
            doc = self.nlp(text)
            entities.extend(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "method": "spacy:en_core_web_sm",
                }
                for ent in doc.ents
                if ent.label_ in {"PERSON", "ORG", "DATE"}
            )
            # PDF line breaks can make a name and following title look like one
            # entity. Re-run the unchanged header lines with exact source offsets.
            for line in list(re.finditer(r"(?m)^.+$", text[:250]))[:6]:
                if ROLE.search(line.group()) or "@" in line.group():
                    continue
                line_doc = self.nlp(line.group())
                for ent in line_doc.ents:
                    if ent.label_ == "PERSON" and ent.text.strip() == line.group().strip():
                        entities.append(
                            {
                                "text": ent.text,
                                "label": "PERSON",
                                "start": line.start() + ent.start_char,
                                "end": line.start() + ent.end_char,
                                "method": "spacy:en_core_web_sm:header",
                            }
                        )
        else:
            warnings.append("Rule-only extraction mode; statistical NER is disabled.")
        header_lines = [line.strip() for line in text[:250].splitlines() if line.strip()]
        first_line = next(iter(header_lines), "Unknown candidate")
        persons = [item for item in entities if item["label"] == "PERSON" and item["start"] < 180]
        # A name may follow 'Curriculum Vitae' or a role heading. Statistical
        # entities can choose that later header line; partial names stay uncertain.
        plausible = [
            line
            for line in header_lines
            if not ROLE.search(line)
            and "@" not in line
            and line.lower() not in {"resume", "résumé", "curriculum vitae", "profile", "contact"}
            and 1 <= len(line.split()) <= 5
            and not re.search(r"\d|[|:]", line)
        ]
        name = next(
            (person["text"] for person in persons if person["text"] in plausible),
            next(iter(plausible), first_line),
        )
        if display_name is not None:
            name = display_name
        elif not any(person["text"] == name for person in persons):
            warnings.append(
                "Display name uses the document header; statistical NER did not verify the complete name."
            )
        blocks, date_warnings = work_blocks(text, snapshot_date=self.snapshot_date)
        warnings.extend(date_warnings)
        if not blocks:
            warnings.append("Dated employment was not demonstrated; total experience is unknown.")
        durations: dict[str, list[tuple[int, int, int]]] = {}
        claimed_durations: dict[str, list[tuple[int, int, int]]] = {}
        evidence = []
        for block in blocks:
            start_month, end_month = block["interval"]
            credits, explicit_credits, credit_warnings = _skill_role_credits(
                block["text"], self.skills, end_month - start_month
            )
            warnings.extend(credit_warnings)
            for skill, credit in credits.items():
                if credit > 0:
                    durations.setdefault(skill, []).append((start_month, end_month, credit))
            for skill, credit in explicit_credits.items():
                if credit > 0:
                    claimed_durations.setdefault(skill, []).append((start_month, end_month, credit))
            evidence.append(
                Evidence(
                    candidate_id=candidate_id,
                    source_id=source_id,
                    source_path=source_path,
                    start=block["start"],
                    end=block["end"],
                    quote=block["text"],
                    document_sha256=sha256,
                    kind="dated_employment",
                )
            )
            for skill in credits:
                evidence.append(evidence[-1].model_copy(update={"kind": f"skill_work:{skill}"}))
            for skill in explicit_credits:
                evidence.append(evidence[-1].model_copy(update={"kind": f"skill_duration:{skill}"}))
        for item in skill_spans:
            start = text.rfind("\n", 0, item["start"]) + 1
            end = text.find("\n", item["end"])
            end = len(text) if end < 0 else end
            if not any(existing.start == start and existing.end == end for existing in evidence):
                evidence.append(
                    Evidence(
                        candidate_id=candidate_id,
                        source_id=source_id,
                        source_path=source_path,
                        start=start,
                        end=end,
                        quote=text[start:end],
                        document_sha256=sha256,
                        kind="skill_mention",
                    )
                )
            if owned_work(text[start:end]):
                evidence.append(
                    Evidence(
                        candidate_id=candidate_id,
                        source_id=source_id,
                        source_path=source_path,
                        start=start,
                        end=end,
                        quote=text[start:end],
                        document_sha256=sha256,
                        kind=f"skill_application:{item['canonical']}",
                    )
                )
        return Candidate(
            candidate_id=candidate_id,
            source_id=source_id,
            name=name,
            source_path=source_path,
            skills=sorted({item["canonical"] for item in skill_spans}),
            experience_years=round(union_months([block["interval"] for block in blocks]) / 12, 2)
            if blocks
            else None,
            skill_years={
                skill: round(_credited_union(values) / 12, 2) for skill, values in durations.items()
            },
            experience_months=union_months([block["interval"] for block in blocks])
            if blocks
            else None,
            claimed_skill_months={
                skill: _credited_union(values) for skill, values in claimed_durations.items()
            },
            evidence=evidence,
            warnings=warnings,
            entities=entities,
        )


def redact_identity(text: str, candidate: Candidate) -> str:
    def replace_name(value: str, replacement: str) -> None:
        nonlocal text
        if value:
            text = re.sub(r"(?<!\w)" + re.escape(value) + r"(?!\w)", lambda _: replacement, text)

    replace_name(candidate.name, "[candidate]")
    if candidate.source_id:
        text = text.replace(candidate.source_id, "[source]")
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[email]", text)
    text = re.sub(r"(?<!\w)(?:\+\d[\d ()-]{8,}\d)(?!\w)", "[phone]", text)
    # PERSON annotations are not an identity authority: the statistical model
    # labels Redux, Jest and company names as people in this corpus. Mask known
    # candidate identity/contact fields; keep uncertain NER annotations separate.
    # This is candidate-name masking, not a guarantee of full de-identification.
    return text
