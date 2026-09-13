"""Role aliases for request normalization and readable labels."""

from __future__ import annotations

import re
from typing import Literal

RoleName = Literal[
    "software_developer",
    "frontend_developer",
    "backend_developer",
    "full_stack_developer",
    "data_engineer",
    "machine_learning_engineer",
    "devops_engineer",
    "operations_analyst",
]

ROLE_LABELS: dict[RoleName, str] = {
    "software_developer": "Software developer",
    "frontend_developer": "Frontend developer",
    "backend_developer": "Backend developer",
    "full_stack_developer": "Full-stack developer",
    "data_engineer": "Data engineer",
    "machine_learning_engineer": "Machine learning engineer",
    "devops_engineer": "DevOps engineer",
    "operations_analyst": "Operations analyst",
}

ROLE_ALIASES: dict[RoleName, list[str]] = {
    "software_developer": [
        "software developer",
        "software engineer",
        "software dev",
        "software development",
        "application developer",
        "applications developer",
        "web developer",
        "computer programmer",
    ],
    "frontend_developer": [
        "frontend developer",
        "front-end developer",
        "front end developer",
        "frontend engineer",
        "front-end engineer",
        "front end engineer",
        "frontend software engineer",
        "front-end software engineer",
        "front end software engineer",
        "ui developer",
        "ui engineer",
        "frontend",
        "front-end",
        "front end",
    ],
    "backend_developer": [
        "backend developer",
        "back-end developer",
        "back end developer",
        "backend engineer",
        "back-end engineer",
        "back end engineer",
        "backend software engineer",
        "back-end software engineer",
        "back end software engineer",
        "server-side developer",
        "server side developer",
        "backend",
        "back-end",
        "back end",
    ],
    "full_stack_developer": [
        "full stack developer",
        "full-stack developer",
        "fullstack developer",
        "full stack engineer",
        "full-stack engineer",
        "fullstack engineer",
        "full stack software engineer",
        "full-stack software engineer",
        "full stack",
        "full-stack",
        "fullstack",
    ],
    "data_engineer": ["data engineer", "data engineering", "analytics engineer"],
    "machine_learning_engineer": [
        "machine learning engineer",
        "machine-learning engineer",
        "ml engineer",
        "machine learning engineering",
    ],
    "devops_engineer": [
        "devops engineer",
        "devops developer",
        "site reliability engineer",
        "sre",
        "devops",
    ],
    "operations_analyst": [
        "operations analyst",
        "operational analyst",
        "business operations analyst",
    ],
}


def role_mentions(text: str) -> list[tuple[int, int, RoleName]]:
    """Find phrases for planning; longer, more specific overlapping phrases win."""
    candidates = []
    for role, aliases in ROLE_ALIASES.items():
        for alias in [role, *aliases]:
            pattern = re.escape(alias).replace(r"\ ", r"\s+")
            # Job queries commonly use plural titles.
            if alias.endswith(("developer", "engineer", "analyst", "programmer")):
                pattern += "s?"
            for match in re.finditer(r"(?<!\w)" + pattern + r"(?!\w)", text, re.I):
                candidates.append((match.start(), match.end(), role))
    selected = []
    for span in sorted(
        candidates,
        key=lambda item: (-(item[1] - item[0]), item[2] == "software_developer", item[0]),
    ):
        if not any(span[0] < end and span[1] > start for start, end, _ in selected):
            selected.append(span)
    return sorted(selected)
