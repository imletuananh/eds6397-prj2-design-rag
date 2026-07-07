from __future__ import annotations

import re


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def infer_time_filters(question: str, allowed_years: set[int]) -> dict[str, list[int]]:
    """Infer non-oracle metadata filters from the user question text."""
    lowered = question.lower()
    years = {
        int(match)
        for match in re.findall(r"\b(19\d{2}|20\d{2})\b", lowered)
        if int(match) in allowed_years
    }
    months = {number for name, number in MONTHS.items() if re.search(rf"\b{name}\b", lowered)}

    filters: dict[str, list[int]] = {}
    if years:
        filters["year"] = sorted(years)
    if months:
        filters["month"] = sorted(months)
    return filters

