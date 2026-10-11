"""Domain slot extraction for Persian credit-scoring queries.

Extracts:
  - BANK: known Iranian banks / financial platforms
  - RANK: credit rank A1..E3
  - PERSONA: individual vs legal/corporate entity
  - DEBT: debt category
  - AMOUNT / NUMS: numeric values
  - TIME: presence of temporal expressions
  - NEGATED: presence of explicit negation

All matching is performed on normalize_fa() output.
"""

import re
from typing import Any, Dict, Iterable, Optional

from .normalize_fa import normalize_fa


# Canonical names exposed to callers.
BANKS = [
    "توسعه صادرات",
    "صادرات",
    "تجارت",
    "ملی",
    "ملت",
    "سپه",
    "پاسارگاد",
    "دیجی پی",
    "اسنپ پی",
    "تارا",
    "دیجیکالا",
]


# Surface forms -> canonical bank/entity name.
BANK_ALIASES = {
    "توسعه صادرات": (
        "بانک توسعه صادرات",
        "توسعه صادرات",
    ),
    "صادرات": (
        "بانک صادرات",
        "صادرات",
    ),
    "تجارت": (
        "بانک تجارت",
        "تجارت",
    ),
    "ملی": (
        "بانک ملی",
        "ملی",
    ),
    "ملت": (
        "بانک ملت",
        "ملت",
    ),
    "سپه": (
        "بانک سپه",
        "سپه",
    ),
    "پاسارگاد": (
        "بانک پاسارگاد",
        "پاسارگاد",
    ),
    "دیجی پی": (
        "دیجی پی",
        "دیجی‌پی",
    ),
    "اسنپ پی": (
        "اسنپ پی",
        "اسنپ‌پی",
    ),
    "تارا": (
        "تارا",
    ),
    "دیجیکالا": (
        "دیجیکالا",
        "دیجی کالا",
        "دیجی‌کالا",
    ),
}


RANKS = [
    f"{letter}{number}"
    for letter in "ABCDE"
    for number in "123"
]


DEBT_PATTERNS = [
    (
        "چک",
        (
            "چک",
            "چک صیادی",
            "صیادی",
            "برگشتی",
            "رفع سوءاثر",
            "رفع سواثر",
        ),
    ),
    (
        "مالیاتی",
        (
            "بدهی مالیاتی",
            "جریمه مالیاتی",
            "امور مالیاتی",
            "مالیات",
        ),
    ),
    (
        "تسهیلاتی",
        (
            "تسهیلات",
            "وام",
            "قسط",
            "اقساط",
            "معوق",
        ),
    ),
    (
        "محکومیت",
        (
            "محکومیت",
            "ورشکستگی",
            "دادگاه",
            "قضایی",
        ),
    ),
]


INDIVIDUAL_PATTERNS = (
    "شخص حقیقی",
    "حقیقی",
    "فرد",
    "خودم",
)


LEGAL_PATTERNS = (
    "شخص حقوقی",
    "حقوقی",
    "شرکت",
    "کسب و کار",
    "سازمان",
)


TIME_PATTERNS = (
    "ساعت",
    "روز",
    "هفته",
    "ماه",
    "سال",
    "مدت",
    "بازه",
    "ماهانه",
    "سالانه",
)


NEGATION_TERMS = (
    "نیست",
    "فاقد",
    "بدون",
    "ندارم",
    "ندارد",
    "ندارند",
    "معاف",
    "عدم",
)


def _contains_phrase(text: str, phrase: str) -> bool:
    """Return True only when phrase occurs on word boundaries."""
    canonical = normalize_fa(phrase)

    pattern = rf"(?<!\w){re.escape(canonical)}(?!\w)"

    return bool(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
    )


def _contains_any(
    text: str,
    phrases: Iterable[str],
) -> bool:
    return any(
        _contains_phrase(text, phrase)
        for phrase in phrases
    )


def _extract_bank(norm: str) -> Optional[str]:
    """Return the longest matching bank/entity alias."""

    candidates = []

    for canonical, aliases in BANK_ALIASES.items():
        for alias in aliases:
            normalized_alias = normalize_fa(alias)

            candidates.append(
                (
                    len(normalized_alias),
                    canonical,
                    normalized_alias,
                )
            )

    # Longest phrase first prevents:
    #
    #   "توسعه صادرات"
    #
    # from being classified as merely "صادرات".
    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    for _, canonical, alias in candidates:
        if _contains_phrase(norm, alias):
            return canonical

    return None


def _extract_rank(norm: str) -> Optional[str]:
    """Extract A1..E3 after digit normalization."""
    match = re.search(
        r"(?<![A-Za-z0-9])([A-Ea-e])\s*([1-3])(?![A-Za-z0-9])",
        norm,
    )

    if not match:
        return None

    return (
        match.group(1).upper()
        + match.group(2)
    )


def _extract_persona(norm: str) -> Optional[str]:
    is_individual = _contains_any(
        norm,
        INDIVIDUAL_PATTERNS,
    )

    is_legal = _contains_any(
        norm,
        LEGAL_PATTERNS,
    )

    if is_individual and is_legal:
        return "both"

    if is_individual:
        return "حقیقی"

    if is_legal:
        return "حقوقی"

    return None


def _extract_debt(norm: str) -> Optional[str]:
    for category, keywords in DEBT_PATTERNS:
        if _contains_any(norm, keywords):
            return category

    return None


def _extract_numbers(norm: str) -> list[str]:
    """Extract standalone numeric values without rank digits."""

    numbers = re.findall(
        r"(?<!\w)\d+(?:[,.]\d+)*(?!\w)",
        norm,
    )

    # Normalize thousands separators.
    return [
        number.replace(",", "")
        for number in numbers
    ]


def _contains_negation(norm: str) -> bool:
    if _contains_any(
        norm,
        NEGATION_TERMS,
    ):
        return True

    # Handles both:
    #   نمی خواهم
    #   نمیخواهم
    #
    # without treating "نمی" found inside another word as a negation.
    return bool(
        re.search(
            r"(?<!\w)نمی(?=\s|\w|$)",
            norm,
        )
    )


def extract_slots(
    query: str,
) -> Dict[str, Any]:
    """Extract credit-scoring domain slots from a Persian query."""

    norm = normalize_fa(
        query or ""
    )

    numbers = _extract_numbers(norm)

    slots: Dict[str, Any] = {
        "BANK": _extract_bank(norm),
        "RANK": _extract_rank(norm),
        "PERSONA": _extract_persona(norm),
        "DEBT": _extract_debt(norm),

        "AMOUNT": (
            numbers[0]
            if numbers
            else None
        ),

        "TIME": _contains_any(
            norm,
            TIME_PATTERNS,
        ),

        "NUMS": numbers,

        "NEGATED": _contains_negation(
            norm
        ),
    }

    return slots


def extract_slots_broken(
    query: str,
) -> dict:
    """Starter stub retained for baseline compatibility."""
    return {
        "BANK": None,
        "RANK": None,
        "PERSONA": None,
        "DEBT": None,
        "AMOUNT": None,
        "TIME": None,
    }