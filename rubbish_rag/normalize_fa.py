"""Persian text normalization for Credit-Scoring QA.

The normalizer produces a deterministic search-oriented canonical form.

It handles:
  1. Persian / Arabic digits -> ASCII numerals
  2. Arabic letter variants -> canonical Persian letters
  3. ZWNJ / half-space -> ordinary space for retrieval consistency
  4. Common colloquial Persian forms -> canonical written forms
  5. Whitespace normalization

Important:
    This function is intended for retrieval / matching, not display.
    Therefore ZWNJ is flattened to a normal space so variants such as
    "می‌شود" and "می شود" match identically.

The transformation is idempotent:

    normalize_fa(normalize_fa(x)) == normalize_fa(x)
"""

import re
from typing import Optional


FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EN_DIGITS = "0123456789"

_DIGIT_MAP = {
    ord(src): dst
    for src, dst in zip(
        FA_DIGITS + AR_DIGITS,
        EN_DIGITS * 2,
    )
}


# Arabic-script variants commonly encountered in Persian text.
_CHAR_MAP = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "ـ": "",
    }
)


# All outputs are already in canonical search form.
# In particular, outputs do NOT introduce a new ZWNJ.
COLLOQUIAL_MAP = {
    "میشه": "می شود",
    "نمیشه": "نمی شود",
    "چیه": "چیست",
    "چقد": "چقدر",

    "رتبم": "رتبه من",
    "رتبه م": "رتبه من",

    "امتیازم": "امتیاز من",
    "امتیاز م": "امتیاز من",

    "سابقه م": "سابقه من",
    "سابقم": "سابقه من",

    "شرکتم": "شرکت من",
    "شرکت م": "شرکت من",

    "چکم": "چک من",
    "چک م": "چک من",

    "وامم": "وام من",
    "وام م": "وام من",

    "قسطامو": "اقساط من را",
    "قسطام": "اقساط من",

    "گزارشمو": "گزارش من را",
    "گزارشم": "گزارش من",

    "مالیاتمو": "مالیات من را",
    "مالیاتم": "مالیات من",
}


def _replace_phrase(text: str, source: str, target: str) -> str:
    """Replace a complete word/phrase without matching inside another word."""
    pattern = rf"(?<!\w){re.escape(source)}(?!\w)"
    return re.sub(pattern, target, text)


def normalize_fa(text: Optional[str]) -> str:
    """Normalize Persian text into a deterministic retrieval form."""
    if not text:
        return ""

    # 1. Normalize Persian and Arabic-Indic digits.
    normalized = str(text).translate(_DIGIT_MAP)

    # 2. Normalize Arabic character variants.
    normalized = normalized.translate(_CHAR_MAP)

    # 3. Flatten ZWNJ to an ordinary space for search equivalence.
    normalized = normalized.replace("\u200c", " ")

    # 4. Collapse whitespace before phrase matching.
    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    # 5. Apply longest colloquial forms first.
    # This prevents e.g. "قسطام" from partially handling "قسطامو".
    for source in sorted(
        COLLOQUIAL_MAP,
        key=len,
        reverse=True,
    ):
        normalized = _replace_phrase(
            normalized,
            source,
            COLLOQUIAL_MAP[source],
        )

    # 6. Re-collapse whitespace introduced by replacements.
    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    return normalized