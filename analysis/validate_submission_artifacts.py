"""Validate final submission artifacts without making remote calls.

Usage:
    python -m analysis.validate_submission_artifacts
"""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PROOF = ROOT / "docs" / "PROOF.md"
TRACE = ROOT / "traces" / "trace.jsonl"

REQUIRED_PLOTS = [
    ROOT / "plots" / "length_vs_bm25.png",
    ROOT / "plots" / "hybrid_ablation.png",
    ROOT / "plots" / "hybrid_rerank_ablation.png",
    ROOT / "plots" / "chunking_ablation.png",
    ROOT / "plots" / "judge_ablation.png",
]


def main() -> None:
    problems = []

    if not PROOF.exists():
        problems.append("Missing docs/PROOF.md")

    if not TRACE.exists():
        problems.append("Missing traces/trace.jsonl")

    for path in REQUIRED_PLOTS:
        if not path.exists():
            problems.append(f"Missing {path.relative_to(ROOT)}")

    if PROOF.exists():
        text = PROOF.read_text(encoding="utf-8")

        forbidden_patterns = [
            r"TRACE_ID_TODO_\d+",
            r"\{\{TRACE_[A-Z0-9_]+\}\}",
            r"\bTBD\b",
        ]

        for pattern in forbidden_patterns:
            matches = sorted(set(re.findall(pattern, text)))
            if matches:
                problems.append(
                    f"Unresolved placeholders for {pattern}: {matches}"
                )

        if "quota exceeded" not in text:
            problems.append(
                "PROOF.md should document the final remote quota limitation."
            )

        if "DEV" not in text:
            problems.append(
                "PROOF.md should explicitly label development-set evidence."
            )

        # Expect at least 15 distinct numeric trace IDs in backticks in
        # the Trace Evidence Index area.
        if "## Trace Evidence Index" not in text:
            problems.append("Missing Trace Evidence Index section.")

    if problems:
        print("Submission artifact validation: FAIL")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)

    print("Submission artifact validation: PASS")
    print("- docs/PROOF.md")
    print("- traces/trace.jsonl")
    for path in REQUIRED_PLOTS:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
