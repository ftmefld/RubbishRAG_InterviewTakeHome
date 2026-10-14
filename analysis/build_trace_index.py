"""Build an exact trace index for the actual RubbishRAG trace schema.

The challenge trace records have this shape:
    {"ts": ..., "key": ..., "q": ..., "topk": ..., "hits": [...]}

There is no separate trace_id field. The provided `rubbish.py inspect-trace`
accepts the JSONL line number as an ID, so this script uses line number as the
reproducible CLI trace ID.

Usage:
    python -m analysis.build_trace_index
    python -m analysis.build_trace_index --contains "رتبه C3"
    python -m analysis.build_trace_index --contains 9004
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = ROOT / "traces" / "trace.jsonl"
OUTPUT_PATH = ROOT / "analysis" / "trace_index.md"


def compact(text: str, limit: int = 95) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contains",
        default="",
        help="Optional substring filter over the raw JSONL record.",
    )
    args = parser.parse_args()

    if not TRACE_PATH.exists():
        raise SystemExit(f"Trace file not found: {TRACE_PATH}")

    rows = []

    with TRACE_PATH.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            if args.contains and args.contains not in raw_line:
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            hits = record.get("hits") or []
            doc_ids = [
                str(hit.get("doc_id"))
                for hit in hits
                if isinstance(hit, dict) and hit.get("doc_id") is not None
            ]

            fused = [
                str(hit.get("fused"))
                for hit in hits[:5]
                if isinstance(hit, dict) and hit.get("fused") is not None
            ]

            rows.append(
                {
                    "trace_id": line_number,
                    "ts": record.get("ts", ""),
                    "query": compact(record.get("q", "")),
                    "topk": record.get("topk", ""),
                    "doc_ids": ", ".join(doc_ids[:12]),
                    "fused": ", ".join(fused),
                }
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    out = [
        "# Trace Index",
        "",
        "The raw trace schema has no separate `trace_id`. "
        "`rubbish.py inspect-trace` accepts the JSONL line number, "
        "so **CLI Trace ID = JSONL line number**.",
        "",
        f"Source: `{TRACE_PATH.relative_to(ROOT)}`",
        "",
        "| CLI Trace ID | ts | Query | topk | Observed doc IDs | First fused scores |",
        "|---:|---:|---|---:|---|---|",
    ]

    for row in rows:
        query = row["query"].replace("|", r"\|")
        out.append(
            f"| {row['trace_id']} | {row['ts']} | {query} | "
            f"{row['topk']} | {row['doc_ids']} | {row['fused']} |"
        )

    OUTPUT_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")

    print(f"Wrote {len(rows)} trace rows to {OUTPUT_PATH}")
    if rows:
        print(
            "Inspect one with: "
            f"python rubbish.py inspect-trace {rows[0]['trace_id']}"
        )


if __name__ == "__main__":
    main()
