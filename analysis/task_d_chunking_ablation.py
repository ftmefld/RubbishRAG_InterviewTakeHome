import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import rubbish_rag as R
import rubbish_rag.pipeline  # registers @R.chunker


CORPUS_PATH = Path("corpus/test.csv")
OUTPUT_PATH = Path("analysis/task_d_chunking_results.json")

BASELINE_CHUNK_SIZE = 300

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!؟!?؛;])\s+|\n+"
)

# Includes sentence/clause delimiters.
_BOUNDARY_END_RE = re.compile(
    r"[.!؟!?؛;،,:]\s*$"
)

_METADATA_PREFIX_RE = re.compile(
    r"^\[[^\]]+\]\s*"
)

_NUMERIC_PHRASE_RE = re.compile(
    r"(?<!\w)"
    r"[0-9۰-۹٠-٩][0-9۰-۹٠-٩,./]*"
    r"(?:\s*(?:تا|الی|-)\s*"
    r"[0-9۰-۹٠-٩][0-9۰-۹٠-٩,./]*)?"
    r"(?:\s+(?:میلیون|میلیارد))?"
    r"(?:\s+(?:ریال|تومان|درصد|٪|امتیاز|روز|ماه|سال|ساعت))?"
    r"(?!\w)"
)


def load_docs():
    with CORPUS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    docs = []

    for index, row in enumerate(rows):
        item = dict(row)

        # Defensive fallback in case the CSV does not expose doc_id.
        if not item.get("doc_id"):
            item["doc_id"] = str(index)

        docs.append(item)

    return docs


def source_text(doc):
    return (
        doc.get("Answer")
        or doc.get("BriefAnswer")
        or ""
    ).strip()


# ---------------------------------------------------------
# Baseline chunker: exact fixed-300 behavior
# ---------------------------------------------------------

def baseline_chunk(docs):
    chunks = []

    for doc in docs:
        text = source_text(doc)

        for i in range(
            0,
            len(text),
            BASELINE_CHUNK_SIZE,
        ):
            part = text[
                i:i + BASELINE_CHUNK_SIZE
            ]

            if not part.strip():
                continue

            chunks.append(
                {
                    "doc_id": doc["doc_id"],
                    "category": doc.get(
                        "Category",
                        "",
                    ),
                    "text": part,
                    "chunk_id": (
                        f"{doc['doc_id']}:"
                        f"{i // BASELINE_CHUNK_SIZE}"
                    ),
                }
            )

    return chunks


def chunk_body(text):
    """
    Remove retrieval-visible [Category | Keyword] metadata
    before measuring preservation of the original Answer.
    """
    return _METADATA_PREFIX_RE.sub(
        "",
        text,
        count=1,
    ).strip()


def group_chunks(chunks):
    grouped = defaultdict(list)

    for chunk in chunks:
        grouped[str(chunk["doc_id"])].append(
            chunk
        )

    return grouped


# ---------------------------------------------------------
# Metric 1: hard boundary / mid-phrase cuts
# ---------------------------------------------------------

def boundary_stats(chunks):
    grouped = group_chunks(chunks)

    internal_boundaries = 0
    natural_boundaries = 0

    for doc_chunks in grouped.values():
        if len(doc_chunks) <= 1:
            continue

        # Every chunk except the last creates an internal boundary.
        for chunk in doc_chunks[:-1]:
            internal_boundaries += 1

            body = chunk_body(
                chunk["text"]
            )

            if _BOUNDARY_END_RE.search(body):
                natural_boundaries += 1

    bad_boundaries = (
        internal_boundaries
        - natural_boundaries
    )

    rate = (
        bad_boundaries / internal_boundaries
        if internal_boundaries
        else 0.0
    )

    return {
        "internal_boundaries": internal_boundaries,
        "natural_boundaries": natural_boundaries,
        "bad_boundaries": bad_boundaries,
        "bad_boundary_rate": rate,
    }


# ---------------------------------------------------------
# Metric 2: intact sentence preservation
# ---------------------------------------------------------

def sentence_preservation(docs, chunks):
    grouped = group_chunks(chunks)

    eligible = 0
    preserved = 0

    for doc in docs:
        text = source_text(doc)

        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_SPLIT_RE.split(text)
            if sentence.strip()
        ]

        bodies = [
            chunk_body(chunk["text"])
            for chunk in grouped.get(
                str(doc["doc_id"]),
                [],
            )
        ]

        for sentence in sentences:
            # Focus on meaningful sentences that could reasonably
            # fit inside a bounded chunk.
            if not (
                20 <= len(sentence) <= 250
            ):
                continue

            eligible += 1

            if any(
                sentence in body
                for body in bodies
            ):
                preserved += 1

    rate = (
        preserved / eligible
        if eligible
        else 0.0
    )

    return {
        "eligible_sentences": eligible,
        "preserved_sentences": preserved,
        "preservation_rate": rate,
    }


# ---------------------------------------------------------
# Metric 3: numeric phrase + unit preservation
# ---------------------------------------------------------

def numeric_phrase_preservation(
    docs,
    chunks,
):
    grouped = group_chunks(chunks)

    total = 0
    preserved = 0

    broken_examples = []

    for doc in docs:
        text = source_text(doc)

        phrases = [
            match.group(0)
            for match in _NUMERIC_PHRASE_RE.finditer(
                text
            )
        ]

        if not phrases:
            continue

        bodies = [
            chunk_body(chunk["text"])
            for chunk in grouped.get(
                str(doc["doc_id"]),
                [],
            )
        ]

        for phrase in phrases:
            # Ignore trivial bare single digits.
            if (
                len(phrase.strip()) <= 1
                and " " not in phrase
            ):
                continue

            total += 1

            intact = any(
                phrase in body
                for body in bodies
            )

            if intact:
                preserved += 1

            elif len(broken_examples) < 10:
                broken_examples.append(
                    {
                        "doc_id": doc["doc_id"],
                        "phrase": phrase,
                    }
                )

    rate = (
        preserved / total
        if total
        else 0.0
    )

    return {
        "numeric_phrases": total,
        "preserved": preserved,
        "preservation_rate": rate,
        "broken_examples": broken_examples,
    }


# ---------------------------------------------------------
# Metric 4: metadata visibility
# ---------------------------------------------------------

def metadata_stats(chunks):
    total = len(chunks)

    prefixed = sum(
        1
        for chunk in chunks
        if _METADATA_PREFIX_RE.match(
            chunk["text"]
        )
    )

    category_metadata = sum(
        1
        for chunk in chunks
        if chunk.get("category")
    )

    keyword_metadata = sum(
        1
        for chunk in chunks
        if chunk.get("keyword")
    )

    return {
        "total_chunks": total,
        "retrieval_visible_prefixes": prefixed,
        "prefix_coverage": (
            prefixed / total
            if total
            else 0.0
        ),
        "category_metadata_chunks": (
            category_metadata
        ),
        "keyword_metadata_chunks": (
            keyword_metadata
        ),
    }


# ---------------------------------------------------------
# General length statistics
# ---------------------------------------------------------

def length_stats(chunks):
    lengths = [
        len(chunk["text"])
        for chunk in chunks
    ]

    if not lengths:
        return {}

    sorted_lengths = sorted(lengths)

    p95_index = min(
        len(sorted_lengths) - 1,
        int(
            0.95
            * (len(sorted_lengths) - 1)
        ),
    )

    return {
        "chunks": len(lengths),
        "mean_length": mean(lengths),
        "median_length": median(lengths),
        "p95_length": (
            sorted_lengths[p95_index]
        ),
        "max_length": max(lengths),
        "min_length": min(lengths),
    }


def evaluate(
    name,
    docs,
    chunks,
):
    return {
        "name": name,
        "length": length_stats(chunks),
        "boundaries": boundary_stats(chunks),
        "sentences": sentence_preservation(
            docs,
            chunks,
        ),
        "numeric_phrases": (
            numeric_phrase_preservation(
                docs,
                chunks,
            )
        ),
        "metadata": metadata_stats(
            chunks
        ),
    }


def pct(value):
    return f"{100 * value:.1f}%"


def main():
    docs = load_docs()

    baseline_chunks = baseline_chunk(
        docs
    )

    registered_chunker = R.get(
        "chunker"
    )

    improved_chunks = registered_chunker(
        docs
    )

    baseline = evaluate(
        "fixed_300",
        docs,
        baseline_chunks,
    )

    improved = evaluate(
        "boundary_metadata",
        docs,
        improved_chunks,
    )

    print(
        "\nTask D — Chunking Ablation"
    )
    print("=" * 78)

    print(
        f"Documents: {len(docs)}"
    )

    print("\nChunk counts / lengths")
    print("-" * 78)

    for result in (
        baseline,
        improved,
    ):
        stats = result["length"]

        print(
            f"{result['name']:<20} "
            f"chunks={stats['chunks']:<5} "
            f"mean={stats['mean_length']:.1f} "
            f"median={stats['median_length']:.1f} "
            f"p95={stats['p95_length']} "
            f"max={stats['max_length']}"
        )

    print("\nBad internal boundary rate")
    print("-" * 78)

    print(
        "Fixed 300        :",
        pct(
            baseline[
                "boundaries"
            ][
                "bad_boundary_rate"
            ]
        ),
    )

    print(
        "Improved chunker :",
        pct(
            improved[
                "boundaries"
            ][
                "bad_boundary_rate"
            ]
        ),
    )

    print("\nIntact sentence preservation")
    print("-" * 78)

    print(
        "Fixed 300        :",
        pct(
            baseline[
                "sentences"
            ][
                "preservation_rate"
            ]
        ),
    )

    print(
        "Improved chunker :",
        pct(
            improved[
                "sentences"
            ][
                "preservation_rate"
            ]
        ),
    )

    print("\nNumeric phrase/unit preservation")
    print("-" * 78)

    print(
        "Fixed 300        :",
        pct(
            baseline[
                "numeric_phrases"
            ][
                "preservation_rate"
            ]
        ),
    )

    print(
        "Improved chunker :",
        pct(
            improved[
                "numeric_phrases"
            ][
                "preservation_rate"
            ]
        ),
    )

    print("\nRetrieval-visible metadata coverage")
    print("-" * 78)

    print(
        "Fixed 300        :",
        pct(
            baseline[
                "metadata"
            ][
                "prefix_coverage"
            ]
        ),
    )

    print(
        "Improved chunker :",
        pct(
            improved[
                "metadata"
            ][
                "prefix_coverage"
            ]
        ),
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "documents": len(docs),
                "baseline": baseline,
                "improved": improved,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()