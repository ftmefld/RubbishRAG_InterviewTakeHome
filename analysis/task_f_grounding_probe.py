import csv
import json
from pathlib import Path

from server._hidden_retriever import remote_retrieve
from rubbish_rag.normalize_fa import normalize_fa


BENCH_PATH = Path("server/visible_bench.json")
CORPUS_PATH = Path("corpus/test.csv")
OUTPUT_PATH = Path("analysis/task_f_grounding_probe_results.json")

TOPK = 5


def clean(text):
    return normalize_fa(text or "").strip()


def load_trusted_corpus():
    """
    Build a normalized trusted content set from the clean corpus.

    We keep Answer and BriefAnswer separately because server hits may
    originate from either representation.
    """
    records = []

    with CORPUS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row_index, row in enumerate(reader):
            answer = clean(
                row.get("Answer", "")
            )

            brief = clean(
                row.get("BriefAnswer", "")
            )

            question = clean(
                row.get("Question", "")
            )

            category = clean(
                row.get("Category", "")
            )

            keyword = clean(
                row.get("Keyword", "")
            )

            records.append(
                {
                    "row_index": row_index,
                    "answer": answer,
                    "brief": brief,
                    "question": question,
                    "category": category,
                    "keyword": keyword,
                }
            )

    return records


def grounding_match(hit_text, trusted_records):
    """
    Conservative content-grounding detector.

    A retrieved hit is considered grounded if its normalized text is
    an exact substring of a trusted Answer or BriefAnswer, or if a
    trusted Answer/BriefAnswer is fully contained in the hit.

    This does NOT use doc_id.
    """
    hit = clean(hit_text)

    if not hit:
        return {
            "grounded": False,
            "matched_row": None,
            "matched_field": None,
        }

    for record in trusted_records:
        for field in ("answer", "brief"):
            trusted_text = record[field]

            if not trusted_text:
                continue

            if (
                hit in trusted_text
                or trusted_text in hit
            ):
                return {
                    "grounded": True,
                    "matched_row": record["row_index"],
                    "matched_field": field,
                }

    return {
        "grounded": False,
        "matched_row": None,
        "matched_field": None,
    }


def poison_label_for_evaluation(hit):
    """
    Evaluation-only label from challenge metadata.

    IMPORTANT:
    This is NOT used by the detector itself.
    """
    try:
        return int(hit["doc_id"]) >= 9000
    except (TypeError, ValueError, KeyError):
        return False


def main():
    trusted_records = load_trusted_corpus()

    with BENCH_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        benchmark = json.load(f)

    rows = []

    # Detector confusion matrix:
    #
    # predicted poison = NOT grounded
    # true poison      = doc_id >= 9000  [evaluation only]
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    total_hits = 0
    total_poison = 0
    total_genuine = 0

    query_summaries = []

    print("\nTrack F — Trusted-Corpus Grounding Probe")
    print("=" * 88)

    for example in benchmark:
        hits = remote_retrieve(
            example["query"],
            topk=TOPK,
        )

        query_rows = []

        for rank, hit in enumerate(
            hits,
            start=1,
        ):
            total_hits += 1

            true_poison = (
                poison_label_for_evaluation(
                    hit
                )
            )

            match = grounding_match(
                hit.get("text", ""),
                trusted_records,
            )

            predicted_poison = (
                not match["grounded"]
            )

            if true_poison:
                total_poison += 1
            else:
                total_genuine += 1

            if true_poison and predicted_poison:
                true_positive += 1

            elif (
                not true_poison
                and predicted_poison
            ):
                false_positive += 1

            elif (
                not true_poison
                and not predicted_poison
            ):
                true_negative += 1

            else:
                false_negative += 1

            row = {
                "query_id": example["id"],
                "rank": rank,
                "doc_id": hit["doc_id"],
                "true_poison_eval_only": (
                    true_poison
                ),
                "grounded": match["grounded"],
                "predicted_poison": (
                    predicted_poison
                ),
                "matched_row": (
                    match["matched_row"]
                ),
                "matched_field": (
                    match["matched_field"]
                ),
                "text": hit.get(
                    "text",
                    "",
                ),
            }

            rows.append(row)
            query_rows.append(row)

        ungrounded_ids = [
            row["doc_id"]
            for row in query_rows
            if not row["grounded"]
        ]

        poison_ids = [
            row["doc_id"]
            for row in query_rows
            if row[
                "true_poison_eval_only"
            ]
        ]

        query_summaries.append(
            {
                "id": example["id"],
                "ungrounded_ids": (
                    ungrounded_ids
                ),
                "known_poison_ids": (
                    poison_ids
                ),
            }
        )

        if (
            ungrounded_ids
            or poison_ids
        ):
            print(
                f"{example['id']:<5} "
                f"ungrounded={ungrounded_ids} "
                f"known_poison={poison_ids}"
            )

    precision = (
        true_positive
        / (true_positive + false_positive)
        if (
            true_positive
            + false_positive
        )
        else 0.0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if (
            true_positive
            + false_negative
        )
        else 0.0
    )

    specificity = (
        true_negative
        / (true_negative + false_positive)
        if (
            true_negative
            + false_positive
        )
        else 0.0
    )

    accuracy = (
        true_positive
        + true_negative
    ) / total_hits

    print("\nDetector evaluation")
    print("=" * 60)

    print(
        f"Retrieved hits       : {total_hits}"
    )

    print(
        f"Known poison hits    : {total_poison}"
    )

    print(
        f"Known genuine hits   : {total_genuine}"
    )

    print()

    print(
        f"TP poison detected   : "
        f"{true_positive}"
    )

    print(
        f"FP genuine rejected  : "
        f"{false_positive}"
    )

    print(
        f"TN genuine accepted  : "
        f"{true_negative}"
    )

    print(
        f"FN poison accepted   : "
        f"{false_negative}"
    )

    print()

    print(
        f"Poison precision     : "
        f"{precision:.3f}"
    )

    print(
        f"Poison recall        : "
        f"{recall:.3f}"
    )

    print(
        f"Genuine specificity  : "
        f"{specificity:.3f}"
    )

    print(
        f"Overall accuracy     : "
        f"{accuracy:.3f}"
    )

    print("\nFalse positives")
    print("-" * 60)

    fps = [
        row
        for row in rows
        if (
            not row[
                "true_poison_eval_only"
            ]
            and row["predicted_poison"]
        )
    ]

    for row in fps[:20]:
        print(
            f"{row['query_id']} "
            f"rank={row['rank']} "
            f"doc={row['doc_id']} "
            f"text={row['text'][:100]!r}"
        )

    print("\nFalse negatives")
    print("-" * 60)

    fns = [
        row
        for row in rows
        if (
            row[
                "true_poison_eval_only"
            ]
            and not row["predicted_poison"]
        )
    ]

    for row in fns[:20]:
        print(
            f"{row['query_id']} "
            f"rank={row['rank']} "
            f"doc={row['doc_id']} "
            f"text={row['text'][:100]!r}"
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
                "total_hits": total_hits,
                "known_poison_hits": (
                    total_poison
                ),
                "known_genuine_hits": (
                    total_genuine
                ),
                "confusion_matrix": {
                    "tp": true_positive,
                    "fp": false_positive,
                    "tn": true_negative,
                    "fn": false_negative,
                },
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "accuracy": accuracy,
                "queries": query_summaries,
                "hits": rows,
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