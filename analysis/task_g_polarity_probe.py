import json
from pathlib import Path

import rubbish_rag as R
import rubbish_rag.pipeline

from rubbish_rag.slots import extract_slots
from server._hidden_retriever import remote_retrieve


BENCH_PATH = Path("server/visible_bench.json")
OUTPUT_PATH = Path(
    "analysis/task_g_polarity_baseline.json"
)

TOPK = 5


def is_negated(text):
    return bool(
        extract_slots(text or "").get(
            "NEGATED"
        )
    )


def main():
    with BENCH_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        bench = json.load(f)

    reranker = R.get("reranker")

    negated_queries = [
        item
        for item in bench
        if is_negated(item["query"])
    ]

    rows = []

    total_top3 = 0
    polarity_mismatch_top3 = 0
    queries_with_mismatch = 0

    print(
        "\nTrack G — Negation / Polarity Baseline"
    )
    print("=" * 95)

    print(
        f"Negated visible queries: "
        f"{len(negated_queries)}/{len(bench)}"
    )

    for item in negated_queries:
        query = item["query"]

        raw_hits = remote_retrieve(
            query,
            topk=TOPK,
        )

        ranked = reranker(
            query,
            [dict(hit) for hit in raw_hits],
        )

        top3 = ranked[:3]

        mismatches = []

        print("\n" + "-" * 95)

        print(
            f"{item['id']} | gold={item['gold_doc_id']}"
        )

        print(
            f"QUERY: {query}"
        )

        print(
            "Query polarity: NEGATED"
        )

        for rank, hit in enumerate(
            top3,
            start=1,
        ):
            candidate_negated = is_negated(
                hit.get("text", "")
            )

            mismatch = not candidate_negated

            total_top3 += 1

            if mismatch:
                polarity_mismatch_top3 += 1

                mismatches.append(
                    hit["doc_id"]
                )

            print(
                f"\n  rank={rank} "
                f"doc={hit['doc_id']} "
                f"candidate_negated="
                f"{candidate_negated} "
                f"mismatch={mismatch}"
            )

            print(
                "  text="
                + repr(
                    hit.get(
                        "text",
                        "",
                    )[:220]
                )
            )

        if mismatches:
            queries_with_mismatch += 1

        rows.append(
            {
                "id": item["id"],
                "query": query,
                "gold_doc_id": (
                    item["gold_doc_id"]
                ),
                "mismatch_doc_ids": (
                    mismatches
                ),
                "top3": [
                    {
                        "doc_id": hit["doc_id"],
                        "candidate_negated": (
                            is_negated(
                                hit.get(
                                    "text",
                                    "",
                                )
                            )
                        ),
                        "text": hit.get(
                            "text",
                            "",
                        ),
                    }
                    for hit in top3
                ],
            }
        )

    mismatch_rate = (
        polarity_mismatch_top3
        / total_top3
        if total_top3
        else 0.0
    )

    print("\n" + "=" * 95)

    print("Polarity diagnostic")
    print("-" * 95)

    print(
        f"Queries with at least one "
        f"polarity mismatch in Top-3 : "
        f"{queries_with_mismatch}/"
        f"{len(negated_queries)}"
    )

    print(
        f"Top-3 polarity mismatches       : "
        f"{polarity_mismatch_top3}/"
        f"{total_top3} "
        f"({100 * mismatch_rate:.1f}%)"
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
                "negated_queries": len(
                    negated_queries
                ),
                "queries_with_mismatch": (
                    queries_with_mismatch
                ),
                "top3_mismatches": (
                    polarity_mismatch_top3
                ),
                "top3_total": (
                    total_top3
                ),
                "mismatch_rate": (
                    mismatch_rate
                ),
                "details": rows,
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