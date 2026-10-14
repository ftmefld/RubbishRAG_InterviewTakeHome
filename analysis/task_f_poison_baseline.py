import csv
import json
from pathlib import Path

from server._hidden_retriever import remote_retrieve
from rubbish_rag.normalize_fa import normalize_fa


BENCH_PATH = Path("server/visible_bench.json")
CORPUS_PATH = Path("corpus/test.csv")
OUTPUT_PATH = Path("analysis/task_f_poison_baseline.json")


def load_verified_doc_ids():
    with CORPUS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        return {
            str(row["doc_id"]).strip()
            for row in reader
            if row.get("doc_id")
        }


def is_verified(hit, verified_ids):
    return (
        str(hit.get("doc_id", "")).strip()
        in verified_ids
    )


def baseline_rerank(query, hits):
    """
    Current Task-C reranker, before poison defense.
    """
    normalized_query = normalize_fa(query)

    query_tokens = set(
        normalized_query.split()
    )

    rescored = []

    for hit in hits:
        item = dict(hit)

        normalized_text = normalize_fa(
            item.get("text", "")
        )

        hit_tokens = set(
            normalized_text.split()
        )

        item["rerank_score"] = len(
            query_tokens & hit_tokens
        )

        rescored.append(item)

    return sorted(
        rescored,
        key=lambda h: (
            h["rerank_score"],
            h.get("fused", 0.0),
        ),
        reverse=True,
    )


def gold_rank(hits, gold_doc_id):
    for rank, hit in enumerate(
        hits,
        start=1,
    ):
        if hit["doc_id"] == gold_doc_id:
            return rank

    return None


def main():
    verified_ids = load_verified_doc_ids()

    with BENCH_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        benchmark = json.load(f)

    total_queries = len(benchmark)

    queries_with_poison = 0
    poison_citations = 0
    total_citations = 0

    hit3 = 0
    top1 = 0

    rows = []

    for example in benchmark:
        hits = remote_retrieve(
            example["query"],
            topk=5,
        )

        ranked = baseline_rerank(
            example["query"],
            hits,
        )

        top3 = ranked[:3]

        poison_top3 = [
            hit
            for hit in top3
            if not is_verified(
                hit,
                verified_ids,
            )
        ]

        if poison_top3:
            queries_with_poison += 1

        poison_citations += len(
            poison_top3
        )

        total_citations += len(
            top3
        )

        rank = gold_rank(
            ranked,
            example["gold_doc_id"],
        )

        if (
            rank is not None
            and rank <= 3
        ):
            hit3 += 1

        if rank == 1:
            top1 += 1

        rows.append(
            {
                "id": example["id"],
                "gold_doc_id": example[
                    "gold_doc_id"
                ],
                "gold_rank": rank,
                "top3": [
                    {
                        "doc_id": hit["doc_id"],
                        "verified": is_verified(
                            hit,
                            verified_ids,
                        ),
                    }
                    for hit in top3
                ],
                "poison_doc_ids": [
                    hit["doc_id"]
                    for hit in poison_top3
                ],
            }
        )

    query_poison_rate = (
        queries_with_poison
        / total_queries
    )

    citation_poison_rate = (
        poison_citations
        / total_citations
        if total_citations
        else 0.0
    )

    print(
        "\nTrack F — Poison Defense Baseline"
    )
    print("=" * 65)

    print(
        f"Queries with unverified Top-3 evidence : "
        f"{queries_with_poison}/{total_queries} "
        f"({query_poison_rate * 100:.1f}%)"
    )

    print(
        f"Unverified Top-3 citation rate        : "
        f"{poison_citations}/{total_citations} "
        f"({citation_poison_rate * 100:.1f}%)"
    )

    print(
        f"Gold Top-1 rate                       : "
        f"{top1}/{total_queries} "
        f"({top1 / total_queries * 100:.1f}%)"
    )

    print(
        f"Gold Hit@3                            : "
        f"{hit3}/{total_queries} "
        f"({hit3 / total_queries * 100:.1f}%)"
    )

    print(
        "\nQueries containing unverified evidence:"
    )

    for row in rows:
        if row["poison_doc_ids"]:
            print(
                f"  {row['id']}: "
                f"{row['poison_doc_ids']}"
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
                "query_poison_rate": (
                    query_poison_rate
                ),
                "citation_poison_rate": (
                    citation_poison_rate
                ),
                "queries_with_poison": (
                    queries_with_poison
                ),
                "poison_citations": (
                    poison_citations
                ),
                "top1_rate": (
                    top1 / total_queries
                ),
                "hit3": (
                    hit3 / total_queries
                ),
                "queries": rows,
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