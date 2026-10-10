import json
from pathlib import Path

import rubbish_rag as R
import rubbish_rag.pipeline  # noqa: F401 - registers pipeline stages


BENCH_PATH = Path("server/visible_bench.json")


def gold_rank(hits, gold_doc_id):
    for rank, hit in enumerate(hits, start=1):
        if hit["doc_id"] == gold_doc_id:
            return rank

    return None


def hit_at(rank, k):
    return rank is not None and rank <= k


def main():
    with BENCH_PATH.open("r", encoding="utf-8") as f:
        benchmark = json.load(f)

    retriever = R.get("retriever")

    print("Registered retriever:")
    print(" module :", retriever.__module__)
    print(" name   :", retriever.__name__)

    hit3 = 0
    hit5 = 0

    print("\nID   GoldRank   Hit@3   Hit@5")
    print("-" * 40)

    for item in benchmark:
        hits = retriever(
            item["query"],
            topk=5,
        )

        rank = gold_rank(
            hits,
            item["gold_doc_id"],
        )

        h3 = hit_at(rank, 3)
        h5 = hit_at(rank, 5)

        hit3 += h3
        hit5 += h5

        rank_text = (
            str(rank)
            if rank is not None
            else "MISS"
        )

        print(
            f"{item['id']:<5}"
            f"{rank_text:<11}"
            f"{str(h3):<8}"
            f"{h5}"
        )

    n = len(benchmark)

    print("\nRegistered retriever metrics")
    print("=" * 40)

    print(
        f"Hit@3 = {hit3 / n:.3f} "
        f"({100 * hit3 / n:.1f}%)"
    )

    print(
        f"Hit@5 = {hit5 / n:.3f} "
        f"({100 * hit5 / n:.1f}%)"
    )


if __name__ == "__main__":
    main()