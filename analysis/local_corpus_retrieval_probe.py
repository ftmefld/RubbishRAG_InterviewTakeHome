import csv
import json
import math
from collections import Counter
from pathlib import Path

from rubbish_rag.normalize_fa import normalize_fa


CORPUS_PATH = Path("corpus/test.csv")
BENCH_PATH = Path("server/visible_bench.json")
OUTPUT_PATH = Path(
    "analysis/local_corpus_retrieval_results.json"
)


def tokenize(text):
    normalized = normalize_fa(
        text or ""
    )

    return [
        token.strip(
            "؟?،,.:;!()[]«»\"'"
        )
        for token in normalized.split()
        if token.strip(
            "؟?،,.:;!()[]«»\"'"
        )
    ]


def cosine_overlap(query_tokens, doc_tokens):
    """
    Cosine similarity over term-frequency vectors.
    """
    q = Counter(query_tokens)
    d = Counter(doc_tokens)

    if not q or not d:
        return 0.0

    dot = sum(
        q[token] * d.get(token, 0)
        for token in q
    )

    q_norm = math.sqrt(
        sum(v * v for v in q.values())
    )

    d_norm = math.sqrt(
        sum(v * v for v in d.values())
    )

    if not q_norm or not d_norm:
        return 0.0

    return dot / (
        q_norm * d_norm
    )


def load_corpus():
    docs = []

    with CORPUS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for doc_id, row in enumerate(reader):
            question = (
                row.get("Question")
                or ""
            )

            brief = (
                row.get("BriefAnswer")
                or ""
            )

            keyword = (
                row.get("Keyword")
                or ""
            )

            category = (
                row.get("Category")
                or ""
            )

            # Give the Question the strongest representation.
            retrieval_text = " ".join(
                [
                    question,
                    question,
                    keyword,
                    category,
                    brief,
                ]
            )

            docs.append(
                {
                    "doc_id": doc_id,
                    "question": question,
                    "brief": brief,
                    "keyword": keyword,
                    "category": category,
                    "tokens": tokenize(
                        retrieval_text
                    ),
                }
            )

    return docs


def rank_local(query, docs):
    query_tokens = tokenize(query)

    scored = []

    for doc in docs:
        score = cosine_overlap(
            query_tokens,
            doc["tokens"],
        )

        scored.append(
            (
                score,
                doc["doc_id"],
                doc,
            )
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return scored


def rank_of(ranked, gold_doc_id):
    for rank, (
        score,
        doc_id,
        doc,
    ) in enumerate(
        ranked,
        start=1,
    ):
        if str(doc_id) == str(
            gold_doc_id
        ):
            return rank

    return None


def main():
    docs = load_corpus()

    with BENCH_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        benchmark = json.load(f)

    hit1 = 0
    hit3 = 0
    hit5 = 0
    hit10 = 0
    hit20 = 0

    rows = []

    print(
        "\nLocal Trusted-Corpus Retrieval Probe"
    )
    print("=" * 92)

    for item in benchmark:
        ranked = rank_local(
            item["query"],
            docs,
        )

        gold_rank = rank_of(
            ranked,
            item["gold_doc_id"],
        )

        hit1 += int(
            gold_rank is not None
            and gold_rank <= 1
        )

        hit3 += int(
            gold_rank is not None
            and gold_rank <= 3
        )

        hit5 += int(
            gold_rank is not None
            and gold_rank <= 5
        )

        hit10 += int(
            gold_rank is not None
            and gold_rank <= 10
        )

        hit20 += int(
            gold_rank is not None
            and gold_rank <= 20
        )

        top5 = [
            {
                "doc_id": doc_id,
                "score": round(
                    score,
                    4,
                ),
                "question": (
                    doc["question"]
                ),
            }
            for score, doc_id, doc
            in ranked[:5]
        ]

        rows.append(
            {
                "id": item["id"],
                "gold_doc_id": (
                    item["gold_doc_id"]
                ),
                "gold_rank": gold_rank,
                "top5": top5,
            }
        )

        print(
            f"{item['id']:<5} "
            f"gold="
            f"{str(item['gold_doc_id']):<5} "
            f"local_rank="
            f"{str(gold_rank):<5} "
            f"top1="
            f"{ranked[0][1]}"
        )

    total = len(benchmark)

    print("\nLocal retrieval recall")
    print("-" * 92)

    print(
        f"Hit@1  : {hit1}/{total} "
        f"({100 * hit1 / total:.1f}%)"
    )

    print(
        f"Hit@3  : {hit3}/{total} "
        f"({100 * hit3 / total:.1f}%)"
    )

    print(
        f"Hit@5  : {hit5}/{total} "
        f"({100 * hit5 / total:.1f}%)"
    )

    print(
        f"Hit@10 : {hit10}/{total} "
        f"({100 * hit10 / total:.1f}%)"
    )

    print(
        f"Hit@20 : {hit20}/{total} "
        f"({100 * hit20 / total:.1f}%)"
    )

    print(
        "\nPreviously absent from remote Top-20"
    )
    print("-" * 92)

    missing_remote = {
        "v3",
        "v6",
        "v9",
        "v12",
        "v16",
        "v17",
        "v18",
    }

    for row in rows:
        if row["id"] in missing_remote:
            print(
                f"{row['id']}: "
                f"local_rank="
                f"{row['gold_rank']}"
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
                "hit_at_1": (
                    hit1 / total
                ),
                "hit_at_3": (
                    hit3 / total
                ),
                "hit_at_5": (
                    hit5 / total
                ),
                "hit_at_10": (
                    hit10 / total
                ),
                "hit_at_20": (
                    hit20 / total
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