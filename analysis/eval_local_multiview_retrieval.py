import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from rubbish_rag.normalize_fa import normalize_fa


CORPUS_PATH = Path("corpus/test.csv")
EVAL_PATH = Path("analysis/paraphrase_eval.json")

RRF_K = 60


# ---------------------------------------------------------------------
# Text representations
# ---------------------------------------------------------------------

def tokenize(text):
    """
    Normalize Persian first, then extract actual lexical tokens.

    Important difference from split():
    punctuation such as '/' no longer glues words together.
    """
    text = normalize_fa(text or "")

    return re.findall(
        r"[A-Za-z0-9\u0600-\u06FF]+",
        text,
    )


def token_counter(text):
    return Counter(
        tokenize(text)
    )


def cosine_counter(a, b):
    if not a or not b:
        return 0.0

    common = set(a) & set(b)

    numerator = sum(
        a[token] * b[token]
        for token in common
    )

    a_norm = math.sqrt(
        sum(v * v for v in a.values())
    )

    b_norm = math.sqrt(
        sum(v * v for v in b.values())
    )

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return numerator / (
        a_norm * b_norm
    )


def char_ngrams(text, n=3):
    """
    Character trigrams over normalized text.

    Spaces are retained between words but repeated whitespace is collapsed.
    """
    text = normalize_fa(
        text or ""
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    # Convert punctuation boundaries into spaces so:
    # صادرات/واردات -> صادرات واردات
    text = re.sub(
        r"[^\w\u0600-\u06FF]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if len(text) < n:
        return Counter(
            [text]
        ) if text else Counter()

    return Counter(
        text[i : i + n]
        for i in range(
            len(text) - n + 1
        )
    )


# ---------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------

def load_corpus():
    docs = []

    with CORPUS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row_idx, row in enumerate(
            reader
        ):
            raw_id = row.get("doc_id")

            if raw_id:
                try:
                    doc_id = int(raw_id)
                except ValueError:
                    doc_id = raw_id
            else:
                doc_id = row_idx

            question = (
                row.get("Question")
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

            metadata = (
                f"{keyword} {category}"
            )

            docs.append(
                {
                    "doc_id": doc_id,
                    "question": question,
                    "keyword": keyword,
                    "category": category,

                    "q_words": (
                        token_counter(
                            question
                        )
                    ),

                    "q_chars": (
                        char_ngrams(
                            question
                        )
                    ),

                    "meta_words": (
                        token_counter(
                            metadata
                        )
                    ),
                }
            )

    return docs


# ---------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------

def rank_by_score(
    docs,
    score_name,
):
    return sorted(
        docs,
        key=lambda d: (
            d[score_name],
            -(
                d["doc_id"]
                if isinstance(
                    d["doc_id"],
                    int,
                )
                else 0
            ),
        ),
        reverse=True,
    )


def rank_map(
    ranked,
):
    return {
        str(item["doc_id"]): rank
        for rank, item in enumerate(
            ranked,
            start=1,
        )
    }


def rank_of(
    ranked,
    gold,
):
    target = str(gold)

    for rank, item in enumerate(
        ranked,
        start=1,
    ):
        if str(
            item["doc_id"]
        ) == target:
            return rank

    return None


def multiview_rank(
    query,
    corpus,
):
    query_words = token_counter(
        query
    )

    query_chars = char_ngrams(
        query
    )

    scored = []

    for doc in corpus:
        item = dict(doc)

        item["word_score"] = (
            cosine_counter(
                query_words,
                doc["q_words"],
            )
        )

        item["char_score"] = (
            cosine_counter(
                query_chars,
                doc["q_chars"],
            )
        )

        item["meta_score"] = (
            cosine_counter(
                query_words,
                doc["meta_words"],
            )
        )

        scored.append(item)

    word_ranked = rank_by_score(
        scored,
        "word_score",
    )

    char_ranked = rank_by_score(
        scored,
        "char_score",
    )

    meta_ranked = rank_by_score(
        scored,
        "meta_score",
    )

    word_ranks = rank_map(
        word_ranked
    )

    char_ranks = rank_map(
        char_ranked
    )

    meta_ranks = rank_map(
        meta_ranked
    )

    fused = []

    for item in scored:
        key = str(
            item["doc_id"]
        )

        item = dict(item)

        item["rrf_score"] = (
            1.0
            / (
                RRF_K
                + word_ranks[key]
            )
            + 1.0
            / (
                RRF_K
                + char_ranks[key]
            )
            + 1.0
            / (
                RRF_K
                + meta_ranks[key]
            )
        )

        item["word_rank"] = (
            word_ranks[key]
        )

        item["char_rank"] = (
            char_ranks[key]
        )

        item["meta_rank"] = (
            meta_ranks[key]
        )

        fused.append(item)

    fused.sort(
        key=lambda item: (
            item["rrf_score"],
            item["char_score"],
            item["word_score"],
        ),
        reverse=True,
    )

    return {
        "word": word_ranked,
        "char": char_ranked,
        "meta": meta_ranked,
        "rrf": fused,
    }


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def metrics(ranks):
    total = len(ranks)

    def hit(k):
        return sum(
            rank is not None
            and rank <= k
            for rank in ranks
        ) / total

    mrr = sum(
        (
            1.0 / rank
            if rank is not None
            else 0.0
        )
        for rank in ranks
    ) / total

    return {
        "H1": hit(1),
        "H3": hit(3),
        "H5": hit(5),
        "H20": hit(20),
        "MRR": mrr,
    }


def print_metrics(
    name,
    values,
):
    print(
        f"{name:<12} "
        f"H1={values['H1']*100:5.1f}% "
        f"H3={values['H3']*100:5.1f}% "
        f"H5={values['H5']*100:5.1f}% "
        f"H20={values['H20']*100:5.1f}% "
        f"MRR={values['MRR']:.3f}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    corpus = load_corpus()

    with EVAL_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        cases = json.load(f)

    all_ranks = {
        "word": [],
        "char": [],
        "meta": [],
        "rrf": [],
    }

    important = {
        "v3",
        "v9",
        "v10",
        "v12",
    }

    print(
        "\nLocal Multi-View Retrieval Ablation"
    )

    print(
        "=" * 100
    )

    print(
        f"Corpus docs : {len(corpus)}"
    )

    print(
        f"Queries     : {len(cases)}"
    )

    print(
        "Answer/BriefAnswer used for retrieval: NO"
    )

    print(
        "Remote API calls: 0"
    )

    for case in cases:
        results = multiview_rank(
            case["query"],
            corpus,
        )

        gold = case[
            "gold_doc_id"
        ]

        ranks = {}

        for name in (
            "word",
            "char",
            "meta",
            "rrf",
        ):
            rank = rank_of(
                results[name],
                gold,
            )

            all_ranks[
                name
            ].append(rank)

            ranks[name] = rank

        if case["id"] in important:
            gold_item = next(
                item
                for item
                in results["rrf"]
                if str(
                    item["doc_id"]
                )
                == str(gold)
            )

            print(
                f"\n{case['id']} "
                f"gold={gold}"
            )

            print(
                f"  word rank : "
                f"{ranks['word']}"
            )

            print(
                f"  char rank : "
                f"{ranks['char']}"
            )

            print(
                f"  meta rank : "
                f"{ranks['meta']}"
            )

            print(
                f"  RRF rank  : "
                f"{ranks['rrf']}"
            )

            print(
                f"  scores    : "
                f"word="
                f"{gold_item['word_score']:.4f} "
                f"char="
                f"{gold_item['char_score']:.4f} "
                f"meta="
                f"{gold_item['meta_score']:.4f}"
            )

            print(
                "  Top-5 RRF :",
                [
                    item["doc_id"]
                    for item
                    in results["rrf"][:5]
                ],
            )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FULL 20-QUERY METRICS"
    )

    print(
        "-" * 100
    )

    for name in (
        "word",
        "char",
        "meta",
        "rrf",
    ):
        print_metrics(
            name.upper(),
            metrics(
                all_ranks[name]
            ),
        )


if __name__ == "__main__":
    main()