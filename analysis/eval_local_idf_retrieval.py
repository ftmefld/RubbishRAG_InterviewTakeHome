import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from rubbish_rag.normalize_fa import normalize_fa


CORPUS_PATH = Path("corpus/test.csv")
EVAL_PATH = Path("analysis/paraphrase_eval.json")

# Standard BM25 defaults.
# These are not tuned on our evaluation set.
BM25_K1 = 1.2
BM25_B = 0.75


# ---------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------

def tokenize(text):
    """
    Persian normalization + punctuation-aware tokenization.

    Example:
        صادرات/واردات
    becomes:
        صادرات
        واردات
    """

    text = normalize_fa(
        text or ""
    )

    return re.findall(
        r"[A-Za-z0-9\u0600-\u06FF]+",
        text,
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

            raw_id = row.get(
                "doc_id"
            )

            if raw_id:
                try:
                    doc_id = int(
                        raw_id
                    )
                except ValueError:
                    doc_id = raw_id
            else:
                doc_id = row_idx

            question = (
                row.get("Question")
                or ""
            )

            tokens = tokenize(
                question
            )

            docs.append(
                {
                    "doc_id": doc_id,
                    "question": question,
                    "tokens": tokens,
                    "tf": Counter(tokens),
                    "length": len(tokens),
                }
            )

    return docs


# ---------------------------------------------------------------------
# Corpus statistics
# ---------------------------------------------------------------------

def build_statistics(docs):

    document_frequency = (
        defaultdict(int)
    )

    for doc in docs:

        for token in set(
            doc["tokens"]
        ):
            document_frequency[
                token
            ] += 1

    n_docs = len(docs)

    avg_length = (
        sum(
            doc["length"]
            for doc in docs
        )
        / n_docs
    )

    return {
        "n_docs": n_docs,
        "df": document_frequency,
        "avg_length": avg_length,
    }


# ---------------------------------------------------------------------
# Baseline Count Cosine
# ---------------------------------------------------------------------

def cosine_counter(a, b):

    if not a or not b:
        return 0.0

    common = (
        set(a)
        & set(b)
    )

    numerator = sum(
        a[token] * b[token]
        for token in common
    )

    norm_a = math.sqrt(
        sum(
            value * value
            for value in a.values()
        )
    )

    norm_b = math.sqrt(
        sum(
            value * value
            for value in b.values()
        )
    )

    if (
        norm_a == 0
        or norm_b == 0
    ):
        return 0.0

    return (
        numerator
        / (norm_a * norm_b)
    )


# ---------------------------------------------------------------------
# TF-IDF Cosine
# ---------------------------------------------------------------------

def tfidf_idf(
    token,
    stats,
):
    """
    Smooth IDF.

    Common words receive lower weight;
    rare discriminative words receive higher weight.
    """

    n_docs = stats[
        "n_docs"
    ]

    df = stats[
        "df"
    ].get(
        token,
        0,
    )

    return (
        math.log(
            (n_docs + 1)
            / (df + 1)
        )
        + 1.0
    )


def tfidf_vector(
    counter,
    stats,
):

    vector = {}

    for token, tf in (
        counter.items()
    ):

        vector[token] = (
            tf
            * tfidf_idf(
                token,
                stats,
            )
        )

    return vector


def cosine_vector(a, b):

    if not a or not b:
        return 0.0

    common = (
        set(a)
        & set(b)
    )

    numerator = sum(
        a[token]
        * b[token]
        for token in common
    )

    norm_a = math.sqrt(
        sum(
            value * value
            for value in a.values()
        )
    )

    norm_b = math.sqrt(
        sum(
            value * value
            for value in b.values()
        )
    )

    if (
        norm_a == 0
        or norm_b == 0
    ):
        return 0.0

    return (
        numerator
        / (norm_a * norm_b)
    )


# ---------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------

def bm25_idf(
    token,
    stats,
):

    n_docs = stats[
        "n_docs"
    ]

    df = stats[
        "df"
    ].get(
        token,
        0,
    )

    return math.log(
        1.0
        + (
            n_docs
            - df
            + 0.5
        )
        / (
            df
            + 0.5
        )
    )


def bm25_score(
    query_tokens,
    doc,
    stats,
):

    if not query_tokens:
        return 0.0

    score = 0.0

    avgdl = stats[
        "avg_length"
    ]

    dl = doc[
        "length"
    ]

    tf = doc[
        "tf"
    ]

    # Repeated query words should not
    # repeatedly add the same BM25 evidence.
    for token in set(
        query_tokens
    ):

        frequency = tf.get(
            token,
            0,
        )

        if frequency == 0:
            continue

        idf = bm25_idf(
            token,
            stats,
        )

        denominator = (
            frequency
            + BM25_K1
            * (
                1
                - BM25_B
                + BM25_B
                * dl
                / avgdl
            )
        )

        score += (
            idf
            * (
                frequency
                * (
                    BM25_K1
                    + 1
                )
            )
            / denominator
        )

    return score


# ---------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------

def rank_query(
    query,
    docs,
    stats,
):

    query_tokens = tokenize(
        query
    )

    query_counter = Counter(
        query_tokens
    )

    query_tfidf = (
        tfidf_vector(
            query_counter,
            stats,
        )
    )

    scored = []

    for doc in docs:

        item = dict(doc)

        # -----------------------------
        # Current baseline
        # -----------------------------

        item[
            "count_score"
        ] = cosine_counter(
            query_counter,
            doc["tf"],
        )

        # -----------------------------
        # TF-IDF cosine
        # -----------------------------

        doc_tfidf = (
            tfidf_vector(
                doc["tf"],
                stats,
            )
        )

        item[
            "tfidf_score"
        ] = cosine_vector(
            query_tfidf,
            doc_tfidf,
        )

        # -----------------------------
        # BM25
        # -----------------------------

        item[
            "bm25_score"
        ] = bm25_score(
            query_tokens,
            doc,
            stats,
        )

        scored.append(
            item
        )

    rankings = {}

    for score_name in (
        "count_score",
        "tfidf_score",
        "bm25_score",
    ):

        rankings[
            score_name
        ] = sorted(
            scored,
            key=lambda item: (
                item[
                    score_name
                ],
                -(
                    item["doc_id"]
                    if isinstance(
                        item["doc_id"],
                        int,
                    )
                    else 0
                ),
            ),
            reverse=True,
        )

    return rankings


def rank_of(
    ranked,
    gold_doc_id,
):

    target = str(
        gold_doc_id
    )

    for rank, item in enumerate(
        ranked,
        start=1,
    ):

        if str(
            item["doc_id"]
        ) == target:

            return rank

    return None


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_metrics(ranks):

    total = len(
        ranks
    )

    def hit_at(k):

        return (
            sum(
                rank is not None
                and rank <= k
                for rank in ranks
            )
            / total
        )

    mrr = (
        sum(
            (
                1.0 / rank
                if rank is not None
                else 0.0
            )
            for rank in ranks
        )
        / total
    )

    return {
        "H1": hit_at(1),
        "H3": hit_at(3),
        "H5": hit_at(5),
        "H20": hit_at(20),
        "MRR": mrr,
    }


def print_metrics(
    name,
    values,
):

    print(
        f"{name:<10} "
        f"H1="
        f"{values['H1']*100:5.1f}% "
        f"H3="
        f"{values['H3']*100:5.1f}% "
        f"H5="
        f"{values['H5']*100:5.1f}% "
        f"H20="
        f"{values['H20']*100:5.1f}% "
        f"MRR="
        f"{values['MRR']:.3f}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    docs = load_corpus()

    stats = (
        build_statistics(
            docs
        )
    )

    with EVAL_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        cases = json.load(f)

    all_ranks = {
        "COUNT": [],
        "TFIDF": [],
        "BM25": [],
    }

    interesting = {
        "v1",
        "v3",
        "v9",
        "v10",
        "v14",
        "v19",
    }

    print(
        "\nLocal IDF-Aware Retrieval Ablation"
    )

    print(
        "=" * 100
    )

    print(
        f"Corpus docs : "
        f"{len(docs)}"
    )

    print(
        f"Queries     : "
        f"{len(cases)}"
    )

    print(
        "Retrieval field: Question only"
    )

    print(
        "Answer/BriefAnswer used: NO"
    )

    print(
        "Remote API calls: 0"
    )

    print(
        f"BM25 constants: "
        f"k1={BM25_K1}, "
        f"b={BM25_B}"
    )

    for case in cases:

        rankings = rank_query(
            case["query"],
            docs,
            stats,
        )

        gold = case[
            "gold_doc_id"
        ]

        count_rank = rank_of(
            rankings[
                "count_score"
            ],
            gold,
        )

        tfidf_rank = rank_of(
            rankings[
                "tfidf_score"
            ],
            gold,
        )

        bm25_rank = rank_of(
            rankings[
                "bm25_score"
            ],
            gold,
        )

        all_ranks[
            "COUNT"
        ].append(
            count_rank
        )

        all_ranks[
            "TFIDF"
        ].append(
            tfidf_rank
        )

        all_ranks[
            "BM25"
        ].append(
            bm25_rank
        )

        if case["id"] in interesting:

            print(
                f"\n{case['id']} "
                f"gold={gold}"
            )

            print(
                f"  COUNT : "
                f"{count_rank}"
            )

            print(
                f"  TFIDF : "
                f"{tfidf_rank}"
            )

            print(
                f"  BM25  : "
                f"{bm25_rank}"
            )

            print(
                "  TFIDF Top-5:",
                [
                    item[
                        "doc_id"
                    ]
                    for item
                    in rankings[
                        "tfidf_score"
                    ][:5]
                ],
            )

            print(
                "  BM25  Top-5:",
                [
                    item[
                        "doc_id"
                    ]
                    for item
                    in rankings[
                        "bm25_score"
                    ][:5]
                ],
            )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FULL DEV-20 METRICS"
    )

    print(
        "-" * 100
    )

    for name in (
        "COUNT",
        "TFIDF",
        "BM25",
    ):

        print_metrics(
            name,
            compute_metrics(
                all_ranks[
                    name
                ]
            ),
        )


if __name__ == "__main__":
    main()