import csv
import json
import math
from collections import Counter
from pathlib import Path

from rubbish_rag.normalize_fa import normalize_fa
from server._hidden_retriever import remote_retrieve


PARAPHRASE_PATH = Path(
    "analysis/paraphrase_eval.json"
)

CORPUS_PATH = Path(
    "corpus/test.csv"
)

REMOTE_CACHE_PATH = Path(
    "analysis/paraphrase_remote_cache.json"
)

OUTPUT_PATH = Path(
    "analysis/paraphrase_retrieval_results.json"
)

TOPK = 20

# Fixed before looking at evaluation results.
# We are NOT tuning this on the paraphrase set.
RRF_K = 60


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def normalize_doc_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def unique_preserve_order(items):
    seen = set()
    result = []

    for item in items:
        key = str(item)

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


def rank_of(doc_ids, gold_doc_id):
    gold = str(gold_doc_id)

    for rank, doc_id in enumerate(
        doc_ids,
        start=1,
    ):
        if str(doc_id) == gold:
            return rank

    return None


# ---------------------------------------------------------
# Local retrieval
# ---------------------------------------------------------

def token_counter(text):
    normalized = normalize_fa(
        text or ""
    )

    return Counter(
        normalized.split()
    )


def cosine_similarity(counter_a, counter_b):
    if not counter_a or not counter_b:
        return 0.0

    shared = (
        set(counter_a)
        & set(counter_b)
    )

    numerator = sum(
        counter_a[token]
        * counter_b[token]
        for token in shared
    )

    norm_a = math.sqrt(
        sum(
            value * value
            for value in counter_a.values()
        )
    )

    norm_b = math.sqrt(
        sum(
            value * value
            for value in counter_b.values()
        )
    )

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return numerator / (
        norm_a * norm_b
    )


def build_local_index(corpus):
    index = []

    for row_index, row in enumerate(corpus):
        # corpus/test.csv does not need an explicit
        # doc_id column; row position is the document ID
        # used by this dataset.
        explicit_doc_id = row.get(
            "doc_id"
        )

        if (
            explicit_doc_id is not None
            and str(explicit_doc_id).strip()
        ):
            doc_id = normalize_doc_id(
                explicit_doc_id
            )
        else:
            doc_id = row_index

        question = (
            row.get("Question")
            or ""
        )

        index.append(
            {
                "doc_id": doc_id,
                "question": question,
                "tokens": token_counter(
                    question
                ),
            }
        )

    return index


def local_retrieve(
    query,
    local_index,
    topk=TOPK,
):
    query_tokens = token_counter(
        query
    )

    scored = []

    for row in local_index:
        score = cosine_similarity(
            query_tokens,
            row["tokens"],
        )

        scored.append(
            {
                "doc_id": row[
                    "doc_id"
                ],
                "score": score,
            }
        )

    scored.sort(
        key=lambda item: (
            -item["score"],
            str(item["doc_id"]),
        )
    )

    return [
        item["doc_id"]
        for item in scored[:topk]
    ]


# ---------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------

def rrf_fuse(
    remote_ids,
    local_ids,
    topk=TOPK,
):
    """
    Equal-weight Reciprocal Rank Fusion.

    No learned/tuned weights are used.
    """

    scores = {}

    for rank, doc_id in enumerate(
        remote_ids,
        start=1,
    ):
        key = str(doc_id)

        scores.setdefault(
            key,
            {
                "doc_id": doc_id,
                "score": 0.0,
            },
        )

        scores[key]["score"] += (
            1.0 / (RRF_K + rank)
        )

    for rank, doc_id in enumerate(
        local_ids,
        start=1,
    ):
        key = str(doc_id)

        scores.setdefault(
            key,
            {
                "doc_id": doc_id,
                "score": 0.0,
            },
        )

        scores[key]["score"] += (
            1.0 / (RRF_K + rank)
        )

    ranked = sorted(
        scores.values(),
        key=lambda item: (
            -item["score"],
            str(item["doc_id"]),
        ),
    )

    return [
        item["doc_id"]
        for item in ranked[:topk]
    ]


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def compute_metrics(ranks, total):
    def hit_at(k):
        hits = sum(
            1
            for rank in ranks
            if (
                rank is not None
                and rank <= k
            )
        )

        return hits / total

    mrr = sum(
        (
            1.0 / rank
            if rank is not None
            else 0.0
        )
        for rank in ranks
    ) / total

    return {
        "hit_at_1": hit_at(1),
        "hit_at_3": hit_at(3),
        "hit_at_5": hit_at(5),
        "hit_at_20": hit_at(20),
        "mrr": mrr,
    }


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * 55)

    print(
        f"Hit@1  : "
        f"{100 * metrics['hit_at_1']:.1f}%"
    )

    print(
        f"Hit@3  : "
        f"{100 * metrics['hit_at_3']:.1f}%"
    )

    print(
        f"Hit@5  : "
        f"{100 * metrics['hit_at_5']:.1f}%"
    )

    print(
        f"Hit@20 : "
        f"{100 * metrics['hit_at_20']:.1f}%"
    )

    print(
        f"MRR     : "
        f"{metrics['mrr']:.3f}"
    )


# ---------------------------------------------------------
# Cache
# ---------------------------------------------------------

def load_remote_cache():
    if not REMOTE_CACHE_PATH.exists():
        return {}

    with REMOTE_CACHE_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_remote_cache(cache):
    REMOTE_CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REMOTE_CACHE_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            cache,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def main():
    with PARAPHRASE_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        cases = json.load(f)

    with CORPUS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        corpus = list(
            csv.DictReader(f)
        )

    local_index = build_local_index(
        corpus
    )

    # -----------------------------------------------------
    # Safety check:
    # do not evaluate exact corpus Question matches.
    # -----------------------------------------------------
    corpus_questions = {
        normalize_fa(
            row.get("Question") or ""
        )
        for row in corpus
    }

    exact_cases = [
        case["id"]
        for case in cases
        if normalize_fa(
            case["query"]
        ) in corpus_questions
    ]

    if exact_cases:
        raise RuntimeError(
            "Exact corpus Question matches "
            f"found: {exact_cases}"
        )

    print(
        "\nParaphrase Retrieval Evaluation"
    )

    print("=" * 85)

    print(
        f"Queries: {len(cases)}"
    )

    print(
        "Exact normalized matches: 0"
    )

    print(
        "Local representation: "
        "Question only"
    )

    print(
        "Local similarity: "
        "normalized token cosine"
    )

    print(
        f"Hybrid: equal-weight RRF "
        f"(k={RRF_K})"
    )

    # -----------------------------------------------------
    # Remote cache:
    # rerunning this script will NOT use API quota again
    # unless a query changed or is missing from cache.
    # -----------------------------------------------------
    remote_cache = load_remote_cache()

    remote_ranks = []
    local_ranks = []
    hybrid_ranks = []

    details = []

    for case in cases:
        case_id = case["id"]
        query = case["query"]

        gold = case[
            "gold_doc_id"
        ]

        cached = remote_cache.get(
            case_id
        )

        if (
            cached
            and cached.get("query")
            == query
        ):
            raw_remote_ids = [
                normalize_doc_id(x)
                for x in cached[
                    "doc_ids_raw"
                ]
            ]

            cache_status = "cache"

        else:
            hits = remote_retrieve(
                query,
                topk=TOPK,
            )

            raw_remote_ids = [
                normalize_doc_id(
                    hit["doc_id"]
                )
                for hit in hits
            ]

            remote_cache[
                case_id
            ] = {
                "query": query,
                "doc_ids_raw": (
                    raw_remote_ids
                ),
            }

            # Save immediately after every call.
            # If evaluation is interrupted, completed
            # retrieval calls are not wasted.
            save_remote_cache(
                remote_cache
            )

            cache_status = "API"

        # Document-level comparison:
        # duplicate chunks from the same document do
        # not receive multiple positions.
        remote_ids = (
            unique_preserve_order(
                raw_remote_ids
            )
        )

        local_ids = local_retrieve(
            query,
            local_index,
            topk=TOPK,
        )

        hybrid_ids = rrf_fuse(
            remote_ids,
            local_ids,
            topk=TOPK,
        )

        remote_rank = rank_of(
            remote_ids,
            gold,
        )

        local_rank = rank_of(
            local_ids,
            gold,
        )

        hybrid_rank = rank_of(
            hybrid_ids,
            gold,
        )

        remote_ranks.append(
            remote_rank
        )

        local_ranks.append(
            local_rank
        )

        hybrid_ranks.append(
            hybrid_rank
        )

        details.append(
            {
                "id": case_id,
                "query": query,
                "gold_doc_id": gold,
                "remote_rank": (
                    remote_rank
                ),
                "local_rank": (
                    local_rank
                ),
                "hybrid_rank": (
                    hybrid_rank
                ),
                "remote_top20": (
                    remote_ids[:TOPK]
                ),
                "local_top20": (
                    local_ids[:TOPK]
                ),
                "hybrid_top20": (
                    hybrid_ids[:TOPK]
                ),
            }
        )

        def show_rank(rank):
            return (
                str(rank)
                if rank is not None
                else "-"
            )

        print(
            f"{case_id:<4} "
            f"gold={gold:<3} "
            f"remote={show_rank(remote_rank):<3} "
            f"local={show_rank(local_rank):<3} "
            f"hybrid={show_rank(hybrid_rank):<3} "
            f"[{cache_status}]"
        )

    total = len(cases)

    remote_metrics = compute_metrics(
        remote_ranks,
        total,
    )

    local_metrics = compute_metrics(
        local_ranks,
        total,
    )

    hybrid_metrics = compute_metrics(
        hybrid_ranks,
        total,
    )

    print("\n" + "=" * 85)

    print_metrics(
        "REMOTE",
        remote_metrics,
    )

    print_metrics(
        "LOCAL",
        local_metrics,
    )

    print_metrics(
        "HYBRID",
        hybrid_metrics,
    )

    # -----------------------------------------------------
    # Complementarity analysis
    # -----------------------------------------------------

    remote_only = 0
    local_only = 0
    both = 0
    neither = 0

    for remote_rank, local_rank in zip(
        remote_ranks,
        local_ranks,
    ):
        remote_hit = (
            remote_rank is not None
            and remote_rank <= 20
        )

        local_hit = (
            local_rank is not None
            and local_rank <= 20
        )

        if remote_hit and local_hit:
            both += 1

        elif remote_hit:
            remote_only += 1

        elif local_hit:
            local_only += 1

        else:
            neither += 1

    print("\nComplementarity @20")
    print("-" * 55)

    print(
        f"Found by both        : {both}"
    )

    print(
        f"Remote only          : {remote_only}"
    )

    print(
        f"Local only           : {local_only}"
    )

    print(
        f"Found by neither     : {neither}"
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
                "evaluation": (
                    "synthetic_non_exact_"
                    "paraphrase_stress_test"
                ),
                "exact_matches": 0,
                "topk": TOPK,
                "rrf_k": RRF_K,
                "remote": (
                    remote_metrics
                ),
                "local": (
                    local_metrics
                ),
                "hybrid": (
                    hybrid_metrics
                ),
                "complementarity_at_20": {
                    "both": both,
                    "remote_only": (
                        remote_only
                    ),
                    "local_only": (
                        local_only
                    ),
                    "neither": neither,
                },
                "details": details,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )

    print(
        f"Remote cache: "
        f"{REMOTE_CACHE_PATH}"
    )


if __name__ == "__main__":
    main()