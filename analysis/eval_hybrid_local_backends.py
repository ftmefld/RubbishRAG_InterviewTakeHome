import json
from pathlib import Path

import rubbish_rag as R
import rubbish_rag.pipeline  # noqa: F401

from analysis.eval_hybrid_full_pipeline import (
    load_remote_hits_from_traces,
)

from analysis.eval_local_idf_retrieval import (
    build_statistics,
    load_corpus,
    rank_query,
)

from rubbish_rag.pipeline import (
    _hybrid_union,
    _load_local_question_index,
    _rrf_fusion,
    _should_use_rrf,
)


EVAL_PATH = Path(
    "analysis/paraphrase_eval.json"
)

SOURCE_TOPK = 20

BACKENDS = (
    "COUNT",
    "TFIDF",
    "BM25",
)


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def rank_of(
    hits,
    gold_doc_id,
):
    target = str(
        gold_doc_id
    )

    for rank, hit in enumerate(
        hits,
        start=1,
    ):
        if str(
            hit.get("doc_id")
        ) == target:
            return rank

    return None


def compute_metrics(
    ranks,
):
    total = len(
        ranks
    )

    def hit_at(k):
        if not total:
            return 0.0

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
        if total
        else 0.0
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
    metrics,
):
    print(
        f"{name:<12} "
        f"H1={metrics['H1']*100:5.1f}% "
        f"H3={metrics['H3']*100:5.1f}% "
        f"H5={metrics['H5']*100:5.1f}% "
        f"H20={metrics['H20']*100:5.1f}% "
        f"MRR={metrics['MRR']:.3f}"
    )


# ---------------------------------------------------------------------
# Build local hits in the same schema expected by _hybrid_union()
# ---------------------------------------------------------------------

def build_trusted_doc_map():
    """
    Reuse the canonical local corpus representation already used
    by the production pipeline.

    This gives us canonical answer text for local candidates while
    retrieval itself still uses Question only.
    """

    return {
        str(doc["doc_id"]): doc
        for doc in _load_local_question_index()
    }


def build_local_hits(
    ranking,
    score_key,
    trusted_docs,
    topk=20,
):
    """
    Convert a local Question-only ranking into the candidate schema
    expected by the Hybrid pipeline.

    Important:
    - retrieval ranking uses Question only
    - Answer/BriefAnswer is attached only AFTER document selection
    - no answer leakage is used to choose documents
    """

    hits = []

    for item in ranking[:topk]:

        doc_id = item[
            "doc_id"
        ]

        trusted = (
            trusted_docs.get(
                str(doc_id),
                {},
            )
        )

        score = float(
            item.get(
                score_key,
                0.0,
            )
        )

        hits.append(
            {
                "doc_id": doc_id,

                # Canonical corpus content.
                # This is not used to retrieve the document.
                "text": (
                    trusted.get("text")
                    or ""
                ),

                "question": (
                    trusted.get("question")
                    or item.get(
                        "question",
                        "",
                    )
                ),

                "category": (
                    trusted.get("category")
                    or ""
                ),

                # Hybrid metadata
                "local_score": score,
                "retrieval_source": "local",

                # Keep a compatible hit schema.
                "bm25": 0.0,
                "dense": 0.0,
                "fused": score,
            }
        )

    return hits


# ---------------------------------------------------------------------
# Evaluate one backend
# ---------------------------------------------------------------------

def evaluate_backend(
    backend_name,
    cases,
    remote_cache,
    corpus,
    stats,
    trusted_docs,
):

    if backend_name == "COUNT":
        ranking_key = "count_score"

    elif backend_name == "TFIDF":
        ranking_key = "tfidf_score"

    elif backend_name == "BM25":
        ranking_key = "bm25_score"

    else:
        raise ValueError(
            f"Unknown backend: "
            f"{backend_name}"
        )

    reranker = R.get(
        "reranker"
    )

    resolver = R.get(
        "resolver"
    )

    candidate_ranks = []
    reranked_ranks = []

    candidate_found = 0
    gold_cited = 0

    union_sizes = []

    source_counts = {
        "remote": 0,
        "local": 0,
        "remote+local": 0,
        "missing": 0,
    }

    details = []

    print(
        "\n"
        + "=" * 110
    )

    print(
        f"HYBRID + {backend_name}"
    )

    print(
        "-" * 110
    )

    for case in cases:

        case_id = case[
            "id"
        ]

        query = case[
            "query"
        ]

        gold = case[
            "gold_doc_id"
        ]

        # -------------------------------------------------------------
        # Remote branch — replayed from saved traces
        # -------------------------------------------------------------

        remote_hits = [
            dict(hit)
            for hit in remote_cache[
                query
            ]
        ]

        if (
            remote_hits
            and _should_use_rrf(
                remote_hits
            )
        ):
            remote_hits = (
                _rrf_fusion(
                    remote_hits
                )
            )

        # -------------------------------------------------------------
        # Local branch — COUNT / TF-IDF / BM25
        # -------------------------------------------------------------

        local_rankings = (
            rank_query(
                query,
                corpus,
                stats,
            )
        )

        local_hits = (
            build_local_hits(
                local_rankings[
                    ranking_key
                ],
                ranking_key,
                trusted_docs,
                topk=SOURCE_TOPK,
            )
        )

        # -------------------------------------------------------------
        # Full Remote + Local deduplicated union
        # -------------------------------------------------------------

        candidates = (
            _hybrid_union(
                remote_hits,
                local_hits,
            )
        )

        union_sizes.append(
            len(candidates)
        )

        candidate_rank = rank_of(
            candidates,
            gold,
        )

        candidate_ranks.append(
            candidate_rank
        )

        if candidate_rank is not None:
            candidate_found += 1

        # -------------------------------------------------------------
        # Current production reranker
        # -------------------------------------------------------------

        ranked = reranker(
            query,
            [
                dict(hit)
                for hit in candidates
            ],
        )

        reranked_rank = rank_of(
            ranked,
            gold,
        )

        reranked_ranks.append(
            reranked_rank
        )

        # -------------------------------------------------------------
        # Resolver
        # -------------------------------------------------------------

        final_hits = (
            ranked[:5]
        )

        result = resolver(
            query,
            final_hits,
        )

        cites = (
            result.get("cites")
            or []
        )

        cited = any(
            str(doc_id)
            == str(gold)
            for doc_id in cites
        )

        if cited:
            gold_cited += 1

        # -------------------------------------------------------------
        # Find which retrieval branch supplied the gold
        # -------------------------------------------------------------

        gold_source = None

        for hit in candidates:

            if str(
                hit.get("doc_id")
            ) != str(gold):
                continue

            gold_source = (
                hit.get(
                    "retrieval_source"
                )
            )

            break

        if gold_source is None:
            source_counts[
                "missing"
            ] += 1

        elif gold_source in source_counts:
            source_counts[
                gold_source
            ] += 1

        details.append(
            {
                "id": case_id,
                "gold": gold,
                "candidate_rank": (
                    candidate_rank
                ),
                "reranked_rank": (
                    reranked_rank
                ),
                "source": (
                    gold_source
                ),
                "gold_cited": (
                    cited
                ),
            }
        )

        print(
            f"{case_id:<4} "
            f"gold={gold:<4} "
            f"candidate="
            f"{str(candidate_rank):<4} "
            f"reranked="
            f"{str(reranked_rank):<4} "
            f"source="
            f"{str(gold_source):<13} "
            f"cited={cited}"
        )

    total = len(
        cases
    )

    candidate_recall = (
        candidate_found
        / total
    )

    candidate_metrics = (
        compute_metrics(
            candidate_ranks
        )
    )

    reranked_metrics = (
        compute_metrics(
            reranked_ranks
        )
    )

    citation_rate = (
        gold_cited
        / total
    )

    print(
        "\nCandidate pool coverage:"
    )

    print(
        f"  {candidate_found}/{total} "
        f"= "
        f"{candidate_recall*100:.1f}%"
    )

    print(
        "\nCandidate ordering:"
    )

    print_metrics(
        "CANDIDATE",
        candidate_metrics,
    )

    print(
        "\nAfter reranker:"
    )

    print_metrics(
        "RERANKED",
        reranked_metrics,
    )

    print(
        "\nGold cited:"
    )

    print(
        f"  {gold_cited}/{total} "
        f"= "
        f"{citation_rate*100:.1f}%"
    )

    print(
        "\nGold source:"
    )

    print(
        f"  remote only    : "
        f"{source_counts['remote']}"
    )

    print(
        f"  local only     : "
        f"{source_counts['local']}"
    )

    print(
        f"  remote + local : "
        f"{source_counts['remote+local']}"
    )

    print(
        f"  missing        : "
        f"{source_counts['missing']}"
    )

    return {
        "backend": backend_name,

        "candidate_recall": (
            candidate_recall
        ),

        "candidate_metrics": (
            candidate_metrics
        ),

        "reranked_metrics": (
            reranked_metrics
        ),

        "gold_citation_rate": (
            citation_rate
        ),

        "gold_cited": (
            gold_cited
        ),

        "source_counts": (
            source_counts
        ),

        "union_mean": (
            sum(union_sizes)
            / len(union_sizes)
        ),

        "details": details,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    with EVAL_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        cases = json.load(f)

    queries = [
        case["query"]
        for case in cases
    ]

    # -------------------------------------------------------------
    # Reuse saved Remote Top-20 results
    # -------------------------------------------------------------

    remote_cache = (
        load_remote_hits_from_traces(
            queries
        )
    )

    # -------------------------------------------------------------
    # Local corpus + IDF/BM25 statistics
    # -------------------------------------------------------------

    corpus = load_corpus()

    stats = build_statistics(
        corpus
    )

    trusted_docs = (
        build_trusted_doc_map()
    )

    print(
        "\nHybrid Local-Backend Ablation"
    )

    print(
        "=" * 110
    )

    print(
        f"Queries: {len(cases)}"
    )

    print(
        f"Corpus docs: {len(corpus)}"
    )

    print(
        f"Remote candidates/query: "
        f"{SOURCE_TOPK}"
    )

    print(
        f"Local candidates/query: "
        f"{SOURCE_TOPK}"
    )

    print(
        "Remote source: "
        "saved trace replay"
    )

    print(
        "New remote calls: 0"
    )

    print(
        "Local retrieval field: "
        "Question only"
    )

    print(
        "Answer used for retrieval: NO"
    )

    results = {}

    for backend in BACKENDS:

        results[
            backend
        ] = evaluate_backend(
            backend,
            cases,
            remote_cache,
            corpus,
            stats,
            trusted_docs,
        )

    # -------------------------------------------------------------
    # Final comparison
    # -------------------------------------------------------------

    print(
        "\n"
        + "=" * 110
    )

    print(
        "FINAL HYBRID COMPARISON"
    )

    print(
        "-" * 110
    )

    print(
        "BACKEND   "
        "POOL-RECALL   "
        "H@1    "
        "H@3    "
        "H@5    "
        "H@20   "
        "MRR     "
        "CITED"
    )

    for backend in BACKENDS:

        result = results[
            backend
        ]

        metrics = result[
            "reranked_metrics"
        ]

        print(
            f"{backend:<9}"
            f"{result['candidate_recall']*100:6.1f}%      "
            f"{metrics['H1']*100:5.1f}% "
            f"{metrics['H3']*100:5.1f}% "
            f"{metrics['H5']*100:5.1f}% "
            f"{metrics['H20']*100:5.1f}% "
            f"{metrics['MRR']:.3f}   "
            f"{result['gold_citation_rate']*100:5.1f}%"
        )

    # -------------------------------------------------------------
    # Direct pairwise gold-rank comparison
    # -------------------------------------------------------------

    print(
        "\n"
        + "=" * 110
    )

    print(
        "PER-QUERY FINAL RANKS"
    )

    print(
        "-" * 110
    )

    print(
        "ID    GOLD   COUNT   TFIDF   BM25"
    )

    detail_maps = {}

    for backend in BACKENDS:

        detail_maps[
            backend
        ] = {
            row["id"]: row
            for row in results[
                backend
            ][
                "details"
            ]
        }

    for case in cases:

        case_id = case[
            "id"
        ]

        gold = case[
            "gold_doc_id"
        ]

        row = []

        for backend in BACKENDS:

            rank = detail_maps[
                backend
            ][
                case_id
            ][
                "reranked_rank"
            ]

            row.append(
                "-" if rank is None
                else str(rank)
            )

        print(
            f"{case_id:<5} "
            f"{gold:<6} "
            f"{row[0]:<7} "
            f"{row[1]:<7} "
            f"{row[2]:<7}"
        )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    output_path = Path(
        "analysis/"
        "hybrid_local_backend_ablation.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nSaved: {output_path}"
    )

    print(
        "New remote retrieval calls: 0"
    )


if __name__ == "__main__":
    main()