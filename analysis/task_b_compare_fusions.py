# analysis/task_b_compare_fusions.py

from server._hidden_retriever import remote_retrieve

from analysis.task_b_rrf_ablation import (
    add_candidate_ids,
    reciprocal_rank_fusion,
)

from analysis.task_b_borda_ablation import (
    borda_fusion,
)


QUERY = "رتبه C3 یعنی چی؟"
TOPK = 20


hits = add_candidate_ids(
    remote_retrieve(QUERY, topk=TOPK)
)

rrf = reciprocal_rank_fusion(hits)

borda = borda_fusion(
    hits,
    bm25_weight=0.5,
)


print("\nRRF top 10")
print("-" * 40)

for i, hit in enumerate(rrf[:10], start=1):
    print(
        i,
        hit["doc_id"],
        hit["bm25_rank"],
        hit["dense_rank"],
        round(hit["rrf_score"], 6),
    )


print("\nBorda top 10")
print("-" * 40)

for i, hit in enumerate(borda[:10], start=1):
    print(
        i,
        hit["doc_id"],
        hit["bm25_rank"],
        hit["dense_rank"],
        round(hit["borda_score"], 6),
    )


print("\nSame doc ordering?")
print(
    [h["doc_id"] for h in rrf[:10]]
    ==
    [h["doc_id"] for h in borda[:10]]
)