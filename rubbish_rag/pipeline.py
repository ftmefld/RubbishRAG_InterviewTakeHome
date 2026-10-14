"""RubbishRAG pipeline — the 1-minute version.

Stages (override any of them with @rubbish_rag.* decorators —
see docs/DECORATORS.md for the mechanism):
  chunk -> retrieve (remote) -> rerank -> resolve
"""
import rubbish_rag as R
from .normalize_fa import normalize_fa
from .slots import extract_slots
import re
import csv
from functools import lru_cache
from pathlib import Path
import math
from collections import Counter, defaultdict



MAX_CHUNK_CHARS = 360

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!؟!?؛;])\s+|\n+"
)

_CLAUSE_SPLIT_RE = re.compile(
    r"(?<=[،,:])\s+"
)

# Temporary marker used so numeric expressions behave as one unit
# during fallback word splitting.
_ATOMIC_MARKER = "\uFFF0"

_NUMERIC_PHRASE_RE = re.compile(
    r"(?<!\w)"
    r"[0-9۰-۹٠-٩][0-9۰-۹٠-٩,./]*"
    r"(?:\s*(?:تا|الی|-)\s*"
    r"[0-9۰-۹٠-٩][0-9۰-۹٠-٩,./]*)?"
    r"(?:\s+(?:میلیون|میلیارد))?"
    r"(?:\s+(?:ریال|تومان|درصد|٪|امتیاز|روز|ماه|سال|ساعت))?"
    r"(?!\w)"
)


def _clean_text(value):
    if not value:
        return ""

    return re.sub(
        r"[ \t]+",
        " ",
        str(value),
    ).strip()


def _metadata_prefix(doc):
    """
    Add retrieval-visible structural context to every chunk.

    Full metadata is also retained separately in the chunk dict.
    """
    category = normalize_fa(
        _clean_text(doc.get("Category", ""))
    )

    keyword = normalize_fa(
        _clean_text(doc.get("Keyword", ""))
    )

    parts = [
        value
        for value in (category, keyword)
        if value
    ]

    if not parts:
        return ""

    prefix = "[" + " | ".join(parts) + "]"

    # Metadata fields themselves are preserved separately.
    # This only prevents an unusually large prefix from consuming
    # the entire chunk budget.
    if len(prefix) > 120:
        prefix = prefix[:118].rstrip() + "…]"

    return prefix


def _protect_numeric_phrases(text):
    """
    Prevent fallback word splitting from separating expressions like:

        250 میلیون تومان
        520 تا 539 امتیاز
        5 سال
    """

    def replace(match):
        return match.group(0).replace(
            " ",
            _ATOMIC_MARKER,
        )

    return _NUMERIC_PHRASE_RE.sub(
        replace,
        text,
    )


def _restore_numeric_phrases(text):
    return text.replace(
        _ATOMIC_MARKER,
        " ",
    )


def _split_by_words(text, max_chars):
    """
    Last-resort splitter for a sentence/clause that is itself too long.

    It splits on words rather than arbitrary character positions and
    preserves common numeric expressions as atomic spans.
    """
    protected = _protect_numeric_phrases(
        text
    )

    words = protected.split()

    pieces = []
    current = []

    for word in words:
        candidate = " ".join(
            current + [word]
        )

        candidate_visible = (
            _restore_numeric_phrases(candidate)
        )

        if (
            not current
            or len(candidate_visible) <= max_chars
        ):
            current.append(word)
            continue

        pieces.append(
            _restore_numeric_phrases(
                " ".join(current)
            ).strip()
        )

        visible_word = (
            _restore_numeric_phrases(word)
        )

        if len(visible_word) <= max_chars:
            current = [word]

        else:
            # Pathological no-space token.
            # There is no semantic boundary available here.
            current = []

            for i in range(
                0,
                len(visible_word),
                max_chars,
            ):
                pieces.append(
                    visible_word[
                        i:i + max_chars
                    ]
                )

    if current:
        pieces.append(
            _restore_numeric_phrases(
                " ".join(current)
            ).strip()
        )

    return [
        piece
        for piece in pieces
        if piece
    ]


def _split_long_sentence(text, max_chars):
    """
    Prefer clause boundaries before falling back to word boundaries.
    """
    clauses = [
        part.strip()
        for part in _CLAUSE_SPLIT_RE.split(text)
        if part.strip()
    ]

    if len(clauses) <= 1:
        return _split_by_words(
            text,
            max_chars,
        )

    pieces = []
    current = ""

    for clause in clauses:
        if len(clause) > max_chars:
            if current:
                pieces.append(current)
                current = ""

            pieces.extend(
                _split_by_words(
                    clause,
                    max_chars,
                )
            )

            continue

        candidate = (
            clause
            if not current
            else f"{current} {clause}"
        )

        if len(candidate) <= max_chars:
            current = candidate

        else:
            if current:
                pieces.append(current)

            current = clause

    if current:
        pieces.append(current)

    return pieces


def _body_chunks(text, max_chars):
    """
    Sentence-aware greedy packing.

    Natural sentence boundaries are preserved whenever possible.
    """
    text = _clean_text(text)

    if not text:
        return []

    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT_RE.split(
            text
        )
        if sentence.strip()
    ]

    units = []

    for sentence in sentences:
        if len(sentence) <= max_chars:
            units.append(sentence)
        else:
            units.extend(
                _split_long_sentence(
                    sentence,
                    max_chars,
                )
            )

    chunks = []
    current = ""

    for unit in units:
        candidate = (
            unit
            if not current
            else f"{current} {unit}"
        )

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        current = unit

    if current:
        chunks.append(current)

    return chunks


@R.chunker
def chunk(docs):
    """
    Boundary-aware chunking with retrieval-visible metadata.

    Strategy:
      1. Prefer Answer, then BriefAnswer.
      2. Prefix every chunk with [Category | Keyword].
      3. Split at sentence boundaries first.
      4. Fall back to clause/word boundaries for oversized sentences.
      5. Keep common numeric expressions with their units.
      6. Bound final chunk length.
    """
    chunks = []

    for doc in docs:
        source_text = (
            doc.get("Answer")
            or doc.get("BriefAnswer")
            or ""
        )

        source_text = _clean_text(
            source_text
        )

        if not source_text:
            continue

        prefix = _metadata_prefix(doc)

        separator_len = 1 if prefix else 0

        body_budget = (
            MAX_CHUNK_CHARS
            - len(prefix)
            - separator_len
        )

        if body_budget <= 0:
            # Extremely defensive fallback.
            body_budget = MAX_CHUNK_CHARS

            prefix = ""

        bodies = _body_chunks(
            source_text,
            body_budget,
        )

        for index, body in enumerate(bodies):
            chunk_text = (
                f"{prefix} {body}"
                if prefix
                else body
            )

            chunks.append(
                {
                    "doc_id": doc["doc_id"],
                    "category": doc.get(
                        "Category",
                        "",
                    ),
                    "keyword": doc.get(
                        "Keyword",
                        "",
                    ),
                    "question": doc.get(
                        "Question",
                        "",
                    ),
                    "text": chunk_text,
                    "chunk_id": (
                        f"{doc['doc_id']}:{index}"
                    ),
                }
            )

    return chunks

def _quantile(values, q):
    values = sorted(values)

    if not values:
        return 0.0

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    fraction = position - lower

    return (
        values[lower] * (1.0 - fraction)
        + values[upper] * fraction
    )


def _has_high_outlier(values):
    """
    Detect an upper score outlier with Tukey's standard 1.5*IQR rule.
    """
    if len(values) < 4:
        return False

    q1 = _quantile(values, 0.25)
    q3 = _quantile(values, 0.75)

    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr

    return max(values) > upper_fence


def _rank_map(hits, score_key):
    ranked_ids = sorted(
        range(len(hits)),
        key=lambda i: hits[i].get(
            score_key,
            float("-inf"),
        ),
        reverse=True,
    )

    return {
        candidate_id: rank
        for rank, candidate_id in enumerate(
            ranked_ids,
            start=1,
        )
    }


def _rrf_fusion(hits, k=60):
    """
    Equal-weight reciprocal rank fusion over BM25 and dense rankings.
    """
    bm25_ranks = _rank_map(
        hits,
        "bm25",
    )

    dense_ranks = _rank_map(
        hits,
        "dense",
    )

    rescored = []

    for candidate_id, hit in enumerate(hits):
        item = dict(hit)

        item["rrf_score"] = (
            1.0 / (k + bm25_ranks[candidate_id])
            + 1.0 / (k + dense_ranks[candidate_id])
        )

        rescored.append(item)

    return sorted(
        rescored,
        key=lambda hit: (
            hit["rrf_score"],
            hit.get("fused", 0.0),
        ),
        reverse=True,
    )


def _should_use_rrf(hits):
    if not hits:
        return False

    bm25_scores = [
        hit.get("bm25", 0.0)
        for hit in hits
    ]

    dense_scores = [
        hit.get("dense", 0.0)
        for hit in hits
    ]

    has_outlier = (
        _has_high_outlier(bm25_scores)
        or _has_high_outlier(dense_scores)
    )

    bm25_top = max(
        range(len(hits)),
        key=lambda i: hits[i].get(
            "bm25",
            float("-inf"),
        ),
    )

    dense_top = max(
        range(len(hits)),
        key=lambda i: hits[i].get(
            "dense",
            float("-inf"),
        ),
    )

    top1_disagreement = (
        bm25_top != dense_top
    )

    return (
        has_outlier
        and top1_disagreement
    )


# ------------------------------------------------------------------
# Hybrid retrieval: local FAQ search + remote retrieval
# ------------------------------------------------------------------

_LOCAL_CORPUS_PATH = (
    Path(__file__).resolve().parents[1]
    / "corpus"
    / "test.csv"
)


import re


def _token_counter(text):
    """
    Normalize Persian text and extract lexical tokens.

    Punctuation is treated as a boundary so forms such as
    'صادرات/واردات' become two searchable tokens instead of
    one glued token.
    """
    normalized = normalize_fa(
        text or ""
    )

    tokens = re.findall(
        r"[A-Za-z0-9\u0600-\u06FF]+",
        normalized,
    )

    return Counter(tokens)


def _cosine_similarity(counter_a, counter_b):
    """
    Cosine similarity between two sparse token-frequency vectors.
    """
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

    return (
        numerator
        / (norm_a * norm_b)
    )


@lru_cache(maxsize=1)
def _load_local_question_index():
    """
    Build the local FAQ index once per process.

    IMPORTANT:
    Retrieval similarity uses Question only.
    Answer/BriefAnswer is attached only after a document has
    been retrieved, so answer content is not used to select itself.
    """
    index = []

    with _LOCAL_CORPUS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    for row_index, row in enumerate(rows):
        explicit_doc_id = row.get(
            "doc_id"
        )

        if (
            explicit_doc_id is not None
            and str(explicit_doc_id).strip()
        ):
            try:
                doc_id = int(
                    explicit_doc_id
                )
            except ValueError:
                doc_id = (
                    explicit_doc_id
                )
        else:
            # corpus/test.csv uses row position as
            # the document ID in this dataset.
            doc_id = row_index

        question = (
            row.get("Question")
            or ""
        )

        answer = (
            row.get("BriefAnswer")
            or row.get("Answer")
            or ""
        )

        index.append(
            {
                "doc_id": doc_id,
                "question": question,
                "question_tokens": (
                    _token_counter(
                        question
                    )
                ),
                "text": answer,
                "category": (
                    row.get("Category")
                    or ""
                ),
            }
        )

    return index

BM25_K1 = 1.2
BM25_B = 0.75


def _local_tokens(text):
    """
    Persian normalization + punctuation-aware tokenization.
    """
    normalized = normalize_fa(text or "")

    return re.findall(
        r"[A-Za-z0-9\u0600-\u06FF]+",
        normalized,
    )


@lru_cache(maxsize=1)
def _load_local_bm25_index():
    """
    Build a Question-only BM25 index over the trusted corpus.

    Answer/BriefAnswer are stored only so evidence can be attached
    AFTER retrieval. They are never used to score document relevance.
    """

    docs = []

    with _LOCAL_CORPUS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row_index, row in enumerate(reader):

            raw_doc_id = row.get("doc_id")

            if raw_doc_id:
                try:
                    doc_id = int(raw_doc_id)
                except ValueError:
                    doc_id = raw_doc_id
            else:
                doc_id = row_index

            question = row.get("Question") or ""

            tokens = _local_tokens(
                question
            )

            docs.append(
                {
                    "doc_id": doc_id,
                    "question": question,
                    "tokens": tokens,
                    "tf": Counter(tokens),
                    "length": len(tokens),

                    # Evidence only — NOT retrieval fields.
                    "text": (
                        row.get("BriefAnswer")
                        or row.get("Answer")
                        or ""
                    ),
                    "category": (
                        row.get("Category")
                        or ""
                    ),
                }
            )

    df = defaultdict(int)

    for doc in docs:
        for token in set(doc["tokens"]):
            df[token] += 1

    avg_length = (
        sum(doc["length"] for doc in docs)
        / len(docs)
        if docs
        else 0.0
    )

    return {
        "docs": docs,
        "df": dict(df),
        "n_docs": len(docs),
        "avg_length": avg_length,
    }


def _bm25_idf(token, index):
    n_docs = index["n_docs"]
    df = index["df"].get(token, 0)

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


def _bm25_question_score(
    query_tokens,
    doc,
    index,
):
    if not query_tokens:
        return 0.0

    score = 0.0

    avgdl = index["avg_length"]
    dl = doc["length"]
    tf = doc["tf"]

    for token in set(query_tokens):

        freq = tf.get(
            token,
            0,
        )

        if freq == 0:
            continue

        idf = _bm25_idf(
            token,
            index,
        )

        denominator = (
            freq
            + BM25_K1
            * (
                1.0
                - BM25_B
                + BM25_B
                * dl
                / avgdl
            )
        )

        score += (
            idf
            * freq
            * (BM25_K1 + 1.0)
            / denominator
        )

    return score

def _local_question_retrieve(
    query,
    topk=20,
):
    """
    Retrieve against normalized corpus Questions using BM25.

    Important:
    - relevance is computed from Question only
    - Answer/BriefAnswer never participate in retrieval
    - canonical answer text is attached after ranking
    """

    index = _load_local_bm25_index()

    query_tokens = _local_tokens(
        query
    )

    scored = []

    for doc in index["docs"]:

        score = _bm25_question_score(
            query_tokens,
            doc,
            index,
        )

        if score <= 0:
            continue

        scored.append(
            {
                "doc_id": doc["doc_id"],
                "text": doc["text"],
                "question": doc["question"],
                "category": doc["category"],

                "local_score": score,
                "retrieval_source": "local",

                # compatible candidate schema
                "bm25": 0.0,
                "dense": 0.0,
                "fused": score,
            }
        )

    scored.sort(
        key=lambda hit: (
            hit["local_score"],
            -(
                hit["doc_id"]
                if isinstance(
                    hit["doc_id"],
                    int,
                )
                else 0
            ),
        ),
        reverse=True,
    )

    return scored[:topk]


def _trusted_local_doc_map():
    """
    Map corpus document IDs to authoritative local corpus content.

    Retrieval scores may come from Remote or Local retrieval,
    but final evidence text should come from the trusted corpus
    whenever that document exists locally.
    """
    return {
        str(row["doc_id"]): row
        for row in _load_local_question_index()
    }

def _hybrid_union(
    remote_hits,
    local_hits,
):
    """
    Merge Remote and Local candidates by document ID.

    Retrieval signals are preserved from their original sources,
    while evidence text is re-hydrated from the trusted local
    corpus whenever the document exists there.

    Unknown/fabricated documents keep their original retrieved
    text and remain subject to Track F grounding checks.
    """

    candidates = {}

    # ----------------------------------------------------------
    # Remote candidates
    # ----------------------------------------------------------

    for rank, hit in enumerate(
        remote_hits,
        start=1,
    ):
        item = dict(hit)

        doc_id = item.get(
            "doc_id"
        )

        key = str(
            doc_id
        )

        item["remote_rank"] = rank
        item["local_rank"] = None

        item["retrieval_source"] = (
            "remote"
        )

        item["local_score"] = 0.0

        candidates[key] = item

    # ----------------------------------------------------------
    # Local candidates
    # ----------------------------------------------------------

    for rank, hit in enumerate(
        local_hits,
        start=1,
    ):
        item = dict(hit)

        doc_id = item.get(
            "doc_id"
        )

        key = str(
            doc_id
        )

        if key in candidates:
            existing = candidates[
                key
            ]

            existing[
                "local_rank"
            ] = rank

            existing[
                "local_score"
            ] = item.get(
                "local_score",
                0.0,
            )

            existing[
                "retrieval_source"
            ] = "remote+local"

        else:
            item[
                "remote_rank"
            ] = None

            item[
                "local_rank"
            ] = rank

            candidates[
                key
            ] = item

    # ----------------------------------------------------------
    # Re-hydrate evidence from trusted corpus
    # ----------------------------------------------------------

    trusted_docs = (
        _trusted_local_doc_map()
    )

    for key, item in (
        candidates.items()
    ):
        trusted = trusted_docs.get(
            key
        )

        if trusted is not None:
            # Preserve retrieved snippet for diagnostics.
            item[
                "retrieved_text"
            ] = item.get(
                "text",
                "",
            )

            # Replace evidence with canonical trusted content.
            item["text"] = (
                trusted.get("text")
                or ""
            )

            item[
                "canonical_question"
            ] = (
                trusted.get(
                    "question"
                )
                or ""
            )

            item["category"] = (
                trusted.get(
                    "category"
                )
                or item.get(
                    "category",
                    "",
                )
            )

            item[
                "evidence_source"
            ] = "trusted_local_corpus"

        else:
            # Unknown documents are NOT automatically trusted.
            item[
                "evidence_source"
            ] = "retrieved_only"

    # ----------------------------------------------------------
    # Best-source retrieval prior
    # ----------------------------------------------------------

    merged = []

    for item in candidates.values():

        available_ranks = [
            rank
            for rank in (
                item.get(
                    "remote_rank"
                ),
                item.get(
                    "local_rank"
                ),
            )
            if rank is not None
        ]

        best_rank = min(
            available_ranks
        )

        item[
            "best_source_rank"
        ] = best_rank

        item[
            "retrieval_prior"
        ] = (
            1.0
            / (60 + best_rank)
        )

        merged.append(
            item
        )

    # ----------------------------------------------------------
    # Candidate ordering before reranking
    # ----------------------------------------------------------

    merged.sort(
        key=lambda hit: (
            hit[
                "best_source_rank"
            ],
            -hit.get(
                "local_score",
                0.0,
            ),
            -hit.get(
                "fused",
                0.0,
            ),
        )
    )

    return merged


@R.retriever
def retrieve(query, topk=5):
    """
    Hybrid candidate retrieval.

    1. Retrieve a broad Remote candidate set.
    2. Preserve the existing gated BM25/dense RRF correction
       inside the Remote branch.
    3. Retrieve candidates independently from local corpus
       Questions.
    4. Union both candidate sets by document ID.
    5. Rank conservatively by the best source rank.

    The downstream reranker remains responsible for final
    relevance, grounding, polarity and diversity decisions.
    """
    from server._hidden_retriever import (
        remote_retrieve,
    )

    source_topk = max(
        20,
        topk,
    )

    # --------------------------------------------------------------
    # Remote branch
    # --------------------------------------------------------------

    remote_hits = remote_retrieve(
        query,
        topk=source_topk,
    )

    remote_hits = [
        dict(hit)
        for hit in remote_hits
    ]

    # Keep our previous Task B result:
    # only apply BM25/dense RRF when the labels-free gate fires.
    if (
        remote_hits
        and _should_use_rrf(
            remote_hits
        )
    ):
        remote_hits = _rrf_fusion(
            remote_hits
        )

    # --------------------------------------------------------------
    # Local Question branch
    # --------------------------------------------------------------

    local_hits = (
        _local_question_retrieve(
            query,
            topk=source_topk,
        )
    )

    # --------------------------------------------------------------
    # Candidate union
    # --------------------------------------------------------------

    hybrid_hits = _hybrid_union(
        remote_hits,
        local_hits,
    )

    # Respect the retriever contract.
    #
    # For final candidate-pool experiments we will call:
    #
    #     retrieve(query, topk=20)
    #
    # so the reranker receives a broad pool.
    return hybrid_hits[:topk]

@lru_cache(maxsize=1)
def _trusted_corpus_texts():
    """
    Load and normalize trusted Answer / BriefAnswer text from the
    canonical corpus.

    This creates the allow-list from content, not document IDs.
    """
    corpus_path = (
        Path(__file__).resolve().parents[1]
        / "corpus"
        / "test.csv"
    )

    texts = []

    with corpus_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            for field in (
                "Answer",
                "BriefAnswer",
            ):
                value = normalize_fa(
                    row.get(field, "")
                ).strip()

                if value:
                    texts.append(value)

    return tuple(texts)


def _is_grounded_candidate(hit):
    """
    Verify that a retrieved candidate originates from trusted corpus
    content.

    Matching is performed after Persian normalization so digit-script,
    Arabic/Persian character, and whitespace differences do not cause
    false rejections.
    """
    hit_text = normalize_fa(
        hit.get("text", "")
    ).strip()

    if not hit_text:
        return False

    for trusted_text in _trusted_corpus_texts():
        if (
            hit_text in trusted_text
            or trusted_text in hit_text
        ):
            return True

    return False


def _promote_document_diversity(ranked_hits):
    """
    Preserve the best-ranked chunk from each document in the main
    evidence region and move additional chunks from the same document
    behind unique-document evidence.

    Candidate count is preserved.
    """
    seen_docs = set()

    unique_hits = []
    duplicate_hits = []

    for hit in ranked_hits:
        item = dict(hit)

        doc_id = str(
            item.get("doc_id")
        )

        if doc_id in seen_docs:
            item["duplicate_document"] = True
            duplicate_hits.append(item)

        else:
            seen_docs.add(doc_id)

            item["duplicate_document"] = False
            unique_hits.append(item)

    return unique_hits + duplicate_hits

POLARITY_ANCHORS = (
    "سابقه",
    "تاخیر",
    "بدهی",
    "چک",
    "مالیات",
    "وام",
    "تسهیلات",
    "پرداخت",
    "ثبت",
    "محکومیت",
)


def _polarity_signature(text):
    """
    Extract simple concept-level polarity from Persian text.

    This intentionally ignores volitional negation such as
    'نمی خواهم', because it does not negate the factual state
    of the referenced credit concept.
    """
    normalized = normalize_fa(text)

    # Do not treat user preference as factual polarity.
    normalized = normalized.replace(
        "نمی خواهم",
        "می خواهم",
    )

    tokens = normalized.split()

    negative_markers = {
        "ندارد",
        "ندارند",
        "نداشتن",
        "نداشته",
        "نیست",
        "نبوده",
        "نشده",
        "نشد",
        "فاقد",
        "بدون",
        "عدم",
    }

    signature = {}

    for index, token in enumerate(tokens):
        for anchor in POLARITY_ANCHORS:
            if anchor not in token:
                continue

            start = max(
                0,
                index - 3,
            )

            end = min(
                len(tokens),
                index + 4,
            )

            window = tokens[start:end]

            negative = any(
                marker in window
                for marker in negative_markers
            )

            # Handle constructions such as:
            # "هیچ سابقه تاخیری ندارد"
            if (
                "هیچ" in window
                and any(
                    marker in window
                    for marker in (
                        "ندارد",
                        "ندارند",
                        "نیست",
                    )
                )
            ):
                negative = True

            signature[anchor] = (
                "negative"
                if negative
                else "positive"
            )

    return signature

def _polarity_conflicts(
    query,
    candidate_text,
):
    query_polarity = (
        _polarity_signature(query)
    )

    candidate_polarity = (
        _polarity_signature(
            candidate_text
        )
    )

    conflicts = []

    shared_concepts = (
        set(query_polarity)
        & set(candidate_polarity)
    )

    for concept in shared_concepts:
        if (
            query_polarity[concept]
            != candidate_polarity[concept]
        ):
            conflicts.append(concept)

    return conflicts

@R.reranker
def rerank(query, hits):
    """
    Candidate-side reranking with:

    - Task C: Persian normalization + lexical overlap
    - Track F: trusted-corpus grounding
    - Track G: polarity compatibility
    - Hybrid-aware retrieval evidence
    - Track E: document-level diversity

    The reranker always preserves candidate count.
    """

    normalized_query = normalize_fa(
        query
    )

    query_tokens = set(
        normalized_query.split()
    )

    rescored = []

    # =========================================================
    # IMPORTANT:
    # Only candidate feature extraction happens inside this loop.
    # There must NOT be any return inside the loop.
    # =========================================================

    for hit in hits:
        item = dict(hit)

        # -----------------------------------------------------
        # Track F — trusted-corpus grounding
        # -----------------------------------------------------

        item["verified_corpus"] = (
            _is_grounded_candidate(
                item
            )
        )

        # -----------------------------------------------------
        # Task C — normalized lexical matching
        # -----------------------------------------------------

        normalized_text = normalize_fa(
            item.get(
                "text",
                "",
            )
        )

        hit_tokens = set(
            normalized_text.split()
        )

        item["rerank_score"] = len(
            query_tokens
            & hit_tokens
        )

        # -----------------------------------------------------
        # Track G — polarity / negation compatibility
        # -----------------------------------------------------

        conflicts = (
            _polarity_conflicts(
                query,
                item.get(
                    "text",
                    "",
                ),
            )
        )

        item[
            "polarity_conflicts"
        ] = conflicts

        item[
            "polarity_compatible"
        ] = (
            len(conflicts) == 0
        )

        rescored.append(
            item
        )

    # =========================================================
    # IMPORTANT:
    # Everything below this point is OUTSIDE the for-loop.
    # =========================================================

    # Detect whether these candidates came through the
    # Remote + Local hybrid retriever.
    hybrid_mode = any(
        "best_source_rank" in item
        for item in rescored
    )

    # ---------------------------------------------------------
    # Hybrid mode
    # ---------------------------------------------------------

    if hybrid_mode:
        ranked = sorted(
            rescored,
            key=lambda item: (
                # Track F:
                # grounded / trusted evidence first
                item[
                    "verified_corpus"
                ],

                # Track G:
                # factual polarity compatibility
                item[
                    "polarity_compatible"
                ],

                # Preserve retrieval evidence from the
                # strongest source (Remote or Local).
                item.get(
                    "retrieval_prior",
                    0.0,
                ),

                # Local Question similarity is useful when
                # the Answer uses different wording.
                item.get(
                    "local_score",
                    0.0,
                ),

                # Task C answer-text lexical overlap is
                # secondary in Hybrid mode.
                item[
                    "rerank_score"
                ],

                # Final tie-breaker.
                item.get(
                    "fused",
                    0.0,
                ),
            ),
            reverse=True,
        )

    # ---------------------------------------------------------
    # Remote-only mode
    # ---------------------------------------------------------

    else:
        ranked = sorted(
            rescored,
            key=lambda item: (
                # Track F
                item[
                    "verified_corpus"
                ],

                # Track G
                item[
                    "polarity_compatible"
                ],

                # Task C
                item[
                    "rerank_score"
                ],

                # Preserve original retrieval signal.
                item.get(
                    "fused",
                    0.0,
                ),
            ),
            reverse=True,
        )

    # ---------------------------------------------------------
    # Track E — document-level diversity
    # ---------------------------------------------------------

    ranked = (
        _promote_document_diversity(
            ranked
        )
    )

    # =========================================================
    # This is the ONLY return in the reranker.
    # =========================================================

    return ranked

def _contains_any(text, phrases):
    return any(
        phrase in text
        for phrase in phrases
    )


def _is_generic_debt_intent(normalized_query):
    """
    Detect questions whose answer depends on the debt type.
    """
    debt_terms = (
        "بدهی",
        "بدهی ام",
        "بدهی من",
    )

    debt_actions = (
        "تاثیر",
        "اثر",
        "رتبه",
        "امتیاز",
        "حذف",
        "پاک",
        "گزارش",
        "سابقه",
        "رفع",
    )

    return (
        _contains_any(
            normalized_query,
            debt_terms,
        )
        and _contains_any(
            normalized_query,
            debt_actions,
        )
    )


def _is_loan_eligibility_intent(normalized_query):
    """
    Detect direct loan-eligibility requests.

    General questions about the effect of credit history on loans are
    not treated as requests for a bank-specific approval decision.
    """
    direct_eligibility_phrases = (
        "بهم وام میدن",
        "به من وام میدن",
        "بهم وام می دهند",
        "به من وام می دهند",

        "میتونم وام بگیرم",
        "می تونم وام بگیرم",
        "می توانم وام بگیرم",

        "میتونم تسهیلات بگیرم",
        "می تونم تسهیلات بگیرم",
        "می توانم تسهیلات بگیرم",

        "تسهیلات میدن",
        "تسهیلات می دهند",

        "برای دریافت وام قبولم میکنن",
        "برای دریافت وام قبولم می کنند",
        "برای دریافت تسهیلات قبولم میکنن",
        "برای دریافت تسهیلات قبولم می کنند",
    )

    return _contains_any(
        normalized_query,
        direct_eligibility_phrases,
    )

def _has_specific_debt_context(normalized_query, slots):
    """
    A missing DEBT slot does not necessarily mean the debt is
    underspecified. Some queries explicitly identify a debt source
    that is outside the predefined debt taxonomy.
    """
    if slots.get("DEBT"):
        return True

    additional_debt_contexts = (
        "تامین کننده",
        "تامین کنندگان",
        "پیمانکار",
        "پیمانکاران",
        "بدهی تجاری",
        "طلبکار",
        "فروشنده",
        "فروشندگان",
    )

    return _contains_any(
        normalized_query,
        additional_debt_contexts,
    )

def _clarification_for_query(query, slots):
    normalized_query = normalize_fa(query)

    # Generic debt question:
    # the answer depends on which debt category is meant.
    if (
        _is_generic_debt_intent(
            normalized_query
        )
        and not _has_specific_debt_context(
            normalized_query,
            slots,
        )
    ):
        return {
            "status": "clarify",
            "question": (
                "منظورتان کدام نوع بدهی است؟"
            ),
            "options": [
                "بدهی تسهیلاتی / وام",
                "بدهی مالیاتی",
                "چک برگشتی",
                "محکومیت مالی",
                "نمی دانم",
            ],
        }

    # Loan eligibility depends on bank policy.
    if (
        _is_loan_eligibility_intent(
            normalized_query
        )
        and not slots.get("BANK")
    ):
        return {
            "status": "clarify",
            "question": (
                "برای کدام بانک درباره دریافت "
                "وام یا تسهیلات سؤال دارید؟"
            ),
            "options": [
                "ملی",
                "صادرات",
                "تجارت",
                "ملت",
                "سپه",
                "پاسارگاد",
                "توسعه صادرات",
                "نمی دانم",
            ],
        }

    return None

@R.resolver
def resolve(query, hits):
    """
    Answer sufficiently specified queries and ask targeted
    clarification questions when a required domain slot is missing.
    """
    slots = extract_slots(query)

    clarification = _clarification_for_query(
        query,
        slots,
    )

    if clarification is not None:
        return clarification

    top = (
        hits[0]
        if hits
        else {
            "text": "",
            "doc_id": -1,
        }
    )

    return {
        "status": "answer",
        "brief": top["text"][:200],
        "cites": (
            [h["doc_id"] for h in hits[:3]]
            or [top["doc_id"]]
        ),
    }