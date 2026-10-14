"""Unit tests for the RubbishRAG pipeline stages and decorators."""
import pytest
import rubbish_rag as R
import rubbish_rag.pipeline
from rubbish_rag.pipeline import _polarity_conflicts

def test_decorator_registration():
    assert R.get("chunker") is not None
    assert R.get("retriever") is not None
    assert R.get("reranker") is not None
    assert R.get("resolver") is not None


def test_chunker_structure():
    chunker = R.get("chunker")
    mock_docs = [
        {
            "doc_id": 42,
            "Question": "سوال تستی؟",
            "Category": "مفاهیم",
            "BriefAnswer": "پاسخ کوتاه",
            "Answer": "پاسخ تفصیلی برای تست سیستم تقطیع چانک‌ها که باید پردازش شود.",
            "Keyword": "تست",
        }
    ]
    chunks = chunker(mock_docs)
    assert len(chunks) >= 1
    c = chunks[0]
    assert "doc_id" in c and c["doc_id"] == 42
    assert "text" in c
    assert "chunk_id" in c


def test_reranker_contract():
    reranker = R.get("reranker")
    mock_hits = [
        {"doc_id": 1, "text": "رتبه اعتباری و امتیاز", "fused": 0.8},
        {"doc_id": 2, "text": "سایر اطلاعات بی ارتباط", "fused": 0.5},
    ]
    reranked = reranker("رتبه اعتباری", mock_hits)
    assert len(reranked) == 2
    assert "rerank_score" in reranked[0]
    # First item has more overlap with query
    assert reranked[0]["doc_id"] == 1


def test_resolver_contract():
    resolver = R.get("resolver")
    mock_hits = [
        {"doc_id": 106, "text": "رتبه C1 در بازه ۵۶۰ تا ۵۷۹ است", "fused": 0.9},
        {"doc_id": 91, "text": "اطلاعات رتبه‌بندی", "fused": 0.7},
    ]
    res = resolver("رتبه C1", mock_hits)
    assert isinstance(res, dict)
    assert "status" in res
    assert res["status"] in ("answer", "clarify")
    if res["status"] == "answer":
        assert "brief" in res
        assert "cites" in res
        assert isinstance(res["cites"], list)
        assert len(res["cites"]) > 0


def test_chunker_adds_metadata_prefix():
    chunker = R.get("chunker")

    docs = [
        {
            "doc_id": 100,
            "Question": "رتبه اعتباری چیست؟",
            "Category": "مفاهیم پایه",
            "Keyword": "رتبه، امتیاز",
            "Answer": (
                "رتبه اعتباری نشان دهنده "
                "سطح ریسک اعتباری است."
            ),
        }
    ]

    chunks = chunker(docs)

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk["text"].startswith(
        "[مفاهیم پایه | رتبه، امتیاز]"
    )

    assert (
        chunk["category"]
        == "مفاهیم پایه"
    )

    assert (
        chunk["keyword"]
        == "رتبه، امتیاز"
    )

    assert (
        chunk["question"]
        == "رتبه اعتباری چیست؟"
    )


def test_chunker_respects_length_bound():
    chunker = R.get("chunker")

    long_text = " ".join(
        [
            (
                "رفتار بازپرداخت منظم "
                "باعث بهبود امتیاز اعتباری می شود."
            )
            for _ in range(30)
        ]
    )

    docs = [
        {
            "doc_id": 101,
            "Category": "تسهیلات",
            "Keyword": "بازپرداخت",
            "Answer": long_text,
        }
    ]

    chunks = chunker(docs)

    assert len(chunks) > 1

    assert all(
        len(chunk["text"])
        <= 360
        for chunk in chunks
    )


def test_chunker_preserves_sentence_units():
    chunker = R.get("chunker")

    sentence_one = (
        "رفتار بازپرداخت منظم تسهیلات "
        + "اطلاعات اعتباری مهمی ایجاد می کند " * 7
        + "و در رتبه اعتباری موثر است."
    )

    sentence_two = (
        "چک برگشتی نیز یک سیگنال منفی "
        + "در ارزیابی ریسک اعتباری محسوب می شود " * 6
        + "و می تواند امتیاز را کاهش دهد."
    )

    docs = [
        {
            "doc_id": 102,
            "Category": "عوامل موثر بر امتیاز",
            "Keyword": "بازپرداخت، چک",
            "Answer": (
                sentence_one
                + " "
                + sentence_two
            ),
        }
    ]

    chunks = chunker(docs)

    # Each sentence is below the body budget,
    # but together they exceed it.
    assert any(
        sentence_one in chunk["text"]
        for chunk in chunks
    )

    assert any(
        sentence_two in chunk["text"]
        for chunk in chunks
    )


def test_chunker_keeps_numeric_unit_together():
    chunker = R.get("chunker")

    text = (
        ("اطلاعات اعتباری " * 25)
        + "250 میلیون تومان "
        + ("در گزارش بررسی می شود " * 20)
        + "."
    )

    docs = [
        {
            "doc_id": 103,
            "Category": "تسهیلات",
            "Keyword": "مبلغ",
            "Answer": text,
        }
    ]

    chunks = chunker(docs)

    assert any(
        "250 میلیون تومان"
        in chunk["text"]
        for chunk in chunks
    )

    assert all(
        len(chunk["text"])
        <= 360
        for chunk in chunks
    )

def test_reranker_demotes_ungrounded_content():
    import csv
    from pathlib import Path

    reranker = R.get("reranker")

    corpus_path = (
        Path(__file__).resolve().parents[1]
        / "corpus"
        / "test.csv"
    )

    with corpus_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        row = next(csv.DictReader(f))

    genuine_text = (
        row.get("Answer")
        or row.get("BriefAnswer")
    )

    hits = [
        {
            "doc_id": 999999,
            "text": (
                "رتبه C3 C3 C3 رتبه امتیاز "
                "این یک متن ساختگی خارج از "
                "مجموعه معتبر است"
            ),
            "fused": 1.0,
        },
        {
            "doc_id": 1,
            "text": genuine_text,
            "fused": 0.5,
        },
    ]

    ranked = reranker(
        "رتبه اعتباری چیست؟",
        hits,
    )

    # Contract: reranker must preserve candidate count.
    assert len(ranked) == 2

    # Genuine corpus evidence must rank first.
    assert ranked[0]["text"] == genuine_text
    assert ranked[0]["verified_corpus"] is True

    # Ungrounded content remains present but is demoted.
    assert ranked[1]["doc_id"] == 999999
    assert ranked[1]["verified_corpus"] is False

def test_reranker_does_not_use_doc_id_as_poison_rule():
    import csv
    from pathlib import Path

    reranker = R.get("reranker")

    corpus_path = (
        Path(__file__).resolve().parents[1]
        / "corpus"
        / "test.csv"
    )

    with corpus_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        row = next(csv.DictReader(f))

    genuine_text = (
        row.get("Answer")
        or row.get("BriefAnswer")
    )

    hits = [
        {
            "doc_id": 99999,
            "text": genuine_text,
            "fused": 0.5,
        }
    ]

    ranked = reranker(
        "اعتبار",
        hits,
    )

    assert len(ranked) == 1
    assert ranked[0]["verified_corpus"] is True

def test_resolver_clarifies_missing_debt_type():
    resolver = R.get("resolver")

    result = resolver(
        "بدهی دارم، روی امتیازم چه تاثیری داره؟",
        [
            {
                "doc_id": 1,
                "text": "نمونه پاسخ",
                "fused": 0.8,
            }
        ],
    )

    assert result["status"] == "clarify"
    assert "question" in result
    assert "options" in result

def test_resolver_does_not_overclarify_specific_query():
    resolver = R.get("resolver")

    result = resolver(
        "بدهی مالیاتی دارم، چه تاثیری روی امتیازم داره؟",
        [
            {
                "doc_id": 1,
                "text": "نمونه پاسخ",
                "fused": 0.8,
            }
        ],
    )

    assert result["status"] == "answer"

def test_resolver_clarifies_missing_bank_for_loan_eligibility():
    resolver = R.get("resolver")

    result = resolver(
        "رتبه C3 دارم، بهم وام میدن؟",
        [
            {
                "doc_id": 1,
                "text": "نمونه پاسخ",
                "fused": 0.8,
            }
        ],
    )

    assert result["status"] == "clarify"
    assert "question" in result
    assert "options" in result

def test_resolver_does_not_clarify_general_cross_bank_effect():
    resolver = R.get("resolver")

    result = resolver(
        (
            "اگر شرکت در یک بانک خاص سابقه تاخیر در پرداخت بدهی "
            "داشته باشد، آیا این موضوع بر دریافت تسهیلات از سایر "
            "بانک‌ها تاثیر می‌گذارد؟"
        ),
        [
            {
                "doc_id": 1,
                "text": "نمونه پاسخ",
                "fused": 0.8,
            }
        ],
    )

    assert result["status"] == "answer"

def test_resolver_does_not_clarify_supplier_debt():
    resolver = R.get("resolver")

    result = resolver(
        (
            "اگر شرکت بدهی به تامین‌کنندگان داشته باشد، "
            "آیا این بدهی در گزارش اعتباری ثبت می‌شود؟"
        ),
        [
            {
                "doc_id": 1,
                "text": "نمونه پاسخ",
                "fused": 0.8,
            }
        ],
    )

    assert result["status"] == "answer"

def test_reranker_promotes_unique_documents_before_duplicates():
    import csv
    from pathlib import Path

    reranker = R.get("reranker")

    corpus_path = (
        Path(__file__).resolve().parents[1]
        / "corpus"
        / "test.csv"
    )

    with corpus_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        row = next(csv.DictReader(f))

    genuine_text = (
        row.get("Answer")
        or row.get("BriefAnswer")
    )

    # All three candidates use genuine corpus content,
    # so Track F grounding will accept all of them.
    #
    # Before diversity, fused score would prefer:
    # doc 1, doc 1, doc 2
    hits = [
        {
            "doc_id": 1,
            "text": genuine_text,
            "fused": 1.0,
        },
        {
            "doc_id": 1,
            "text": genuine_text,
            "fused": 0.9,
        },
        {
            "doc_id": 2,
            "text": genuine_text,
            "fused": 0.8,
        },
    ]

    ranked = reranker(
        "اعتبار",
        hits,
    )

    # Reranker contract: preserve all candidates.
    assert len(ranked) == 3

    # Diversity should prevent the duplicate document
    # from occupying two high-priority evidence slots.
    assert [
        hit["doc_id"]
        for hit in ranked
    ] == [1, 2, 1]

    assert (
        ranked[0]["duplicate_document"]
        is False
    )

    assert (
        ranked[1]["duplicate_document"]
        is False
    )

    assert (
        ranked[2]["duplicate_document"]
        is True
    )

def test_polarity_detects_negated_history_conflict():
    conflicts = _polarity_conflicts(
        "شرکت سابقه تاخیر ندارد",
        "شرکت سابقه تاخیر دارد",
    )

    assert "سابقه" in conflicts
    assert "تاخیر" in conflicts

def test_polarity_accepts_matching_negation():
    conflicts = _polarity_conflicts(
        "شرکت سابقه تاخیر ندارد",
        "شرکت هیچ سابقه تاخیری ندارد",
    )

    assert conflicts == []

def test_polarity_ignores_volitional_negation():
    conflicts = _polarity_conflicts(
        "نمی‌خواهم چک برگشتی در سابقه‌ام باشد",
        "چک برگشتی روی امتیاز اعتباری تاثیر دارد",
    )

    assert conflicts == []

def test_hybrid_rehydrates_known_remote_doc_from_trusted_corpus():
    from rubbish_rag.pipeline import (
        _hybrid_union,
        _load_local_question_index,
    )

    trusted = (
        _load_local_question_index()[0]
    )

    remote_hits = [
        {
            "doc_id": trusted["doc_id"],
            "text": "remote snippet",
            "bm25": 1.0,
            "dense": 1.0,
            "fused": 1.0,
        }
    ]

    merged = _hybrid_union(
        remote_hits,
        [],
    )

    assert len(merged) == 1

    hit = merged[0]

    assert (
        hit["retrieved_text"]
        == "remote snippet"
    )

    assert (
        hit["text"]
        == trusted["text"]
    )

    assert (
        hit["evidence_source"]
        == "trusted_local_corpus"
    )

    # Remote retrieval signal is still preserved.
    assert hit["remote_rank"] == 1
    assert hit["fused"] == 1.0

def test_hybrid_does_not_trust_unknown_remote_document():
    from rubbish_rag.pipeline import (
        _hybrid_union,
    )

    remote_hits = [
        {
            "doc_id": 999999,
            "text": (
                "ساختگی و خارج از "
                "مجموعه معتبر"
            ),
            "bm25": 10.0,
            "dense": 1.0,
            "fused": 1.0,
        }
    ]

    merged = _hybrid_union(
        remote_hits,
        [],
    )

    assert len(merged) == 1

    hit = merged[0]

    assert (
        hit["text"]
        == "ساختگی و خارج از مجموعه معتبر"
    )

    assert (
        hit["evidence_source"]
        == "retrieved_only"
    )