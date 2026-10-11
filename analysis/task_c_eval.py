import copy
import json
from pathlib import Path

import rubbish_rag as R
import rubbish_rag.pipeline  # registers pipeline stages

from rubbish_rag.normalize_fa import normalize_fa
from rubbish_rag.slots import extract_slots


OUTPUT_PATH = Path("analysis/task_c_eval_results.json")


# ---------------------------------------------------------
# 1. Normalization evaluation
# ---------------------------------------------------------

NORMALIZATION_CASES = [
    ("امتیاز ۲۵۰", "امتیاز 250"),
    ("امتیاز ٢٥٠", "امتیاز 250"),
    ("امتیاز 250", "امتیاز 250"),

    ("بانك ملي", "بانک ملی"),
    ("بانک ملی", "بانک ملی"),

    ("می‌شود", "می شود"),
    ("میشه", "می شود"),

    ("نمی‌شود", "نمی شود"),
    ("نمیشه", "نمی شود"),

    ("رتبه‌بندی", "رتبه بندی"),

    ("رتبم C۲ هست", "رتبه من C2 هست"),
    ("امتیازم ۵۲۰ شده", "امتیاز من 520 شده"),

    ("قسطامو دادم", "اقساط من را دادم"),
]


def evaluate_normalization():
    passed = 0
    rows = []

    for raw, expected in NORMALIZATION_CASES:
        actual = normalize_fa(raw)

        ok = actual == expected

        if ok:
            passed += 1

        rows.append(
            {
                "input": raw,
                "expected": expected,
                "actual": actual,
                "passed": ok,
            }
        )

    return {
        "passed": passed,
        "total": len(NORMALIZATION_CASES),
        "accuracy": passed / len(NORMALIZATION_CASES),
        "cases": rows,
    }


# ---------------------------------------------------------
# 2. Idempotence evaluation
# ---------------------------------------------------------

IDEMPOTENCE_CASES = [
    "میشه وام گرفت؟",
    "نمیشه",
    "رتبه‌بندی اعتباری",
    "بانك ملي",
    "امتیاز ۲۵۰",
    "امتیاز ٢٥٠",
    "رتبم C۲ شده",
    "امتیازم ۵۲۰ شده",
    "قسطامو پرداخت کردم",
    "شرایط شخص حقوقی چیست؟",
]


def evaluate_idempotence():
    passed = 0
    rows = []

    for text in IDEMPOTENCE_CASES:
        once = normalize_fa(text)
        twice = normalize_fa(once)

        ok = once == twice

        if ok:
            passed += 1

        rows.append(
            {
                "input": text,
                "once": once,
                "twice": twice,
                "passed": ok,
            }
        )

    return {
        "passed": passed,
        "total": len(IDEMPOTENCE_CASES),
        "rate": passed / len(IDEMPOTENCE_CASES),
        "cases": rows,
    }


# ---------------------------------------------------------
# 3. Slot extraction evaluation
# ---------------------------------------------------------

SLOT_CASES = [
    (
        "رتبه C۲ یعنی چی؟",
        {
            "RANK": "C2",
        },
    ),
    (
        "رتبه a1 چیست؟",
        {
            "RANK": "A1",
        },
    ),
    (
        "شرایط بانک توسعه صادرات چیست؟",
        {
            "BANK": "توسعه صادرات",
        },
    ),
    (
        "وام بانک ملی را چطور بررسی کنم؟",
        {
            "BANK": "ملی",
            "DEBT": "تسهیلاتی",
        },
    ),
    (
        "شرایط شخص حقوقی چیست؟",
        {
            "PERSONA": "حقوقی",
        },
    ),
    (
        "برای شخص حقیقی چه شرایطی وجود دارد؟",
        {
            "PERSONA": "حقیقی",
        },
    ),
    (
        "من بدهی مالیاتی ندارم",
        {
            "DEBT": "مالیاتی",
            "NEGATED": True,
        },
    ),
    (
        "چک برگشتی روی امتیاز چه اثری دارد؟",
        {
            "DEBT": "چک",
        },
    ),
    (
        "اقساط وامم عقب افتاده",
        {
            "DEBT": "تسهیلاتی",
        },
    ),
    (
        "شرکت محکومیت مالی دارد",
        {
            "PERSONA": "حقوقی",
            "DEBT": "محکومیت",
        },
    ),
    (
        "امتیاز من ۵۲۰ است",
        {
            "AMOUNT": "520",
        },
    ),
    (
        "در ۶ ماه گذشته چه تاثیری دارد؟",
        {
            "AMOUNT": "6",
            "TIME": True,
        },
    ),
]


def evaluate_slots():
    passed_cases = 0
    field_correct = 0
    field_total = 0

    rows = []

    for query, expected in SLOT_CASES:
        actual = extract_slots(query)

        case_ok = True
        field_results = {}

        for key, expected_value in expected.items():
            actual_value = actual.get(key)

            ok = actual_value == expected_value

            field_total += 1
            field_correct += int(ok)

            field_results[key] = {
                "expected": expected_value,
                "actual": actual_value,
                "passed": ok,
            }

            if not ok:
                case_ok = False

        if case_ok:
            passed_cases += 1

        rows.append(
            {
                "query": query,
                "expected": expected,
                "actual": actual,
                "passed": case_ok,
                "fields": field_results,
            }
        )

    return {
        "passed_cases": passed_cases,
        "total_cases": len(SLOT_CASES),
        "case_accuracy": passed_cases / len(SLOT_CASES),
        "field_accuracy": field_correct / field_total,
        "cases": rows,
    }


# ---------------------------------------------------------
# 4. Reranker integration ablation
# ---------------------------------------------------------

def baseline_rerank(query, hits):
    """
    Original starter reranker:
    raw token overlap with no Persian normalization.
    """
    q = set(query.split())

    output = copy.deepcopy(hits)

    for hit in output:
        toks = set(hit["text"].split())

        hit["rerank_score"] = len(
            q & toks
        )

    return sorted(
        output,
        key=lambda h: h["rerank_score"],
        reverse=True,
    )


RERANK_CASES = [
    {
        "name": "persian_digit",
        "query": "امتیاز ۲۵۰ چه رتبه ای دارد؟",
        "gold_doc_id": 1,
        "hits": [
            {
                "doc_id": 2,
                "text": "رتبه اعتباری افراد چگونه تعیین می شود",
            },
            {
                "doc_id": 1,
                "text": "امتیاز 250 در این بازه اعتباری قرار دارد",
            },
        ],
    },

    {
        "name": "arabic_letters",
        "query": "شرایط بانك ملي چیست؟",
        "gold_doc_id": 3,
        "hits": [
            {
                "doc_id": 4,
                "text": "شرایط بانک ملت برای دریافت خدمات",
            },
            {
                "doc_id": 3,
                "text": "شرایط بانک ملی برای دریافت خدمات",
            },
        ],
    },

    {
        "name": "rank_digit_and_colloquial",
        "query": "رتبه C۲ چیه؟",
        "gold_doc_id": 5,
        "hits": [
            {
                "doc_id": 6,
                "text": "رتبه اعتباری به چه معنا است",
            },
            {
                "doc_id": 5,
                "text": "رتبه C2 چیست و چه معنایی دارد",
            },
        ],
    },

    {
        "name": "half_space",
        "query": "چرا امتیازم کم می‌شود؟",
        "gold_doc_id": 7,
        "hits": [
            {
                "doc_id": 8,
                "text": "امتیاز اعتباری چگونه محاسبه می شود",
            },
            {
                "doc_id": 7,
                "text": "چرا امتیاز من کم می شود و چه عواملی موثر است",
            },
        ],
    },
]


def rank_of(hits, doc_id):
    for index, hit in enumerate(hits, start=1):
        if hit["doc_id"] == doc_id:
            return index

    return None


def evaluate_reranker():
    registered_reranker = R.get("reranker")

    improved = 0
    regressed = 0
    unchanged = 0

    baseline_top1 = 0
    normalized_top1 = 0

    rows = []

    for case in RERANK_CASES:
        query = case["query"]
        gold = case["gold_doc_id"]

        baseline = baseline_rerank(
            query,
            case["hits"],
        )

        normalized = registered_reranker(
            query,
            copy.deepcopy(case["hits"]),
        )

        baseline_rank = rank_of(
            baseline,
            gold,
        )

        normalized_rank = rank_of(
            normalized,
            gold,
        )

        baseline_top1 += int(
            baseline_rank == 1
        )

        normalized_top1 += int(
            normalized_rank == 1
        )

        if normalized_rank < baseline_rank:
            improved += 1
            change = "improved"

        elif normalized_rank > baseline_rank:
            regressed += 1
            change = "regressed"

        else:
            unchanged += 1
            change = "unchanged"

        rows.append(
            {
                "name": case["name"],
                "query": query,
                "gold_doc_id": gold,
                "baseline_rank": baseline_rank,
                "normalized_rank": normalized_rank,
                "change": change,
            }
        )

    n = len(RERANK_CASES)

    return {
        "cases": n,
        "baseline_top1_accuracy": (
            baseline_top1 / n
        ),
        "normalized_top1_accuracy": (
            normalized_top1 / n
        ),
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
        "details": rows,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    normalization = evaluate_normalization()
    idempotence = evaluate_idempotence()
    slots = evaluate_slots()
    reranker = evaluate_reranker()

    print("\nTask C — Persian Normalization & Slot Evaluation")
    print("=" * 72)

    print("\nNormalization")
    print("-" * 72)

    print(
        f"Correct: "
        f"{normalization['passed']}/"
        f"{normalization['total']} "
        f"({normalization['accuracy'] * 100:.1f}%)"
    )

    print("\nIdempotence")
    print("-" * 72)

    print(
        f"Stable: "
        f"{idempotence['passed']}/"
        f"{idempotence['total']} "
        f"({idempotence['rate'] * 100:.1f}%)"
    )

    print("\nSlot extraction")
    print("-" * 72)

    print(
        f"Cases: "
        f"{slots['passed_cases']}/"
        f"{slots['total_cases']} "
        f"({slots['case_accuracy'] * 100:.1f}%)"
    )

    print(
        f"Field accuracy: "
        f"{slots['field_accuracy'] * 100:.1f}%"
    )

    print("\nReranker integration ablation")
    print("-" * 72)

    print(
        f"Baseline Top-1 accuracy   : "
        f"{reranker['baseline_top1_accuracy']:.3f}"
    )

    print(
        f"Normalized Top-1 accuracy : "
        f"{reranker['normalized_top1_accuracy']:.3f}"
    )

    print(
        f"Improved={reranker['improved']}  "
        f"Regressed={reranker['regressed']}  "
        f"Unchanged={reranker['unchanged']}"
    )

    print("\nPer-case reranker changes")

    for row in reranker["details"]:
        print(
            f"{row['name']:<26} "
            f"baseline={row['baseline_rank']} "
            f"normalized={row['normalized_rank']} "
            f"{row['change']}"
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
                "normalization": normalization,
                "idempotence": idempotence,
                "slots": slots,
                "reranker": reranker,
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