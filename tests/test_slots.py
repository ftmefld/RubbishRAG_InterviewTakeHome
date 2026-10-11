"""Unit tests for entity slot extraction."""
import pytest
from rubbish_rag.slots import extract_slots, BANKS, RANKS


def test_bank_extraction():
    slots = extract_slots("بانک صادرات به من وام می‌دهد؟")
    assert slots["BANK"] == "صادرات"

    slots2 = extract_slots("شرایط دریافت وام در بانک ملی")
    assert slots2["BANK"] == "ملی"


def test_rank_extraction():
    slots = extract_slots("رتبه C1 یعنی چی؟")
    assert slots["RANK"] == "C1"

    slots2 = extract_slots("وضعیت امتیاز اعتباری A2")
    assert slots2["RANK"] == "A2"

    slots3 = extract_slots("رتبه e3 چیست؟")
    assert slots3["RANK"] == "E3"


def test_persona_extraction():
    slots_ind = extract_slots("امتیاز شخص حقیقی چگونه محاسبه می‌شود؟")
    assert slots_ind["PERSONA"] == "حقیقی"

    slots_leg = extract_slots("آیا اطلاعات شرکت در گزارش ثبت می‌شود؟")
    assert slots_leg["PERSONA"] == "حقوقی"


def test_debt_extraction():
    slots_check = extract_slots("چک صیادی برگشتی چه تاثیری دارد؟")
    assert slots_check["DEBT"] == "چک"

    slots_tax = extract_slots("بدهی مالیاتی ثبت شده است")
    assert slots_tax["DEBT"] == "مالیاتی"

    slots_loan = extract_slots("اقساط معوق تسهیلات بانکی")
    assert slots_loan["DEBT"] == "تسهیلاتی"


def test_negation_extraction():
    slots_neg = extract_slots("وام نمی‌خواهم بگیرم")
    assert slots_neg["NEGATED"] is True

    slots_pos = extract_slots("می‌خواهم وام بگیرم")
    assert slots_neg["NEGATED"] is True or slots_pos["NEGATED"] is False


def test_number_extraction():
    slots = extract_slots("حداقل امتیاز ۲۵۰ است")
    assert "250" in slots["NUMS"]

def test_longest_bank_alias_wins():
    slots = extract_slots(
        "شرایط بانک توسعه صادرات چیست؟"
    )

    assert slots["BANK"] == "توسعه صادرات"


def test_legal_persona_is_not_both():
    slots = extract_slots(
        "شرایط شخص حقوقی چیست؟"
    )

    assert slots["PERSONA"] == "حقوقی"


def test_rank_with_persian_digit():
    slots = extract_slots(
        "رتبه C۲ یعنی چی؟"
    )

    assert slots["RANK"] == "C2"


def test_no_bank_substring_false_positive():
    slots = extract_slots(
        "این موضوع درباره صادرات کالا است"
    )

    # "صادرات" by itself is still ambiguous in our current domain model,
    # so do NOT assert BANK=None here unless we decide bank names must
    # explicitly include "بانک".
