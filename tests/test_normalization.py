"""Unit tests for Persian text normalization."""
import pytest
from rubbish_rag.normalize_fa import normalize_fa


def test_digit_normalization():
    assert normalize_fa("۲۵۰") == "250"
    assert normalize_fa("٢٥٠") == "250"
    assert normalize_fa("امتیاز ۵۶۰ تا ۵۷۹") == "امتیاز 560 تا 579"
    assert normalize_fa("سال ۱۴۰۲") == "سال 1402"


def test_arabic_character_normalization():
    assert normalize_fa("بانك ملي") == "بانک ملی"
    assert normalize_fa("تسهيلات ويژه") == "تسهیلات ویژه"
    assert normalize_fa("دوره") == "دوره"


def test_zwnj_normalization():
    assert normalize_fa("می\u200cشود") == "می شود"
    assert normalize_fa("رتبه\u200cبندی") == "رتبه بندی"


def test_colloquial_mapping():
    assert "می شود" in normalize_fa("میشه وام گرفت؟")
    assert "نمی شود" in normalize_fa("نمیشه")
    assert "رتبه من" in normalize_fa("رتبم افت کرده")
    assert "امتیاز من" in normalize_fa("امتیازم چنده")
    assert "شرکت من" in normalize_fa("شرکتم امتیاز نداره")


def test_whitespace_and_empty():
    assert normalize_fa("") == ""
    assert normalize_fa(None) == ""
    assert normalize_fa("   سلام    روزتون   بخیر   ") == "سلام روزتون بخیر"


def test_normalization_is_idempotent():
    cases = [
        "میشه وام گرفت؟",
        "نمیشه",
        "رتبه‌بندی اعتباری",
        "بانك ملي",
        "امتیاز ۲۵۰",
        "رتبم افت کرده",
    ]

    for text in cases:
        once = normalize_fa(text)

        assert normalize_fa(once) == once
