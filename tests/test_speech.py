import re
from datetime import date
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from engine.speech import (
    MAX_WHOLE,
    date_to_words,
    naira_to_words,
    number_to_words,
    phone_to_words,
    spell_out,
)

_UNITS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"thousand": 10**3, "million": 10**6, "billion": 10**9}
_DIGIT_WORDS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine".split())}


def parse_words(text: str) -> int:
    """The inverse of number_to_words, written independently so the round trip means something."""
    total = current = 0
    for token in re.split(r"[ -]", text):
        if token == "and":
            continue
        if token in _UNITS:
            current += _UNITS[token]
        elif token in _TENS:
            current += _TENS[token]
        elif token == "hundred":
            current *= 100
        elif token in _SCALES:
            total += current * _SCALES[token]
            current = 0
        else:
            raise AssertionError(f"unknown word {token!r}")
    return total + current


# ------------------------------------------------------------ the examples from the spec


def test_spec_examples():
    assert naira_to_words(25000) == "twenty-five thousand naira"
    assert naira_to_words(Decimal("1250.50")) == "one thousand two hundred and fifty naira, fifty kobo"
    assert date_to_words(date(2026, 11, 7)) == "Saturday, the seventh of November"


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "zero"),
        (5, "five"),
        (13, "thirteen"),
        (20, "twenty"),
        (21, "twenty-one"),
        (100, "one hundred"),
        (101, "one hundred and one"),
        (999, "nine hundred and ninety-nine"),
        (1000, "one thousand"),
        (1005, "one thousand and five"),
        (1250, "one thousand two hundred and fifty"),
        (100_000, "one hundred thousand"),
        (101_000, "one hundred and one thousand"),
        (1_000_000, "one million"),
        (1_000_001, "one million and one"),
        (2_500_000, "two million five hundred thousand"),
    ],
)
def test_number_examples(n, words):
    assert number_to_words(n) == words


def test_kobo_rounds_half_up():
    assert naira_to_words("10.005") == "ten naira, one kobo"
    assert naira_to_words("0.50") == "zero naira, fifty kobo"


# ------------------------------------------------------------ properties


@given(st.integers(min_value=0, max_value=MAX_WHOLE))
def test_number_round_trips(n):
    words = number_to_words(n)
    assert parse_words(words) == n
    assert re.fullmatch(r"[a-z \-]+", words)  # no digits, nothing for the voice to trip on


@given(st.integers(min_value=0, max_value=999_999_999), st.integers(min_value=0, max_value=99))
def test_naira_round_trips(whole, kobo):
    text = naira_to_words(Decimal(f"{whole}.{kobo:02d}"))
    naira_part, _, kobo_part = text.partition(", ")
    assert naira_part.endswith(" naira")
    assert parse_words(naira_part.removesuffix(" naira")) == whole
    if kobo:
        assert kobo_part.endswith(" kobo")
        assert parse_words(kobo_part.removesuffix(" kobo")) == kobo
    else:
        assert kobo_part == ""


_WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@given(st.dates(min_value=date(2000, 1, 1), max_value=date(2100, 12, 31)))
def test_dates_are_well_formed(d):
    text = date_to_words(d)
    assert re.fullmatch(r"[A-Z][a-z]+day, the [a-z\-]+ of [A-Z][a-z]+", text)
    assert text.startswith(_WEEKDAY_NAMES[d.weekday()])


def test_every_day_of_month_has_its_own_ordinal():
    ordinals = {date_to_words(date(2026, 1, day)).split(" of ")[0] for day in range(1, 32)}
    assert len(ordinals) == 31


def test_known_weekdays():
    assert date_to_words(date(2026, 11, 10)) == "Tuesday, the tenth of November"
    assert date_to_words(date(2026, 11, 13)) == "Friday, the thirteenth of November"


@given(st.text(alphabet="0123456789", min_size=5, max_size=15), st.booleans())
def test_phone_keeps_every_digit_in_order(digits, plus):
    spoken = phone_to_words(("+" if plus else "") + digits)
    assert spoken.startswith("plus ") == plus
    words = spoken.removeprefix("plus ").replace(",", " ").split()
    assert "".join(_DIGIT_WORDS[w] for w in words) == digits


# ------------------------------------------------------------ phone, ids, errors


def test_phone_examples():
    assert phone_to_words("08031234567") == "zero eight zero three, one two three, four five six seven"
    assert phone_to_words("+234 803 123 4567") == ("plus two three four, eight zero three, one two three, four five six seven")


def test_spell_out_ids():
    assert spell_out("B001") == "B, zero zero one"
    assert spell_out("ab12") == "A B, one two"


@pytest.mark.parametrize("bad", [-1, MAX_WHOLE + 1])
def test_number_out_of_range(bad):
    with pytest.raises(ValueError):
        number_to_words(bad)


@pytest.mark.parametrize("bad", ["5", 5.0, True])
def test_number_wrong_type(bad):
    with pytest.raises(TypeError):
        number_to_words(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["abc", "-5", "NaN", "Infinity"])
def test_naira_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        naira_to_words(bad)


@pytest.mark.parametrize("bad", ["08O31234567", "", "call me"])
def test_phone_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        phone_to_words(bad)


def test_spell_out_rejects_empty():
    with pytest.raises(ValueError):
        spell_out("--")
