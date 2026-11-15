"""Numbers, dates and phone numbers to spoken words.

Deterministic on purpose: the model never writes spoken numbers, this code does.
Commas in the output are short pauses for the text-to-speech voice.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

MAX_WHOLE = 999_999_999_999  # well above any loan here; keeps wording unambiguous

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen",
]  # fmt: skip
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORDINALS = [
    "", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
    "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
    "seventeenth", "eighteenth", "nineteenth", "twentieth", "twenty-first", "twenty-second",
    "twenty-third", "twenty-fourth", "twenty-fifth", "twenty-sixth", "twenty-seventh",
    "twenty-eighth", "twenty-ninth", "thirtieth", "thirty-first",
]  # fmt: skip
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = [
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
]  # fmt: skip


def _below_hundred(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


def _below_thousand(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    parts: list[str] = []
    if hundreds:
        parts.append(f"{_ONES[hundreds]} hundred")
    if rest:
        if hundreds:
            parts.append("and")
        parts.append(_below_hundred(rest))
    return " ".join(parts)


def number_to_words(n: int) -> str:
    """25000 -> 'twenty-five thousand'; 1005 -> 'one thousand and five'."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("number_to_words needs an int")
    if n < 0 or n > MAX_WHOLE:
        raise ValueError(f"number out of range: {n}")
    if n == 0:
        return "zero"
    parts: list[str] = []
    for size, name in ((10**9, "billion"), (10**6, "million"), (10**3, "thousand")):
        value, n = divmod(n, size)
        if value:
            parts.append(f"{_below_thousand(value)} {name}")
    if n:
        words = _below_thousand(n)
        parts.append(f"and {words}" if parts and n < 100 else words)
    return " ".join(parts)


def naira_to_words(amount: Decimal | int | float | str) -> str:
    """1250.50 -> 'one thousand two hundred and fifty naira, fifty kobo'."""
    try:
        value = Decimal(str(amount))
    except InvalidOperation as e:
        raise ValueError(f"not an amount: {amount!r}") from e
    if not value.is_finite() or value < 0:
        raise ValueError(f"not a valid amount: {amount!r}")
    value = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    naira, kobo = divmod(int(value * 100), 100)
    text = f"{number_to_words(naira)} naira"
    if kobo:
        text += f", {number_to_words(kobo)} kobo"
    return text


def date_to_words(d: date) -> str:
    """2026-11-07 -> 'Saturday, the seventh of November'."""
    return f"{_WEEKDAYS[d.weekday()]}, the {_ORDINALS[d.day]} of {_MONTHS[d.month - 1]}"


def _digits_to_words(digits: str) -> str:
    return " ".join(_ONES[int(c)] for c in digits)


def phone_to_words(phone: str) -> str:
    """Digit by digit, with pauses between groups: '08031234567' -> 'zero eight zero three, one two ...'."""
    cleaned = re.sub(r"[\s\-().]", "", phone)
    plus = cleaned.startswith("+")
    digits = cleaned[1:] if plus else cleaned
    if not re.fullmatch(r"[0-9]+", digits):
        raise ValueError(f"not a phone number: {phone!r}")
    if plus and digits.startswith("234") and len(digits) == 13:
        groups = [digits[:3], digits[3:6], digits[6:9], digits[9:]]
    elif not plus and digits.startswith("0") and len(digits) == 11:
        groups = [digits[:4], digits[4:7], digits[7:]]
    else:
        groups = [digits[i : i + 3] for i in range(0, len(digits), 3)]
    return ("plus " if plus else "") + ", ".join(_digits_to_words(g) for g in groups)


def spell_out(code: str) -> str:
    """IDs, letter by letter and digit by digit: 'B001' -> 'B, zero zero one'."""
    runs = re.findall(r"[A-Za-z]+|[0-9]+", code)
    if not runs:
        raise ValueError(f"nothing to spell: {code!r}")
    spoken = [_digits_to_words(r) if r.isdigit() else " ".join(r.upper()) for r in runs]
    return ", ".join(spoken)
