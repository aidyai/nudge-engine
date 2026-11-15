"""The last gate before the agent's words are spoken. Returns a short reason if the reply breaks a rule, else None.

The rules: no digits, nothing banned, short, no loan details before verification, and every amount
and date must be copied word for word from something code produced (a tool result or a read-back).
"""

import re

BANNED = [
    "police",
    "court",
    "arrest",
    "lawyer",
    "legal action",
    "sue you",
    "jail",
    "prison",
    "your employer",
    "your family",
    "your contacts",
    "your relatives",
    "shame",
    "embarrass",
    "final warning",
    "last warning",
]
NUMBER_WORDS = set(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand "
    "million billion and".split()
)
DAYS = "monday tuesday wednesday thursday friday saturday sunday"
MONTHS = "january february march april may june july august september october november december"
DATE = re.compile(rf"(?:(?:{DAYS.replace(' ', '|')}) )?the [a-z]+(?: [a-z]+)? of (?:{MONTHS.replace(' ', '|')})")


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[-,.?!:;]", " ", s.lower())).strip()


def money_phrases(text):
    """'twenty five thousand naira' and 'fifty kobo' as they appear in the text."""
    words, out = norm(text).split(), []
    for i, w in enumerate(words):
        if w in ("naira", "kobo"):
            j = i
            while j > 0 and words[j - 1] in NUMBER_WORDS:
                j -= 1
            out.append(" ".join(words[j : i + 1]))
    return out


def check_output(reply, allowed, verified, must_say=""):
    text, low = reply.strip(), reply.lower()
    allowed_norm = norm(" ".join(allowed))
    if not text:
        return "empty reply"
    if re.search(r"\d", text):
        return "no digits: write numbers as words copied from the tool results"
    if len(re.findall(r"[.!?](?:\s|$)", text)) > (3 if must_say else 2):
        return "too long: two short sentences at most"
    for phrase in BANNED:
        if phrase in low:
            return f"banned wording: {phrase}"
    money, dates = money_phrases(text), DATE.findall(norm(text))
    if not verified and (money or dates):
        return "loan details before the borrower is verified"
    for phrase in money + dates:
        if phrase not in allowed_norm:
            return f"not from a tool: {phrase}"
    if must_say and norm(must_say) not in norm(text):
        return "say the read-back sentence exactly as given"
    return None
