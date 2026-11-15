import pytest

from engine.output_check import check_output

OK = "Hello, this is an automated assistant. Am I speaking with Musa Bello?"
AMOUNT = "twenty-five thousand naira"
DATE = "Tuesday, the tenth of November"


def test_a_normal_greeting_passes():
    assert check_output(OK, [], verified=False) is None


def test_ordinary_number_words_are_not_flagged():
    assert check_output("One moment please. I will check that for you.", [], verified=True) is None


@pytest.mark.parametrize(
    ("reply", "why"),
    [
        ("You owe 25,000 naira.", "digit"),
        ("Hello. This is a call. I am an assistant.", "too long"),
        ("If you do not pay we will call the police.", "banned"),
        ("We will tell your employer.", "banned"),
        ("", "empty"),
    ],
)
def test_bad_replies_are_stopped(reply, why):
    assert check_output(reply, [AMOUNT], verified=True) is not None, why


def test_no_loan_details_before_verification():
    assert check_output(f"You owe {AMOUNT}.", [AMOUNT], verified=False) == "loan details before the borrower is verified"
    assert check_output(f"It is due on {DATE}.", [DATE], verified=False) == "loan details before the borrower is verified"
    assert check_output(f"You owe {AMOUNT}.", [AMOUNT], verified=True) is None


def test_amounts_must_come_from_a_tool():
    assert check_output("You owe thirty thousand naira.", [AMOUNT], verified=True) == "not from a tool: thirty thousand naira"
    assert check_output(f"You owe {AMOUNT}.", [], verified=True) is not None


def test_dates_must_come_from_a_tool():
    assert check_output("It is due on Friday, the thirteenth of November.", [DATE], verified=True) is not None
    assert check_output("It is due on the tenth of November.", [DATE], verified=True) is None  # a piece of an allowed date is fine


def test_hyphens_and_commas_do_not_matter():
    allowed = ["one thousand two hundred and fifty naira, fifty kobo"]
    assert check_output("That is one thousand two hundred and fifty naira fifty kobo.", allowed, verified=True) is None
    assert check_output("That is one thousand two hundred and fifty naira, forty kobo.", allowed, verified=True) is not None


def test_the_readback_must_be_said_exactly():
    readback = f"Just to confirm: a promise to pay {AMOUNT} on {DATE}. Is that right?"
    assert check_output(readback, [readback], verified=True, must_say=readback) is None
    assert check_output("Okay, thank you.", [readback], verified=True, must_say=readback) == "say the read-back sentence exactly as given"
