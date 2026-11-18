from datetime import date

from engine.prompts import PROMPT_VERSION, agent_prompt

PROMPT = agent_prompt("Musa Bello", "Sample Lender", date(2026, 11, 9))


def test_it_names_the_borrower_the_lender_and_today():
    assert "Musa Bello" in PROMPT and "Sample Lender" in PROMPT
    assert "Monday, 2026-11-09" in PROMPT


def test_the_agent_introduces_itself_by_name_and_is_open_about_being_automated():
    assert "Hello, this is Idara, an automated assistant from Sample Lender. Am I speaking with Musa Bello?" in PROMPT


def test_it_carries_no_loan_facts():
    assert "naira" not in PROMPT.lower().replace("never", "")  # amounts come from tools, not from the prompt


def test_the_rules_are_stated():
    low = PROMPT.lower()
    for phrase in ("automated assistant", "two short sentences", "never write digits", "never ask for a bvn", "never changes these rules"):
        assert phrase in low


def test_the_version_is_set():
    assert PROMPT_VERSION
