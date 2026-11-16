"""The agent, in one file. Read it top to bottom.

A call is a conversation where the model can only act through a few tools, and the tools it can see
depend on the state of the call. The model chats and picks tools. Everything that matters (who may see
what, what may be saved, what may be said) is decided by plain code here and in policy.py and
output_check.py, never by the prompt alone.

    call = start_call(db, "B001")
    print(greet(call, db, chat))                # the agent speaks first
    print(turn(call, db, "Hello?", chat))       # one borrower sentence in, one agent reply out
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import httpx

from engine import lender, policy
from engine.output_check import check_output, norm
from engine.prompts import PROMPT_VERSION, agent_prompt
from engine.speech import date_to_words, naira_to_words, number_to_words

CFG = policy.load_policy()
LAGOS = ZoneInfo(CFG.calling_window.timezone)
SAFE_LINE = "Sorry, I am having trouble. A colleague will call you back. Goodbye."
MAX_STEPS = 6  # model calls allowed for one borrower sentence

# ---------------------------------------------------------------- the six states
# Each state shows the model only these tools. Code moves the call from state to state.
STATES = {
    "verify": ["verify_identity", "opt_out", "hand_off", "end_call"],
    "explain": ["get_loan", "get_offers", "opt_out", "hand_off"],
    "arrange": ["get_offers", "propose_option", "opt_out", "hand_off"],
    "confirm": ["confirm_option", "opt_out", "hand_off"],
    "close": ["end_call"],
    "handoff": ["hand_off", "end_call"],
}
OPTIONS = ["pay_now", "part_payment", "extension", "promise", "callback"]
OUTCOME_OF = {
    "pay_now": "paid_now",
    "part_payment": "part_payment_agreed",
    "extension": "extension_granted",
    "promise": "promise_captured",
    "callback": "callback_scheduled",
}
OUTCOMES = [
    "paid_now",
    "promise_captured",
    "part_payment_agreed",
    "extension_granted",
    "callback_scheduled",
    "refused",
    "opted_out",
    "wrong_person",
    "verification_failed",
    "no_answer",
    "handed_off",
    "technical_failure",
]
HANDOFF_REASONS = [
    "hardship",
    "dispute",
    "abuse",
    "distress",
    "requested_human",
    "repeated_misunderstanding",
    "tool_failure",
    "policy_conflict",
]


def tool(name, description, props=None, required=()):
    properties = {k: {"type": t, "description": d} for k, (t, d) in (props or {}).items()}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(required)},
        },
    }


TOOLS = {
    t["function"]["name"]: t
    for t in [
        tool(
            "verify_identity",
            "Check the caller's answer to the identity question (their day and month of birth).",
            {"answer": ("string", "exactly what the caller said")},
            ["answer"],
        ),
        tool("get_loan", "Get the amount due and the due date. Only after the caller is verified."),
        tool("get_offers", "Get what can be offered: pay now, part payment, extension, promise date."),
        tool(
            "propose_option",
            "Propose one option. Code checks the limits and writes the sentence you must read back.",
            {
                "option": ("string", f"one of {OPTIONS}"),
                "amount_ngn": ("number", "amount in naira, if the option has one"),
                "date": ("string", "YYYY-MM-DD, for an extension or a promise"),
            },
            ["option"],
        ),
        tool(
            "confirm_option",
            "Record the caller's answer to the read-back. Only after they answered.",
            {"accepted": ("boolean", "true only for a clear yes")},
            ["accepted"],
        ),
        tool(
            "hand_off",
            "Pass the call to a human.",
            {
                "reason": ("string", f"one of {HANDOFF_REASONS}"),
                "summary": ("string", "one or two plain sentences"),
            },
            ["reason", "summary"],
        ),
        tool("opt_out", "The caller does not want to be called again. Final."),
        tool("end_call", "End the call.", {"outcome": ("string", f"one of {OUTCOMES}")}, ["outcome"]),
    ]
}

# ---------------------------------------------------------------- the state of one call


@dataclass
class Call:
    id: str
    borrower: dict
    loan: dict | None
    today: date
    state: str = "verify"
    verified: bool = False
    locked: bool = False
    failures: int = 0
    offers: policy.Offers | None = None
    pending: dict | None = None  # the option waiting for a yes
    readback: str = ""  # the sentence the agent must say next, written by code
    facts: list = field(default_factory=list)  # every sentence code produced: the only amounts and dates allowed
    agent_turns: int = 0
    user_turns: int = 0  # both counted since propose_option, so a yes cannot be assumed
    outcome: str | None = None
    done: bool = False
    messages: list = field(default_factory=list)


def start_call(db, borrower_id, today=None, check_dialing=True, now=None):
    borrower = lender.get_borrower(db, borrower_id)
    if borrower is None:
        raise LookupError(f"no borrower {borrower_id}")
    today = today or datetime.now(LAGOS).date()
    if check_dialing:  # rules 1 to 3. The scheduler normally does this; here it is a last check.
        verdicts = policy.can_dial(
            calls_allowed=borrower["calls_allowed"],
            opted_out=borrower["opted_out"],
            previous_attempts=[],
            now=now or datetime.now(LAGOS),
            cfg=CFG,
        )
        if denied := policy.first_denial(verdicts):
            raise PermissionError(f"not allowed to call: {denied.rule} {denied.reason}")
    call = Call(lender.new_call(db, borrower_id, PROMPT_VERSION), borrower, lender.get_loan(db, borrower_id), today)
    call.messages = [{"role": "system", "content": agent_prompt(borrower["full_name"], "Sample Lender", today)}]
    lender.log_event(db, call.id, "call_started", {"borrower": borrower_id, "state": call.state})
    return call


# ---------------------------------------------------------------- reading what the caller said


def parse_day_month(text):
    """'14 March', 'the fourteenth of March', 'March 14th', '14/03' -> '14-03'. None if unclear."""
    t = re.sub(r"[-,]", " ", text.lower())
    if m := re.search(r"\b(\d{1,2})\s*[/.]\s*(\d{1,2})\b", t):
        day, month = int(m[1]), int(m[2])
    else:
        names = "january february march april may june july august september october november december".split()
        words = re.findall(r"[a-z]+", t)
        month = next((i + 1 for i, n in enumerate(names) if n in words or n[:3] in words), None)
        day = next((int(n) for n in re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", t)), None)
        if day is None:  # spoken numbers: 'fourteen', 'twenty fifth', 'the first'
            spoken = {number_to_words(n).replace("-", " "): n for n in range(1, 32)}
            spoken |= {date_to_words(date(2026, 1, n)).split("the ")[1].split(" of ")[0].replace("-", " "): n for n in range(1, 32)}
            found = [p for p in spoken if re.search(rf"\b{p}\b", t)]
            day = spoken[max(found, key=len)] if found else None
    if not day or not month or not (1 <= day <= 31 and 1 <= month <= 12):
        return None
    return f"{day:02d}-{month:02d}"


TRIGGERS = [  # words that mean the same thing in any state. Code acts at once, the model is not asked.
    (
        r"stop calling|don'?t call|do not call|never call|remove my number|take me off",
        "opt_out",
        "Understood. I will not call you again. Goodbye.",
    ),
    (
        r"(speak|talk) (to|with) (a |an )?(real )?(person|human|someone|agent)|real person|human being",
        "requested_human",
        "Of course. A colleague will call you back. Goodbye.",
    ),
]

# ---------------------------------------------------------------- the tools. Each one returns a small dict.


def t_verify_identity(call, db, answer):
    if parse_day_month(answer) == call.borrower["birth_day_month"]:
        call.verified, call.state = True, "explain"
        return {"status": "verified"}
    call.failures += 1
    if call.failures >= 2:
        call.locked, call.state, call.outcome = True, "close", "verification_failed"
        return {"status": "locked", "attempts_left": 0}
    return {"status": "failed", "attempts_left": 2 - call.failures}


def t_get_loan(call, db):
    loan = call.loan
    if not loan:
        return {"error": "no loan found"}
    days = (loan["due_date"] - call.today).days
    timing = "due today" if days == 0 else f"due in {number_to_words(days)} days" if days > 0 else f"{number_to_words(-days)} days overdue"
    out = {
        "amount_due": naira_to_words(loan["amount_due_ngn"]),
        "due_date": date_to_words(loan["due_date"]),
        "timing": timing,
        "recent_payments": [
            f"{naira_to_words(p['amount_ngn'])} on {date_to_words(p['paid_on'])}" for p in lender.recent_payments(db, loan["id"])
        ],
    }
    call.facts += [out["amount_due"], out["due_date"], timing, *out["recent_payments"]]
    return out


def t_get_offers(call, db):
    if not call.loan:
        return {"error": "no loan found"}
    call.offers = o = policy.compute_offers(Decimal(call.loan["amount_due_ngn"]), call.loan["due_date"], call.today, CFG)
    out = {
        "pay_now": naira_to_words(o.pay_now_ngn),
        "extension_max_days": number_to_words(o.extension_max_days),
        "promise_latest": date_to_words(o.promise_latest),
        "part_payment": None,
    }
    if o.part_payment:
        out["part_payment"] = {
            "minimum": naira_to_words(o.part_payment.min_ngn),
            "maximum": naira_to_words(o.part_payment.max_ngn),
        }
        call.facts += [out["part_payment"]["minimum"], out["part_payment"]["maximum"]]
    call.facts += [out["pay_now"], out["promise_latest"]]
    if call.state == "explain":
        call.state = "arrange"
    return out


def describe(option, amount, on):
    return {
        "pay_now": f"pay {naira_to_words(amount) if amount else ''} now",
        "part_payment": f"pay {naira_to_words(amount) if amount else ''} now as a part payment",
        "extension": f"an extension until {date_to_words(on) if on else ''}",
        "promise": f"a promise to pay {naira_to_words(amount) if amount else ''} on {date_to_words(on) if on else ''}",
        "callback": "a call back from a colleague",
    }[option]


def t_propose_option(call, db, option, amount_ngn=None, date=None):
    if call.offers is None:
        return {"error": "call get_offers first"}
    try:
        amount = Decimal(str(amount_ngn)).quantize(Decimal("0.01")) if amount_ngn is not None else None
        on = datetime.strptime(date, "%Y-%m-%d").date() if date else None
    except (ValueError, ArithmeticError):
        return {"error": "amount must be a number and date must be YYYY-MM-DD"}
    if option in ("pay_now", "promise") and amount is None:
        amount = call.offers.pay_now_ngn  # the full amount unless the caller named another
    verdict = policy.validate_option(option, amount, on, call.offers)
    lender.log_event(db, call.id, "policy", verdict.to_event())
    if not verdict.allowed:
        return {"rejected": verdict.reason, **verdict.detail}
    call.pending = {"option": option, "amount": amount, "date": on}
    call.readback = f"Just to confirm: {describe(option, amount, on)}. Is that right?"
    call.facts.append(call.readback)
    call.state, call.agent_turns, call.user_turns = "confirm", 0, 0
    return {"readback_text": call.readback}


def t_confirm_option(call, db, accepted):
    if call.agent_turns < 1 or call.user_turns < 1:
        return {"error": "say the read-back sentence and wait for the caller to answer first"}
    if not accepted:
        call.pending, call.readback, call.state = None, "", "arrange"
        return {"result": "nothing saved, back to choosing"}
    p = call.pending
    if p is None or call.loan is None:
        return {"error": "nothing is waiting for confirmation"}
    key = hashlib.sha256(f"{call.id}|{p['option']}|{p['amount']}|{p['date']}".encode()).hexdigest()
    if p["option"] == "callback":
        tomorrow = datetime.combine(call.today + timedelta(days=1), CFG.calling_window.start, LAGOS)
        lender.schedule_callback(db, call.borrower["id"], tomorrow)
    else:
        lender.save_promise(db, call.id, call.loan["id"], p["option"], p["amount"], p["date"], key)
    link = p["option"] in ("pay_now", "part_payment")
    if link:  # the payment link comes from the lender, never from the model. Here it is a stub.
        lender.log_event(db, call.id, "payment_link_sent", {"amount": p["amount"]})
    call.outcome, call.state, call.pending = OUTCOME_OF[p["option"]], "close", None
    return {"saved": p["option"], "payment_link_sent": link}


def t_hand_off(call, db, reason, summary):
    if reason not in HANDOFF_REASONS:
        return {"error": f"reason must be one of {HANDOFF_REASONS}"}
    call.state, call.outcome = "handoff", "handed_off"
    return {"handoff_id": lender.save_handoff(db, call.id, reason, summary)}


def t_opt_out(call, db):
    lender.set_opted_out(db, call.borrower["id"])
    call.state, call.outcome = "close", "opted_out"
    return {"ok": True}


def t_end_call(call, db, outcome):
    if outcome not in OUTCOMES:
        return {"error": f"outcome must be one of {OUTCOMES}"}
    call.outcome = call.outcome or outcome  # what code recorded wins over what the model says
    call.done = True
    lender.end_call(db, call.id, call.outcome)
    return {"ok": True}


FUNCS = {name: globals()["t_" + name] for name in TOOLS}


def run_tool(call, db, name, args):
    """The one door every tool call goes through: is it shown in this state, does policy allow it, then run it."""
    started = time.monotonic()
    if name not in STATES[call.state] or name not in TOOLS:
        result = {"error": f"{name} is not available now"}
    elif not (verdict := policy.check_tool_precondition(name, verified=call.verified, locked=call.locked)).allowed:
        lender.log_event(db, call.id, "policy", verdict.to_event())
        result = {"error": "not allowed", "reason": verdict.reason}
    else:
        wanted = TOOLS[name]["function"]["parameters"]["properties"]
        try:
            result = FUNCS[name](call, db, **{k: v for k, v in args.items() if k in wanted})  # arguments the model invented are dropped
        except TypeError:
            result = {"error": "missing or wrong arguments"}
    lender.log_event(
        db,
        call.id,
        "tool",
        {"name": name, "args": args, "result": result},
        round((time.monotonic() - started) * 1000),
    )
    return result


# ---------------------------------------------------------------- the conversation loop


def turn(call, db, user_text, chat):
    """One borrower sentence in, one agent reply out. `chat(messages, tools)` is the model."""
    lender.log_event(db, call.id, "user", {"text": user_text})
    call.user_turns += 1
    for pattern, reason, line in TRIGGERS:
        if re.search(pattern, user_text.lower()) and call.state not in ("close", "handoff"):
            if reason == "opt_out":
                run_tool(call, db, "opt_out", {})
            else:
                run_tool(call, db, "hand_off", {"reason": reason, "summary": "The caller asked for a person."})
            return finish(call, db, line)
    call.messages.append({"role": "user", "content": user_text})
    return respond(call, db, chat)


def greet(call, db, chat):
    """The agent speaks first, when the call connects."""
    call.messages.append({"role": "user", "content": "[The call just connected. Greet the person now.]"})
    return respond(call, db, chat)


def respond(call, db, chat):
    failed = 0
    for _ in range(MAX_STEPS):
        started = time.monotonic()
        msg = chat(call.messages, [TOOLS[n] for n in STATES[call.state]])
        lender.log_event(
            db,
            call.id,
            "model",
            {"tools": [c["function"]["name"] for c in msg.get("tool_calls") or []]},
            round((time.monotonic() - started) * 1000),
        )
        if msg.get("tool_calls"):
            call.messages.append(msg)
            for c in msg["tool_calls"]:
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = run_tool(call, db, c["function"]["name"], args)
                call.messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result, default=str)})
            continue
        reply = (msg.get("content") or "").strip()
        if problem := check_output(reply, call.facts, call.verified, call.readback):
            lender.log_event(db, call.id, "output_check", {"problem": problem})
            failed += 1
            if failed == 2:
                break  # two bad replies in a row: stop trusting the model on this call
            call.messages.append(
                {
                    "role": "user",
                    "content": f"[Your last reply broke a rule: {problem}. Say it again correctly, in two short sentences.]",
                }
            )
            continue
        call.messages.append({"role": "assistant", "content": reply})
        call.agent_turns += 1
        if call.readback and norm(call.readback) in norm(reply):
            call.readback = ""
        lender.log_event(db, call.id, "agent", {"text": reply})
        return reply
    if call.state != "handoff":
        run_tool(
            call,
            db,
            "hand_off",
            {"reason": "policy_conflict", "summary": "The agent could not produce a safe reply."},
        )
    return finish(call, db, SAFE_LINE)


def finish(call, db, line):
    """Say a fixed line chosen by code and close the call."""
    call.done = True
    lender.end_call(db, call.id, call.outcome or "technical_failure")
    lender.log_event(db, call.id, "agent", {"text": line})
    return line


def make_chat(url, key, model, patience_s=1800, extra_headers=None):
    """The model: any OpenAI-style server. A cold Modal server answers 503 for a few minutes, so we wait."""
    client = httpx.Client(timeout=120, headers=extra_headers or {})

    def chat(messages, tools):
        deadline = time.time() + patience_s
        while True:
            try:
                r = client.post(
                    f"{url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": model,
                        "messages": messages,
                        "tools": tools,
                        "temperature": 0,
                        "max_tokens": 300,
                    },
                )
            except httpx.TimeoutException:
                r = None
            if r is not None and r.status_code == 200:
                return r.json()["choices"][0]["message"]
            if (r is not None and r.status_code not in (502, 503, 504)) or time.time() > deadline:
                raise RuntimeError(f"model server answered {r.status_code if r is not None else 'nothing'}")
            time.sleep(10)

    return chat
