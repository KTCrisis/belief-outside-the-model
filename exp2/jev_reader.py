"""Jev (TypeSafe AI, via Cloudflare Workers AI) as the hybrid's reader.

The hybrid's LLM has one job besides the table: read a returned text and say
how likely the signal is present (diagnosis) or what a source asserts and
where it got it (rumor). Jev answers exactly this kind of narrow, closed
question, with a probability, in one pass. The token and account come from
the environment, never from a file.
"""
import json
import os
import random
import threading
import time
from pathlib import Path

import httpx

URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run"
PRICE_IN = 0.042 / 1e6  # dollars per input token; output is free

_client = httpx.Client(timeout=30)
_lock = threading.Lock()
usage = {"calls": 0, "tokens": 0, "errors": 0, "cached": 0}

# Every answer is kept, keyed by its request: a run stopped by the rate limit
# resumes without paying twice, and a rerun on the same texts reads the same.
CACHE = Path(__file__).resolve().parent / "results" / "jev-cache.jsonl"
_cache: dict[str, dict] = {}
if CACHE.exists():
    for line in CACHE.read_text().splitlines():
        k, v = json.loads(line)
        _cache[k] = v


def ask(state: dict, questions: dict, retries: int = 10) -> dict:
    """Jev's answers to `questions` about `state`, unwrapped from Cloudflare's envelope."""
    key = json.dumps([state, questions], sort_keys=True)
    with _lock:
        if key in _cache:
            usage["cached"] += 1
            return _cache[key]
    url = URL.format(account=os.environ["CLOUDFLARE_ACCOUNT_ID"])
    headers = {"Authorization": f"Bearer {os.environ['CLOUDFLARE_WORKERS_AI_TOKEN']}"}
    for attempt in range(retries):
        r = _client.post(url, headers=headers, json={"model": "typesafe/jev", "input": {"state": state, "questions": questions}})
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
            continue
        r.raise_for_status()
        data = r.json()
        for _ in range(3):  # {"result": {"result": {...answers...}}} on Cloudflare
            if "answers" in data:
                break
            data = data.get("result", {})
        with _lock:
            usage["calls"] += 1
            usage["tokens"] += int((data.get("usage") or {}).get("input_tokens", 0) or 0)
            _cache[key] = data["answers"]
            with CACHE.open("a") as f:
                f.write(json.dumps([key, data["answers"]]) + "\n")
        return data["answers"]
    with _lock:
        usage["errors"] += 1
    raise RuntimeError(f"Jev: HTTP {r.status_code} after {retries} tries")


def clip(p: float) -> float:
    return min(max(float(p), 0.01), 0.99)


def read_signal(check: str, text: str, signal: str, absence: str | None = None) -> float:
    """Diagnosis: P(the returned text shows the signal).

    Without `absence`, the false side was generic ("reports its absence") and Jev
    read a double-negative positive as negative on 6 % of the box's reads
    (2026-10-04); with both sides spelled out, each criterion is a plain fact.
    """
    false = ("The returned text says: " + absence) if absence else "The returned text reports its absence, or says nothing about it"
    a = ask({"check": check, "returned": text},
            {"signal": {"type": "noul",
                        "instructions": "The text returned by the check shows this signal: " + signal,
                        "criteria": {"true": "The returned text says: " + signal, "false": false}}})
    return clip(a["signal"]["noul"])


def read_report(claim: str, source: str, what: str, text: str) -> tuple[float, str]:
    """Rumor: P(the source asserts the claim), and whether it relays outlet A or reports on its own."""
    a = ask({"claim": claim, "source": f"{source} ({what})", "says": text},
            {"asserts": {"type": "noul",
                         "instructions": "This source asserts that the claim is true.",
                         "criteria": {"true": "It confirms or states the claim",
                                      "false": "It denies the claim or says it is not true"}},
             "origin": {"type": "choice",
                        "instructions": "Where did this source get the information?",
                        "criteria": {"relays": "It relays, cites or reprints outlet A or another media report published before it",
                                     "own": "Its own statement, knowledge or reporting"}}})
    probs = a["origin"]["probabilities"]
    return clip(a["asserts"]["noul"]), ("article_a" if probs.get("relays", 0) > probs.get("own", 0) else source)


def estimate_cell(cause: str, meaning: str, check: str, positive: str, negative: str) -> float:
    """Table: P(the check comes out positive | the cause is the true one), Jev's probability."""
    a = ask({"true_cause_of_the_incident": f"{cause}: {meaning}", "check_run": check},
            {"positive": {"type": "noul",
                          "instructions": "Given that this cause is the true one, the check will show: " + positive,
                          "criteria": {"true": "The check shows: " + positive, "false": "The check shows: " + negative}}})
    return clip(a["positive"]["noul"])
