"""Readers of study 2 (PREREGISTRATION-2.md, sections 4 and 5).

Seven readers, one question per check, the questions of study 1 with the
consumer_cpu_profile question made unambiguous:
- reference: the deterministic rules of study 1 (read.py), unchanged;
- jev: Jev, as in study 1;
- haiku: Haiku narrow, as in study 1, median of 5;
- routed: Jev on the checks whose verdict is read off the text; on the checks
  whose verdict is derived, Haiku copies the values (three extractions) and a
  comparison stated in the preregistration decides; the median verdict is kept;
- sonnet_narrow: Sonnet with Haiku's narrow prompt, median of 3;
- haiku_context: the situation (alert, causes, checks) then the narrow
  question; also names the cause the output alone makes most likely; median of 5;
- haiku_seq: one conversation, the narrow question first (recorded as
  haiku_seq_blind), then the situation and a leading question, "do you
  revise?" (haiku_seq); median of 5 conversations;
- haiku_seqn: the same with a neutral question, "give your probability
  again; it may stay the same" (haiku_seqn);
- routed_h55 (exploratory, added 2026-10-09, not preregistered): the routed
  reader with Haiku 5.5 doing the extractions, thinking off, all else equal;
- routed_oss (exploratory, same day): the same with gpt-oss 20B through Ollama,
  locally, reasoning low, output constrained to the same JSON schema.

Each line of results/readings2-<tag>.jsonl is one (episode, check) with the
readers asked; the run resumes, and pairs are read in parallel. The exploratory
readers (routed_h55, routed_oss) write to results/readings2x-<tag>.jsonl instead,
so the preregistered file stays as it was recorded.

usage: python read2.py --tag pilot --readers all            # trial, about 1 $
       python read2.py --tag study2 --readers all
       python read2.py --tag study2-shifted --readers reference,routed --shifted
"""
import argparse
import json
import re
import statistics
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "exp2"))
from read import QUESTIONS as STUDY1  # noqa: E402
from read import reference  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402

OUT = HERE / "results"
SC = SCENARIOS["kafka"]
CAUSES = SC["causes"]
HAIKU, SONNET, HAIKU55 = "claude-haiku-4-5", "claude-sonnet-5-5", "claude-haiku-5-5"
OSS = "ollama:gpt-oss:20b"
OLLAMA = "http://localhost:11434"
PRICE = {HAIKU: (1.0 / 1e6, 5.0 / 1e6), SONNET: (2.0 / 1e6, 10.0 / 1e6),
         HAIKU55: (0.10 / 1e6, 0.50 / 1e6)}  # dollars per token in, out (Haiku 5.5: prompts of 100K or fewer)

# section 2: the questions of study 1, consumer_cpu_profile made unambiguous
QUESTIONS = dict(STUDY1)
QUESTIONS["consumer_cpu_profile"] = (
    STUDY1["consumer_cpu_profile"][0],
    "at least two consumers are at 80 % CPU or more with their main thread in the message processing method "
    "enrichOrder (a cache refresh does not count)",
    "fewer than two consumers are at 80 % CPU or more with their main thread in the message processing method "
    "enrichOrder (a cache refresh does not count)")

# section 7: the questions of the shifted copy, in its units
SHIFTED = dict(QUESTIONS)
SHIFTED["producer_rate_graph"] = (
    "the incoming messages per minute on the topic, per partition, over 10 seconds",
    "the incoming rate is clearly above its usual level of about 21,000 messages per minute",
    "the incoming rate is at its usual level of about 21,000 messages per minute")
SHIFTED["consumer_cpu_profile"] = (
    "each consumer's CPU use over 5 seconds, as a fraction of one CPU, and where its main thread is",
    "at least two consumers use 0.80 of a CPU or more with their main thread in the message processing method "
    "enrichOrder (a cache refresh does not count)",
    "fewer than two consumers use 0.80 of a CPU or more with their main thread in the message processing method "
    "enrichOrder (a cache refresh does not count)")

READ_OFF = ["rebalance_logs", "jvm_gc_logs", "broker_quota_metrics"]
DERIVED = ["restart_consumer", "partition_lag_spread", "producer_rate_graph", "consumer_cpu_profile"]
USUAL_RATE = 350.0          # messages per second, stated in the question
PRESENT, ABSENT, FAILED = 0.98, 0.02, 0.5

_client = None
_lock = threading.Lock()


def client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(max_retries=8)
    return _client


def call(model: str, messages: list, schema: dict, system: str | None = None, max_tokens: int = 300) -> tuple[dict, float]:
    """One structured answer and its price in dollars."""
    if model.startswith("ollama:"):  # local: Ollama's /api/chat, JSON schema as format, no price
        import httpx
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        r = httpx.post(f"{OLLAMA}/api/chat", timeout=600, json={
            "model": model.removeprefix("ollama:"), "messages": msgs, "format": schema, "stream": False,
            "think": "low", "options": {"num_predict": max(max_tokens, 8000)}}).json()  # reasoning counts too
        return json.loads(r["message"]["content"]), 0.0
    kw = {"system": system} if system else {}
    if model == HAIKU55:  # thinking is on by default for Haiku 5.5; off here, as Haiku 4.5 copied without it
        kw["thinking"] = {"type": "disabled"}
    r = client().messages.create(model=model, max_tokens=max_tokens, messages=messages,
                                 output_config={"format": {"type": "json_schema", "schema": schema}}, **kw)
    pin, pout = PRICE[model]
    usd = r.usage.input_tokens * pin + r.usage.output_tokens * pout
    return json.loads(next(b.text for b in r.content if b.type == "text")), usd


def obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


PROB = obj({"probability": {"type": "number"}})
PROB_CAUSE = obj({"probability": {"type": "number"}, "most_likely_cause": {"type": "string", "enum": CAUSES}})


def clip(p: float) -> float:
    return min(max(float(p), 0.01), 0.99)


def narrow_prompt(check: str, text: str, q: dict) -> str:
    """Haiku's prompt of study 1, word for word."""
    what, signal, absence = q[check]
    return (f"An on-call engineer ran a check on a Kafka platform. The check shows {what}.\n"
            f"It returned:\n\"\"\"\n{text}\n\"\"\"\n\n"
            f"What is the probability that this output shows the signal: {signal}? "
            f"(The opposite would be: {absence}.)")


def situation(q: dict) -> str:
    """The situation part of Sonnet's brief: no cost, no price of an error, no goal (arm 7)."""
    causes = "\n".join(f"- {c}: {SC['meaning'][c]}" for c in CAUSES)
    checks = "\n".join(f"- {t}: shows {q[t][0]}" for t in q)
    return ("You are the on-call engineer for a Kafka platform. An alert reports that the consumer group "
            "'lab' is lagging behind on topic 'orders'. In normal operation the order service sends about "
            f"350 messages per second.\n\nPossible causes, exactly one is true:\n{causes}\n\n"
            f"Available checks:\n{checks}")


NAME_CAUSE = "Also name the cause that this output, on its own, makes most likely."


# --------------------------------------------------------------------------- #
#  readers
# --------------------------------------------------------------------------- #
def r_reference(check, text, q):
    return {"reference": reference(check, text)}


def r_jev(check, text, q):
    import jev_reader
    what, signal, absence = q[check]
    return {"jev": jev_reader.read_signal(what, text, signal, absence)}


def narrow(model, n, check, text, q):
    ps, usd = [], 0.0
    for _ in range(n):
        d, u = call(model, [{"role": "user", "content": narrow_prompt(check, text, q)}], PROB,
                    max_tokens=4000 if model == SONNET else 300)
        ps.append(float(d["probability"]))
        usd += u
    return clip(statistics.median(ps)), usd


def r_haiku(check, text, q):
    p, usd = narrow(HAIKU, 5, check, text, q)
    return {"haiku": p, "haiku_usd": usd}


def r_sonnet_narrow(check, text, q):
    p, usd = narrow(SONNET, 3, check, text, q)
    return {"sonnet_narrow": p, "sonnet_narrow_usd": usd}


def r_haiku_context(check, text, q):
    ps, cs, usd = [], [], 0.0
    for _ in range(5):
        d, u = call(HAIKU, [{"role": "user", "content": narrow_prompt(check, text, q) + "\n" + NAME_CAUSE}],
                    PROB_CAUSE, system=situation(q))
        ps.append(float(d["probability"]))
        cs.append(d["most_likely_cause"])
        usd += u
    return {"haiku_context": clip(statistics.median(ps)), "haiku_context_cause": Counter(cs).most_common(1)[0][0],
            "haiku_context_usd": usd}


REVISE = {  # arm 8 asks a leading question, arm 9 a neutral one (pilot, 2026-10-06)
    "haiku_seq": "Given this situation, do you revise your probability that the output shows the signal? "
                 "Give your final probability. ",
    "haiku_seqn": "Now that you know the situation, give your probability that the output shows the signal again. "
                  "It may stay the same if the situation does not change what the output shows. ",
}


def sequential(name, check, text, q):
    blind, informed, cs, usd = [], [], [], 0.0
    for _ in range(5):
        first = [{"role": "user", "content": narrow_prompt(check, text, q)}]
        d1, u1 = call(HAIKU, first, PROB)
        msgs = first + [{"role": "assistant", "content": json.dumps(d1)},
                        {"role": "user", "content": (
                            f"Here is the situation in which the check was run.\n\n{situation(q)}\n\n"
                            + REVISE[name] + NAME_CAUSE)}]
        d2, u2 = call(HAIKU, msgs, PROB_CAUSE)
        blind.append(float(d1["probability"]))
        informed.append(float(d2["probability"]))
        cs.append(d2["most_likely_cause"])
        usd += u1 + u2
    return {f"{name}_blind": clip(statistics.median(blind)), name: clip(statistics.median(informed)),
            f"{name}_cause": Counter(cs).most_common(1)[0][0], f"{name}_usd": usd}


def r_haiku_seq(check, text, q):
    return sequential("haiku_seq", check, text, q)


def r_haiku_seqn(check, text, q):
    return sequential("haiku_seqn", check, text, q)


# --------------------------------------------------------------------------- #
#  the routed reader: extraction, then the comparisons of section 4
# --------------------------------------------------------------------------- #
EXTRACT = {
    "restart_consumer": (
        "the total lag before the restart, and the total lag 30 seconds after it (-1 if a value is absent)",
        obj({"lag_before": {"type": "number"}, "lag_30s_after": {"type": "number"}})),
    "partition_lag_spread": (
        "the lag of each partition",
        obj({"partitions": {"type": "array", "items": obj({"partition": {"type": "integer"}, "lag": {"type": "number"}})}})),
    "producer_rate_graph": (
        "the total incoming rate, as written, and its unit (-1 if absent)",
        obj({"total_rate": {"type": "number"},
             "unit": {"type": "string", "enum": ["messages per second", "messages per minute", "other"]}})),
    "consumer_cpu_profile": (
        "for each consumer: its CPU use as written, whether it is written as a percentage or as a fraction, and the "
        "first method listed in its main thread's stack (for example SomeClass.someMethod)",
        obj({"consumers": {"type": "array", "items": obj({
            "consumer": {"type": "string"}, "cpu": {"type": "number"},
            "cpu_written_as": {"type": "string", "enum": ["percentage", "fraction"]},
            "top_method": {"type": "string"}})}})),
}


def verdict(check: str, d: dict) -> float:
    """The comparisons of section 4, frozen in the preregistration."""
    if check == "restart_consumer":
        b, a = d["lag_before"], d["lag_30s_after"]
        if b < 0 or a < 0:
            return FAILED
        return PRESENT if a < b else ABSENT
    if check == "partition_lag_spread":
        lags = sorted((p["lag"] for p in d["partitions"]), reverse=True)
        if len(lags) < 2:
            return FAILED
        return PRESENT if lags[0] > 3 * statistics.median(lags[1:]) else ABSENT
    if check == "producer_rate_graph":
        rate, unit = d["total_rate"], d["unit"]
        if rate < 0 or unit == "other":
            return FAILED
        per_s = rate / 60 if unit == "messages per minute" else rate
        return PRESENT if per_s >= 1.5 * USUAL_RATE else ABSENT
    if check == "consumer_cpu_profile":
        cs = d["consumers"]
        if not cs:
            return FAILED
        busy = sum(1 for c in cs
                   if (c["cpu"] * 100 if c["cpu_written_as"] == "fraction" else c["cpu"]) >= 80
                   and c["top_method"].split("(")[0].endswith("enrichOrder"))
        return PRESENT if busy >= 2 else ABSENT
    raise ValueError(check)


def r_routed(check, text, q, model=HAIKU, name="routed"):
    if check in READ_OFF:
        return {name: r_jev(check, text, q)["jev"], f"{name}_failed": 0}
    what, schema = EXTRACT[check]
    vs, usd, failed = [], 0.0, 0
    for _ in range(3):
        prompt = (f"An on-call engineer ran a check on a Kafka platform. The check shows {q[check][0]}.\n"
                  f"It returned:\n\"\"\"\n{text}\n\"\"\"\n\n"
                  f"Copy from this output {what}. Copy the values exactly as written; do not interpret or judge them.")
        try:
            d, u = call(model, [{"role": "user", "content": prompt}], schema, max_tokens=1500)
            usd += u
            v = verdict(check, d)
        except (ValueError, KeyError, TypeError):
            v = FAILED
        failed += v == FAILED
        vs.append(v)
    return {name: statistics.median(vs), f"{name}_failed": failed, f"{name}_usd": usd}


def r_routed_h55(check, text, q):
    return r_routed(check, text, q, model=HAIKU55, name="routed_h55")


def r_routed_oss(check, text, q):
    return r_routed(check, text, q, model=OSS, name="routed_oss")


EXPLORATORY = {"routed_h55", "routed_oss"}

READERS = {"reference": r_reference, "jev": r_jev, "haiku": r_haiku, "routed": r_routed,
           "sonnet_narrow": r_sonnet_narrow, "haiku_context": r_haiku_context, "haiku_seq": r_haiku_seq,
           "haiku_seqn": r_haiku_seqn, "routed_h55": r_routed_h55,
           "routed_oss": r_routed_oss}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--readers", default="reference")
    ap.add_argument("--shifted", action="store_true", help="the shifted copy: questions in its units")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    readers = [r for r in READERS if r not in EXPLORATORY] if a.readers == "all" else a.readers.split(",")
    if any(r in EXPLORATORY for r in readers) and not all(r in EXPLORATORY for r in readers):
        sys.exit("exploratory and preregistered readers go in separate runs (separate files)")
    q = SHIFTED if a.shifted else QUESTIONS
    episodes = [json.loads(l) for l in (OUT / f"episodes-{a.tag}.jsonl").open()]
    path = OUT / (f"readings2x-{a.tag}.jsonl" if readers[0] in EXPLORATORY else f"readings2-{a.tag}.jsonl")
    done: dict = {}
    if path.exists():
        for l in path.open():
            r = json.loads(l)
            done.setdefault((r["episode"], r["check"]), {}).update(r)
    jobs = []
    for e in episodes:
        for check, text in e["probes"].items():
            row = done.get((e["episode"], check), {"episode": e["episode"], "cause": e["cause"], "check": check})
            missing = [r for r in readers if r not in row]
            if missing:
                jobs.append((row, check, text, missing))

    def one(job):
        row, check, text, missing = job
        for r in missing:
            try:
                row.update(READERS[r](check, text, q))
            except Exception as err:  # noqa: BLE001 — a reader that fails leaves its field absent; the run resumes
                print(f"{row['episode']} {check} {r}: {err}", file=sys.stderr)
        with _lock, path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, jobs))
    rows = {}
    for l in path.open():
        r = json.loads(l)
        rows.setdefault((r["episode"], r["check"]), {}).update(r)
    usd = sum(v for r in rows.values() for k, v in r.items() if k.endswith("_usd"))
    print(f"{path.name}: {len(rows)} pairs, readers {readers}, {usd:.2f} $ of Claude readings")


if __name__ == "__main__":
    main()
