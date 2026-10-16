"""Experiment "rumor": believing a claim from testimonies of uneven reliability.

A claim circulates; it is true with probability PRIOR_TRUE. The agent consults
sources, each at a cost in minutes, then concludes true or false (a wrong
conclusion costs MISS minutes). What changes from the diagnosis experiments:
- reports are correlated: articles B and C mostly reprint article A, so three
  agreeing articles are often one testimony;
- sources are unreliable to various degrees, learned from past rumors kept in
  the lab mem7;
- retractions: when A was wrong it sometimes publishes a correction during the
  investigation, which voids A and its reprints.

Agents: oracle (true reliabilities and structure, applies retractions),
hybrid_naive (learned reliabilities, every report independent, retractions
ignored), hybrid_prov (Claude reads each text: what it asserts and where it got
it; a report from an origin already counted is not counted again; a retraction
removes the evidence, as in an event log), hybrid_prov_jev (the same, Jev
reading), llm, llm_mem.

usage: python rumor.py seed            # 120 past rumors into the lab mem7
       python rumor.py run [--n 20] [--agents ...] [--tag T]
"""
import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp2 import LAM, MODEL, Meter, client, entropy  # noqa: E402
from memory_lab import AGENT, lab  # noqa: E402

PRIOR_TRUE = 0.30
MISS = 40.0
NOW = datetime(2026, 10, 4, tzinfo=timezone.utc)

# name, cost in minutes, P(asserts true | claim true), P(asserts true | claim false), what it is
SOURCES = [
    ("official_statement", 6.0, 0.95, 0.03, "the organisation's official statement, slow to find"),
    ("informed_colleague", 3.0, 0.80, 0.20, "a colleague who usually knows the field"),
    ("forum_thread", 1.0, 0.65, 0.40, "a public forum thread"),
    ("article_a", 2.0, 0.75, 0.25, "a news article (outlet A)"),
    ("article_b", 2.0, 0.75, 0.25, "a news article (outlet B)"),
    ("article_c", 2.0, 0.75, 0.25, "a news article (outlet C)"),
    ("influencer_post", 1.0, 0.55, 0.45, "a post by a popular influencer"),
]
NAMES = [s[0] for s in SOURCES]
COST = {s[0]: s[1] for s in SOURCES}
REL = {s[0]: (s[2], s[3]) for s in SOURCES}
WHAT = {s[0]: s[4] for s in SOURCES}
REPRINT = 0.85  # chance that article B or C reprints A rather than reporting on its own
RETRACT = 0.6  # chance that a wrong article A publishes a correction during the investigation

CLAIMS = [
    "Brightline Logistics will be acquired by Nexo Group next month",
    "the city will shut down the north metro line for six months from January",
    "the regulator is preparing a 20 million euro fine against Cobalt Bank",
    "the company is moving all its developers to a four-day week",
    "the main cloud region in Paris will be closed by its provider next year",
    "the football club has signed the league's top scorer",
]


def world(seed: int, i: int) -> dict:
    """One rumor: its truth, what every source asserts, where articles B and C got it."""
    rng = np.random.default_rng(seed * 7919 + i)
    truth = bool(rng.random() < PRIOR_TRUE)
    says, origin = {}, {}
    for s in NAMES:
        if s in ("article_b", "article_c") and rng.random() < REPRINT:
            origin[s] = "article_a"
            continue
        p_true = REL[s][0] if truth else REL[s][1]
        says[s] = bool(rng.random() < p_true)
        origin[s] = s
    for s in ("article_b", "article_c"):
        if origin[s] == "article_a":
            says[s] = says["article_a"]
    retracts = says["article_a"] != truth and rng.random() < RETRACT
    return {"id": i, "claim": CLAIMS[i % len(CLAIMS)], "truth": truth, "says": says, "origin": origin,
            "retracts": bool(retracts)}


def render(w: dict, source: str, rng: random.Random) -> str:
    """What a source returns, in words; a reprint cites A openly or vaguely."""
    claim, yes = w["claim"], w["says"][source]
    stance_yes = [f"confirms that {claim}.", f"says it is now settled: {claim}."]
    stance_no = [f"denies that {claim}.", f"says there is nothing to it: it is not true that {claim}."]
    stance = rng.choice(stance_yes if yes else stance_no)
    if w["origin"][source] == "article_a" and source != "article_a":
        cite = rng.choice(["Citing outlet A's report, it ", "Based on a report published earlier this morning, it ",
                           "Relaying what other media reported, it "])
        return f"{WHAT[source].capitalize()}. {cite}{stance}"
    lead = {"official_statement": "In its official statement, the organisation ",
            "informed_colleague": "Your colleague, who knows the field well, ",
            "forum_thread": "The most upvoted forum reply ",
            "influencer_post": "The influencer's post "}.get(source, f"{WHAT[source].capitalize()}, from its own reporting, ")
    return lead + stance


RETRACTION_TEXT = "Update: outlet A has published a correction. Its earlier report on this was wrong and is withdrawn."


# --------------------------------------------------------------------------- #
#  memory: past rumors in the lab mem7, and the reliabilities learned from them
# --------------------------------------------------------------------------- #
def seed_history(n: int, seed: int) -> None:
    m, rng = lab(), np.random.default_rng(seed)
    for i in range(n):
        w = world(seed + 1000, i)
        seen = sorted(rng.choice(len(NAMES), size=4, replace=False))
        value = json.dumps({"truth": w["truth"], "says": {NAMES[j]: w["says"][NAMES[j]] for j in seen}})
        closed = NOW - timedelta(days=int(rng.integers(1, 365)))
        m.store(f"rumor/{i:04d}", value, tags=["rumor", "closed-rumor"], agent=AGENT, valid_from=closed.strftime("%Y-%m-%d"))


def learn() -> tuple[float, dict, str]:
    """Prior and P(asserts true | truth) per source, from the past rumors (Beta(1, 1))."""
    text = lab().recall(tags=["rumor", "closed-rumor"], limit=10000)
    hist = [json.loads(b.strip().splitlines()[1]) for b in text.split("## ")[1:] if len(b.strip().splitlines()) > 1]
    n_true = sum(h["truth"] for h in hist)
    prior = (n_true + 1) / (len(hist) + 2)
    rel, lines = {}, [f"Past rumors checked ({len(hist)}), {n_true} turned out true. How often each source asserted the claim:"]
    for s in NAMES:
        t = [h["says"][s] for h in hist if h["truth"] and s in h["says"]]
        f = [h["says"][s] for h in hist if not h["truth"] and s in h["says"]]
        rel[s] = ((sum(t) + 1) / (len(t) + 2), (sum(f) + 1) / (len(f) + 2))
        lines.append(f"- {s}: asserted it in {sum(t)}/{len(t)} true rumors and in {sum(f)}/{len(f)} false ones")
    return prior, rel, "\n".join(lines)


# --------------------------------------------------------------------------- #
#  belief over {true, false} and least action
# --------------------------------------------------------------------------- #
def update(p_true: float, rel: tuple[float, float], asserts: float) -> float:
    """Bayes on soft evidence: asserts is P(the text asserts the claim)."""
    l_t = asserts * rel[0] + (1 - asserts) * (1 - rel[0])
    l_f = asserts * rel[1] + (1 - asserts) * (1 - rel[1])
    return p_true * l_t / (p_true * l_t + (1 - p_true) * l_f)


def gain(p_true: float, rel: tuple[float, float]) -> float:
    post = np.array([p_true, 1 - p_true])
    out = 0.0
    for a in (1.0, 0.0):
        like = np.array([rel[0], rel[1]]) if a else np.array([1 - rel[0], 1 - rel[1]])
        p = float((post * like).sum())
        if p > 0:
            out += p * entropy(post * like / p)
    return entropy(post) - out


def next_source(p_true: float, left: list[str], rel: dict) -> str | None:
    acts = {s: COST[s] - LAM * gain(p_true, rel[s]) for s in left}
    best = min(acts, key=acts.get) if acts else None
    return best if best is not None and acts[best] < 0 else None


# --------------------------------------------------------------------------- #
#  agents
# --------------------------------------------------------------------------- #
def run_bayes(w: dict, prior: float, rel: dict, *, provenance: bool, retractions: bool, meter: Meter | None,
              true_structure: bool = False, jev: bool = False) -> dict:
    """The hybrid (and the oracle when true_structure): least action over sources.

    Evidence is an event log: one entry per report, keyed by its origin; the
    belief is replayed from the active entries, so a retraction is exact.
    """
    rng = random.Random(w["id"] * 13)
    log, left, minutes, path, retracted = {}, list(NAMES), 0.0, [], False

    def belief() -> float:
        p = prior
        for (src, asserts) in log.values():
            p = update(p, rel[src], asserts)
        return p

    while (s := next_source(belief(), left, rel)) is not None:
        left.remove(s)
        minutes += COST[s]
        path.append(s)
        text = render(w, s, rng)
        if true_structure:
            asserts, origin = float(w["says"][s]), w["origin"][s]
        elif jev:
            import jev_reader
            asserts, origin = jev_reader.read_report(w["claim"], s, WHAT[s], text)
        elif meter is not None:
            asserts, origin = read_report(w["claim"], s, text, meter)
        else:
            asserts, origin = float(w["says"][s]), s
        key = origin if provenance else f"{s}"
        if retracted and key == "article_a":
            continue  # a withdrawn origin stays withdrawn: its reprints add nothing
        if key not in log:
            log[key] = (origin if provenance and origin in rel else s, asserts)
        if retractions and w["retracts"] and not retracted and any(k == "article_a" for k in log):
            retracted = True
            log.pop("article_a", None)
            if "article_a" in left:
                left.remove("article_a")  # withdrawn: nothing left to read there
            path.append("(retraction)")
    return {"answer": belief() >= 0.5, "minutes": minutes, "path": path,
            "dollars": meter.dollars if meter else 0.0}


def read_report(claim: str, source: str, text: str, meter: Meter) -> tuple[float, str]:
    """Claude reads one report: does it assert the claim, and where did it get it."""
    r = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": (
            f"Claim under investigation: {claim}.\nSource consulted: {source} ({WHAT[source]}).\n"
            f"It says: \"{text}\"\n\n"
            "1) asserts: probability (0 to 1) that this source asserts the claim is true.\n"
            "2) origin: where this source got the information. Answer 'article_a' if it relays or reprints "
            "outlet A or another media report published before it, otherwise answer 'own'."
        )}],
        output_config={"format": {"type": "json_schema", "schema": {
            "type": "object", "properties": {"asserts": {"type": "number"}, "origin": {"type": "string", "enum": ["article_a", "own"]}},
            "required": ["asserts", "origin"], "additionalProperties": False}}},
    )
    meter.add(r.usage)
    d = json.loads(next(b.text for b in r.content if b.type == "text"))
    return min(max(float(d["asserts"]), 0.01), 0.99), ("article_a" if d["origin"] == "article_a" else source)


def run_llm(w: dict, memory: str = "") -> dict:
    meter, rng = Meter(), random.Random(w["id"] * 13)
    tools = [
        {"name": "consult", "description": "Consult one source and read what it says.", "strict": True,
         "input_schema": {"type": "object", "properties": {"source": {"type": "string", "enum": NAMES}},
                          "required": ["source"], "additionalProperties": False}},
        {"name": "conclude", "description": "Give your verdict on the claim. This ends the investigation.", "strict": True,
         "input_schema": {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["true", "false"]}},
                          "required": ["verdict"], "additionalProperties": False}},
    ]
    sources = "\n".join(f"- {s} ({COST[s]:.0f} min): {WHAT[s]}" for s in NAMES)
    system = (
        f"You are checking whether a claim is true: {w['claim']}.\n"
        f"About {PRIOR_TRUE:.0%} of the rumors of this kind turn out to be true.\n"
        f"Sources you can consult, each once:\n{sources}\n"
        f"Concluding wrongly costs {MISS:.0f} minutes.\n\n"
        + (f"{memory}\n\n" if memory else "")
        + "Minimise the expected total cost: minutes spent consulting plus the cost of a wrong verdict. "
        "Consult a source only when it is worth its cost, and conclude as soon as further sources are not worth it. "
        "Use exactly one tool per turn."
    )
    messages = [{"role": "user", "content": "Start the investigation."}]
    minutes, path, done, answer, retracted, nudged = 0.0, [], set(), None, False, False
    for _ in range(len(NAMES) + 4):
        r = client.messages.create(model=MODEL, max_tokens=1000, system=system, tools=tools, messages=messages,
                                   tool_choice={"type": "auto", "disable_parallel_tool_use": True})
        meter.add(r.usage)
        messages.append({"role": "assistant", "content": r.content})
        calls = [b for b in r.content if b.type == "tool_use"]
        if not calls:
            if nudged:
                break
            nudged = True
            messages.append({"role": "user", "content": "Please use consult or conclude."})
            continue
        call = calls[0]
        if call.name == "conclude":
            answer = call.input["verdict"] == "true"
            break
        s = call.input["source"]
        if s in done:
            result = "Already consulted."
        else:
            done.add(s)
            minutes += COST[s]
            path.append(s)
            result = render(w, s, rng)
            if w["retracts"] and not retracted and (s == "article_a" or w["origin"][s] == "article_a"):
                retracted = True
                result += " " + RETRACTION_TEXT
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": call.id, "content": result}]})
    return {"answer": answer, "minutes": minutes, "path": path, "dollars": meter.dollars}


# --------------------------------------------------------------------------- #
#  bench
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["seed", "run"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--agents", nargs="+", default=["oracle", "hybrid_naive", "hybrid_prov", "llm", "llm_mem"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.cmd == "seed":
        seed_history(120, 5)
        prior, rel, _ = learn()
        print(f"prior {prior:.2f}; learned reliabilities:", {s: tuple(round(x, 2) for x in rel[s]) for s in NAMES})
        return

    prior, rel, memory = learn()
    true_rel = {s: REL[s] for s in NAMES}

    def one(w: dict) -> list[dict]:
        agents = {
            "oracle": lambda: run_bayes(w, PRIOR_TRUE, true_rel, provenance=True, retractions=True, meter=None, true_structure=True),
            "hybrid_naive": lambda: run_bayes(w, prior, rel, provenance=False, retractions=False, meter=Meter()),
            "hybrid_prov": lambda: run_bayes(w, prior, rel, provenance=True, retractions=True, meter=Meter()),
            "hybrid_prov_jev": lambda: run_bayes(w, prior, rel, provenance=True, retractions=True, meter=Meter(), jev=True),
            "llm": lambda: run_llm(w),
            "llm_mem": lambda: run_llm(w, memory),
        }
        rows = []
        for agent in a.agents:
            res = agents[agent]()
            right = res["answer"] == w["truth"]
            rows.append({"rumor": w["id"], "truth": w["truth"], "retracts": w["retracts"], "agent": agent,
                         **res, "right": right, "loss": res["minutes"] + (0.0 if right else MISS)})
        return rows

    worlds = [world(a.seed, i) for i in range(a.n)]
    with ThreadPoolExecutor(a.workers) as pool:
        rows = [r for rs in pool.map(one, worlds) for r in rs]
    out = Path(__file__).resolve().parent / "results" / f"rumor{'-' + a.tag if a.tag else ''}.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"rumor: {a.n} claims, {sum(w['truth'] for w in worlds)} true, {sum(w['retracts'] for w in worlds)} with a retraction, miss {MISS:.0f} min")
    print(f"{'agent':13} {'loss':>14} {'minutes':>8} {'sources':>8} {'accuracy':>9} {'$/episode':>10}")
    base = {r["rumor"]: r["loss"] for r in rows if r["agent"] == "llm"}
    for agent in a.agents:
        rs = [r for r in rows if r["agent"] == agent]
        loss = np.array([r["loss"] for r in rs])
        se = loss.std(ddof=1) / np.sqrt(len(loss)) if len(loss) > 1 else 0.0
        print(f"{agent:13} {loss.mean():7.2f} ±{se:5.2f} {np.mean([r['minutes'] for r in rs]):8.2f} "
              f"{np.mean([len([p for p in r['path'] if not p.startswith('(')]) for r in rs]):8.2f} "
              f"{np.mean([r['right'] for r in rs]):9.3f} {np.mean([r['dollars'] for r in rs]):10.4f}")
    if "jev_reader" in sys.modules:
        u = sys.modules["jev_reader"].usage
        print(f"jev: {u['calls']} calls, {u['tokens']} input tokens, ${u['tokens'] * sys.modules['jev_reader'].PRICE_IN:.4f}, {u['errors']} errors")


if __name__ == "__main__":
    main()
