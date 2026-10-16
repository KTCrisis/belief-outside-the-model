"""Experiment 2: an LLM alone against the hybrid agent, on the same incidents.

Every incident has a hidden cause and, for each check, a result drawn from
the scenario's P(positive | cause) and rendered as text (observations.py).
All agents see the same texts and the same brief: causes with their past
frequency, checks with their cost and what they look at, the cost of a wrong
conclusion. None is given the true P(positive | cause) table.

- llm: Claude runs checks through a tool and concludes when it judges best.
- hybrid_mem: prior and P(positive | cause) learned from closed incidents
  kept in a lab mem7 (memory_lab.py); Claude only reads each returned text
  (how likely is the signal present); Bayes keeps the belief, least action
  picks the next check and the stop.
- hybrid_llm: the same, with the table estimated by Claude from the brief
  (median of several draws) instead of learned.
- hybrid_mix: the LLM's table as the prior of each cell, corrected by the
  mem7 counts, worth --strength incidents (tables.py chose 10).
- hybrid_eb: hybrid_mix with k estimated from the history itself
  (empirical Bayes, tables.ebayes_strength) instead of fixed.
- hybrid_llm_jev, hybrid_mix_jev: the same two hybrids with Jev as the
  reader instead of Claude (jev_reader.py); the table is unchanged.
- llm_calc: the LLM agent, free to act, with a Bayes calculator tool on the
  mixed table (belief, cost of concluding now, least action's advice);
  llm_calc_def: the same with what 'positive' means for each check, as the
  hybrid's reader is told (the first version left it undefined).
- llm_mem: the LLM agent given the same closed incidents, as counts.
- oracle: least action with the true table and the true results (the
  ceiling, experiment 1's agent).

usage: python exp2.py [--n 20] [--seed 11] [--scenarios kafka internet] [--workers 4]
Writes results/<scenario>.jsonl (one line per episode and agent) and prints a summary.
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from observations import DESCRIPTIONS, NEGATIVE, POSITIVE, render  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402
from memory_lab import lab, learn, read_history, summary_for_llm  # noqa: E402

MODEL = "claude-haiku-4-5"
PRICE_IN, PRICE_OUT = 1.0 / 1e6, 5.0 / 1e6  # dollars per token, Haiku 4.5
LAM = 12.0  # price of a bit in minutes, fixed in experiment 1
NOW = __import__("datetime").datetime(2026, 10, 4, tzinfo=__import__("datetime").timezone.utc)

client = anthropic.Anthropic()


# --------------------------------------------------------------------------- #
#  the case and its incidents
# --------------------------------------------------------------------------- #
class Case:
    def __init__(self, name: str):
        sc = SCENARIOS[name]
        self.name = name
        self.miss = sc["miss"]
        self.causes = sc["causes"]
        self.meaning = sc["meaning"]
        self.prior = np.array(sc["prior"])
        self.tests = [t[0] for t in sc["tests"]]
        self.cost = {t[0]: t[1] for t in sc["tests"]}
        self.lik = np.array([t[2] for t in sc["tests"]])  # tests x causes, the truth

    def brief(self) -> str:
        causes = "\n".join(f"- {c} ({p:.0%} of past incidents): {self.meaning[c]}" for c, p in zip(self.causes, self.prior))
        checks = "\n".join(f"- {t} ({self.cost[t]:.0f} min): {DESCRIPTIONS[self.name][t]}" for t in self.tests)
        return (
            f"Possible causes, exactly one is true:\n{causes}\n\n"
            f"Available checks, each can be run once:\n{checks}\n\n"
            f"Acting on a wrong cause costs {self.miss:.0f} minutes."
        )


def incidents(case: Case, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        cause = int(rng.choice(len(case.causes), p=case.prior))
        positive, text = {}, {}
        for j, t in enumerate(case.tests):
            positive[t] = bool(rng.random() < case.lik[j, cause])
            text[t] = render(case.name, t, positive[t], random.Random(seed * 1000 + i * 31 + j))
        out.append({"id": i, "cause": cause, "positive": positive, "text": text})
    return out


class Meter:
    """Tokens spent by one agent on one episode."""

    def __init__(self):
        self.tin = 0
        self.tout = 0

    def add(self, usage) -> None:
        self.tin += usage.input_tokens
        self.tout += usage.output_tokens

    @property
    def dollars(self) -> float:
        return self.tin * PRICE_IN + self.tout * PRICE_OUT


# --------------------------------------------------------------------------- #
#  belief and least action, on any P(positive | cause) table
# --------------------------------------------------------------------------- #
def entropy(p: np.ndarray) -> float:
    q = p[p > 0]
    return float(-(q * np.log2(q)).sum())


def outcomes(post: np.ndarray, like: np.ndarray) -> list[tuple[float, np.ndarray]]:
    p_pos = float((post * like).sum())
    pos = post * like
    neg = post * (1 - like)
    return [(p_pos, pos / pos.sum()), (1 - p_pos, neg / neg.sum())]


def info_gain(post: np.ndarray, like: np.ndarray) -> float:
    return entropy(post) - sum(p * entropy(q) for p, q in outcomes(post, like) if p > 0)


def least_action_next(case: Case, post: np.ndarray, left: list[str], table: np.ndarray) -> str | None:
    acts = {t: case.cost[t] - LAM * info_gain(post, table[case.tests.index(t)]) for t in left}
    if not acts:
        return None
    best = min(acts, key=acts.get)
    return best if acts[best] < 0 else None


# --------------------------------------------------------------------------- #
#  agents
# --------------------------------------------------------------------------- #
def run_oracle(case: Case, inc: dict) -> dict:
    post, left, minutes, path = case.prior.copy(), list(case.tests), 0.0, []
    while (t := least_action_next(case, post, left, case.lik)) is not None:
        left.remove(t)
        like = case.lik[case.tests.index(t)]
        like = like if inc["positive"][t] else 1 - like
        post = post * like / (post * like).sum()
        minutes += case.cost[t]
        path.append(t)
    return {"answer": int(post.argmax()), "minutes": minutes, "path": path, "dollars": 0.0}


ELICIT_SAMPLES = 5


def elicit_table(case: Case, meter: Meter) -> np.ndarray:
    """The median, cell by cell, of ELICIT_SAMPLES independent estimates.

    Single estimates slip at random (a cell inverted on one draw, right on the
    next, the reasoning contradicting the cause's meaning); the median of
    independent draws removes random slips, not systematic errors.
    """
    return np.median([elicit_once(case, meter) for _ in range(ELICIT_SAMPLES)], axis=0)


def elicit_once(case: Case, meter: Meter) -> np.ndarray:
    """Claude estimates P(positive | cause), one check at a time, reasoning first.

    The first version filled the whole table in one call and inverted a row
    whose positive result is a double negative ("another device fails too");
    one call per check, with a sentence of reasoning per cause before the
    number, is the version kept after the pilot.
    """
    rows = []
    for t in case.tests:
        props = {c: {"type": "object", "properties": {"if_true": {"type": "string"}, "probability": {"type": "number"}},
                     "required": ["if_true", "probability"], "additionalProperties": False}
                 for c in case.causes}
        r = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": (
                f"{case.brief()}\n\nCheck: {t}. {DESCRIPTIONS[case.name][t]}\n"
                f"A positive result means: {POSITIVE[case.name][t]}.\n\n"
                "For each cause, first write in one sentence what this check would show if that cause "
                "were the true one (field if_true), then the probability (0 to 1) that the result is "
                "positive in that case. Checks are imperfect, so avoid exact 0 and 1."
            )}],
            output_config={"format": {"type": "json_schema", "schema": {
                "type": "object", "properties": props, "required": case.causes, "additionalProperties": False}}},
        )
        meter.add(r.usage)
        data = json.loads(next(b.text for b in r.content if b.type == "text"))
        rows.append([float(data[c]["probability"]) for c in case.causes])
    return np.clip(np.array(rows), 0.02, 0.98)


def read_signal(case: Case, test: str, text: str, meter: Meter) -> float:
    """Claude reads one returned text: how likely is the signal present."""
    r = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": (
            f"Check: {DESCRIPTIONS[case.name][test]}\nIt returned: \"{text}\"\n\n"
            f"What is the probability that this result shows the signal: {POSITIVE[case.name][test]}?"
        )}],
        output_config={"format": {"type": "json_schema", "schema": {
            "type": "object", "properties": {"probability": {"type": "number"}},
            "required": ["probability"], "additionalProperties": False}}},
    )
    meter.add(r.usage)
    p = float(json.loads(next(b.text for b in r.content if b.type == "text"))["probability"])
    return min(max(p, 0.01), 0.99)


def read_signal_jev(case: Case, test: str, text: str, meter: Meter) -> float:
    """Jev reads the same text (jev_reader.py); its cost is counted apart, in jev_reader.usage."""
    import jev_reader
    return jev_reader.read_signal(DESCRIPTIONS[case.name][test], text, POSITIVE[case.name][test],
                                  NEGATIVE[case.name][test])


def read_signal_jev_hard(case: Case, test: str, text: str, meter: Meter) -> float:
    """Jev's read rounded to a yes or no: hard evidence, the full likelihood ratio of the check."""
    return 1.0 if read_signal_jev(case, test, text, meter) > 0.5 else 0.0


def run_hybrid(case: Case, inc: dict, table: np.ndarray, prior: np.ndarray | None = None, reader=None) -> dict:
    meter = Meter()
    post = (case.prior if prior is None else prior).copy()
    left, minutes, path, reads = list(case.tests), 0.0, [], []
    while (t := least_action_next(case, post, left, table)) is not None:
        left.remove(t)
        p = (reader or read_signal)(case, t, inc["text"][t], meter)
        like_pos = table[case.tests.index(t)]
        like = p * like_pos + (1 - p) * (1 - like_pos)  # soft evidence
        post = post * like / (post * like).sum()
        minutes += case.cost[t]
        path.append(t)
        reads.append(round(p, 3))
    return {"answer": int(post.argmax()), "minutes": minutes, "path": path, "reads": reads,
            "dollars": meter.dollars, "tin": meter.tin, "tout": meter.tout}


def calculator(case: Case, table: np.ndarray, results: list[dict], done: set) -> str:
    """The Bayes tool given to llm_calc: belief from the results as the model read them, and least action's advice."""
    post = case.prior.copy()
    for r in results:
        like = table[case.tests.index(r["check"])]
        post = post * (like if r["positive"] else 1 - like)
        post = post / post.sum()
    left = [t for t in case.tests if t not in done]
    nxt = least_action_next(case, post, left, table)
    best = int(post.argmax())
    lines = [f"- {c}: {p:.3f}" for c, p in sorted(zip(case.causes, post), key=lambda x: -x[1])]
    return ("Probability of each cause given these results:\n" + "\n".join(lines) +
            f"\nConcluding now on {case.causes[best]}: chance of being wrong {1 - post[best]:.2f}, "
            f"expected cost of that {case.miss * (1 - post[best]):.1f} min."
            f"\nRecommended next step: {'run ' + nxt if nxt else 'conclude now, no remaining check is worth its cost'}.")


def run_llm(case: Case, inc: dict, memory: str = "", table: np.ndarray | None = None, define: bool = False) -> dict:
    meter = Meter()
    tools = [
        {"name": "run_check", "description": "Run one of the available checks and read its result.",
         "strict": True,
         "input_schema": {"type": "object", "properties": {"name": {"type": "string", "enum": case.tests}},
                          "required": ["name"], "additionalProperties": False}},
        {"name": "conclude", "description": "State the cause you act on. This ends the diagnosis.",
         "strict": True,
         "input_schema": {"type": "object", "properties": {"cause": {"type": "string", "enum": case.causes}},
                          "required": ["cause"], "additionalProperties": False}},
    ]
    if table is not None:
        tools.append(
            {"name": "bayes_calculator",
             "description": "Free, takes no time. Give the results you have read so far (each check and whether it "
                            "came out positive, i.e. showed what it looks for); returns the probability of each "
                            "cause, the expected cost of concluding now, and the check worth running next, if any."
                            + ("\nWhat 'positive' means for each check (anything else is negative):\n" + "\n".join(
                                f"- {t}: {POSITIVE[case.name][t]}" for t in case.tests) if define else ""),
             "strict": True,
             "input_schema": {"type": "object", "properties": {"results": {"type": "array", "items": {
                 "type": "object", "properties": {"check": {"type": "string", "enum": case.tests},
                                                  "positive": {"type": "boolean"}},
                 "required": ["check", "positive"], "additionalProperties": False}}},
                 "required": ["results"], "additionalProperties": False}})
    system = (
        "You are diagnosing an incident. "
        f"{case.brief()}\n\n"
        + (f"{memory}\n\n" if memory else "")
        + "Your goal is to minimise the expected total cost: the minutes spent on checks plus the "
        "cost of a wrong conclusion. Run a check only when it is worth its cost, and conclude as "
        "soon as further checks are not worth it. Use exactly one tool per turn."
    )
    messages = [{"role": "user", "content": "The incident has just been reported. Start the diagnosis."}]
    minutes, path, done, answer, nudged, trace = 0.0, [], set(), None, False, []
    for _ in range(2 * len(case.tests) + 6):
        r = client.messages.create(
            model=MODEL, max_tokens=1000, system=system, tools=tools, messages=messages,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        )
        meter.add(r.usage)
        messages.append({"role": "assistant", "content": r.content})
        said = " ".join(b.text for b in r.content if b.type == "text").strip()
        calls = [b for b in r.content if b.type == "tool_use"]
        trace.append({"said": said, "call": (calls[0].name, calls[0].input) if calls else None})
        if not calls:
            if nudged:
                break
            nudged = True
            messages.append({"role": "user", "content": "Please use run_check or conclude."})
            continue
        call = calls[0]
        if call.name == "conclude":
            answer = case.causes.index(call.input["cause"])
            break
        if call.name == "bayes_calculator":
            result = calculator(case, table, call.input["results"], done)
            trace[-1]["result"] = result
            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": call.id, "content": result}]})
            continue
        name = call.input["name"]
        if name in done:
            result = f"Already run earlier. Result was: {inc['text'][name]}"
        else:
            done.add(name)
            minutes += case.cost[name]
            path.append(name)
            result = inc["text"][name]
        trace[-1]["result"] = result
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": call.id, "content": result}]})
    return {"answer": answer, "minutes": minutes, "path": path, "trace": trace,
            "dollars": meter.dollars, "tin": meter.tin, "tout": meter.tout}


# --------------------------------------------------------------------------- #
#  bench
# --------------------------------------------------------------------------- #
def score(case: Case, inc: dict, res: dict) -> dict:
    right = res["answer"] == inc["cause"]
    return {**res, "right": right, "loss": res["minutes"] + (0.0 if right else case.miss)}


def summarise(case: Case, rows: list[dict]) -> None:
    agents = list(dict.fromkeys(r["agent"] for r in rows))
    print(f"\n== {case.name}: {len(rows) // len(agents)} incidents, miss {case.miss:.0f} min, model {MODEL}")
    print(f"{'agent':12} {'loss':>14} {'minutes':>8} {'checks':>7} {'accuracy':>9} {'$/episode':>10}")
    for agent in agents:
        rs = [r for r in rows if r["agent"] == agent]
        loss = np.array([r["loss"] for r in rs])
        se = loss.std(ddof=1) / np.sqrt(len(loss)) if len(loss) > 1 else 0.0
        print(f"{agent:12} {loss.mean():7.2f} ±{se:5.2f} {np.mean([r['minutes'] for r in rs]):8.2f} "
              f"{np.mean([len(r['path']) for r in rs]):7.2f} {np.mean([r['right'] for r in rs]):9.3f} "
              f"{np.mean([r['dollars'] for r in rs]):10.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--scenarios", nargs="+", default=["kafka", "internet"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--agents", nargs="+", default=["oracle", "hybrid_mem", "hybrid_llm", "llm", "llm_mem"])
    ap.add_argument("--tag", default="", help="suffix of the results file, to keep runs apart")
    ap.add_argument("--table-from", default="", help="reuse the elicited table of a results tag (e.g. full) instead of eliciting")
    ap.add_argument("--worst-table", action="store_true", help="use the worst median of 5 from tables.py's draws (exact loss)")
    ap.add_argument("--strength", type=float, default=10.0, help="weight of the LLM table, in incidents, for hybrid_mix")
    a = ap.parse_args()
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    for name in a.scenarios:
        case = Case(name)
        incs = incidents(case, a.n, a.seed)
        elicit_meter = Meter()
        if a.worst_table:
            from tables import expected_loss, medians_of
            table = max(medians_of(name), key=lambda t: expected_loss(case, t)["loss"])
        elif a.table_from:
            with open(out_dir / f"{name}-{a.table_from}.jsonl") as f:
                table = np.array(json.loads(f.readline())["elicited_table"])
        elif {"hybrid_llm", "hybrid_mix"} & set(a.agents):
            table = elicit_table(case, elicit_meter)
        else:
            table = case.lik * np.nan
        print(f"\n[{name}] table estimated by {MODEL} (median of {ELICIT_SAMPLES}), cost ${elicit_meter.dollars:.4f}, "
              f"MAE vs truth {np.abs(table - case.lik).mean():.3f}")
        history = read_history(case, lab())
        mem_prior, mem_table, stats = learn(case, history, NOW)
        flat_prior, flat_table, _ = learn(case, history, NOW, half_life_days=float("inf"))
        mix_table = learn(case, history, NOW, half_life_days=float("inf"), table_prior=table, strength=a.strength)[1]
        from tables import ebayes_strength
        eb_k = ebayes_strength(case, history, table) if "hybrid_eb" in a.agents else float("nan")
        eb_table = learn(case, history, NOW, half_life_days=float("inf"), table_prior=table, strength=eb_k)[1] \
            if "hybrid_eb" in a.agents else table
        if "hybrid_eb" in a.agents:
            print(f"[{name}] empirical Bayes strength k = {eb_k:.1f}")
        memory = summary_for_llm(case, history)
        print(f"[{name}] table learned from mem7 {stats}, MAE vs truth {np.abs(mem_table - case.lik).mean():.3f}"
              f" (no forgetting: {np.abs(flat_table - case.lik).mean():.3f})")

        def one(inc: dict) -> list[dict]:
            rows = []
            agents = {
                "oracle": lambda: run_oracle(case, inc),
                "hybrid_mem": lambda: run_hybrid(case, inc, mem_table, mem_prior),
                "hybrid_mem_flat": lambda: run_hybrid(case, inc, flat_table, flat_prior),
                "hybrid_llm": lambda: run_hybrid(case, inc, table),
                "hybrid_mix": lambda: run_hybrid(case, inc, mix_table),
                "hybrid_eb": lambda: run_hybrid(case, inc, eb_table),
                "hybrid_llm_jev": lambda: run_hybrid(case, inc, table, reader=read_signal_jev),
                "hybrid_mix_jev": lambda: run_hybrid(case, inc, mix_table, reader=read_signal_jev),
                "hybrid_mix_jev_hard": lambda: run_hybrid(case, inc, mix_table, reader=read_signal_jev_hard),
                "llm": lambda: run_llm(case, inc),
                "llm_mem": lambda: run_llm(case, inc, memory),
                "llm_calc": lambda: run_llm(case, inc, table=mix_table),
                "llm_calc_def": lambda: run_llm(case, inc, table=mix_table, define=True),
            }
            for agent in a.agents:
                fn = agents[agent]
                t0 = time.time()
                res = score(case, inc, fn())
                rows.append({"scenario": name, "incident": inc["id"], "cause": case.causes[inc["cause"]],
                             "agent": agent, "seconds": round(time.time() - t0, 1), **res})
            return rows

        with ThreadPoolExecutor(a.workers) as pool:
            rows = [r for rs in pool.map(one, incs) for r in rs]
        with open(out_dir / f"{name}{'-' + a.tag if a.tag else ''}.jsonl", "w") as f:
            f.write(json.dumps({"elicited_table": table.round(3).tolist(), "elicit_dollars": elicit_meter.dollars,
                                "memory_table": mem_table.round(3).tolist(), "memory_prior": mem_prior.round(3).tolist(),
                                "memory_stats": stats}) + "\n")
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summarise(case, rows)
        if "jev_reader" in sys.modules:
            u = sys.modules["jev_reader"].usage
            print(f"jev so far: {u['calls']} calls, {u['tokens']} input tokens, ${u['tokens'] * sys.modules['jev_reader'].PRICE_IN:.4f}, {u['errors']} errors")


if __name__ == "__main__":
    main()
