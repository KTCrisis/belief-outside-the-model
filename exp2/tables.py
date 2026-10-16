"""How much does the hybrid depend on the table it is given?

The full series elicited the table once per scenario: 200 incidents, one
draw of the table. This script draws many single estimates, forms
independent medians of 5 (as exp2 does), and evaluates each table exactly:
the agent believes and decides with its table (least action, lambda 12),
the results come from the true table, the reader is perfect (hard
evidence), so only the table differs. Expected loss is summed over every
branch of the decision tree, no sampling.

It also evaluates the table learned from mem7, alone and with the LLM's
table as its prior (memory_lab.learn, table_prior and strength).

usage: python tables.py draw [--n 50]   # API spend, ~0.018 $ per single draw
       python tables.py eval            # no API call
       python tables.py validate        # k chosen on the lab history, tested on fresh ones
       python tables.py ebayes          # k estimated from each history itself
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp2 import NOW, Case, Meter, elicit_once, least_action_next  # noqa: E402
from memory_lab import draw_history, lab, learn, read_history  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"


def medians_of(name: str) -> list[np.ndarray]:
    """The independent medians of 5 formed from the stored single draws."""
    draws = np.array(json.loads((OUT / f"tables-{name}.json").read_text())["draws"])
    return [np.median(draws[i:i + 5], axis=0) for i in range(0, len(draws) - 4, 5)]


def expected_loss(case: Case, table: np.ndarray, prior: np.ndarray | None = None) -> dict:
    """Exact expected loss of least action run on `table`, under the truth."""
    agent_prior = case.prior if prior is None else prior

    def walk(post, weight, left, minutes):
        # weight[c] = P(true cause c and the results seen so far)
        t = least_action_next(case, post, left, table)
        if t is None:
            answer = int(post.argmax())
            mass = weight.sum()
            return np.array([mass * minutes + case.miss * (mass - weight[answer]), weight[answer], mass * minutes])
        j = case.tests.index(t)
        rest = [x for x in left if x != t]
        total = np.zeros(3)
        for positive in (True, False):
            agent_like = table[j] if positive else 1 - table[j]
            true_like = case.lik[j] if positive else 1 - case.lik[j]
            p = post * agent_like
            total += walk(p / p.sum(), weight * true_like, rest, minutes + case.cost[t])
        return total

    loss, right, minutes = walk(agent_prior.copy(), case.prior.copy(), list(case.tests), 0.0)
    return {"loss": round(float(loss), 2), "accuracy": round(float(right), 3), "minutes": round(float(minutes), 2)}


def draw(n: int) -> None:
    for name in ["kafka", "internet"]:
        case = Case(name)

        def one(_):
            meter = Meter()
            return elicit_once(case, meter).tolist(), meter.dollars

        with ThreadPoolExecutor(8) as ex:
            got = list(ex.map(one, range(n)))
        dollars = sum(d for _, d in got)
        path = OUT / f"tables-{name}.json"
        path.write_text(json.dumps({"draws": [t for t, _ in got], "dollars": dollars}))
        print(f"{name}: {n} draws, {dollars:.2f} $ -> {path.name}")


def evaluate() -> None:
    m = lab()
    for name in ["kafka", "internet"]:
        case = Case(name)
        draws = np.array(json.loads((OUT / f"tables-{name}.json").read_text())["draws"])
        medians = [np.median(draws[i:i + 5], axis=0) for i in range(0, len(draws) - 4, 5)]
        hist = read_history(case, m)
        flat = learn(case, hist, NOW, half_life_days=1e9)[1]
        print(f"\n== {name} (miss {case.miss:.0f} min), exact expected loss per incident")
        print(f"{'table':34} {'loss':>7} {'minutes':>8} {'accuracy':>9} {'MAE':>6}")

        def row(label, table):
            r = expected_loss(case, table)
            print(f"{label:34} {r['loss']:7.2f} {r['minutes']:8.2f} {r['accuracy']:9.3f} {np.abs(table - case.lik).mean():6.3f}")
            return r["loss"]

        row("truth", case.lik)
        row("mem7 alone (flat)", flat)
        singles = [expected_loss(case, d)["loss"] for d in draws]
        meds = [expected_loss(case, t)["loss"] for t in medians]
        mixes = {k: [expected_loss(case, learn(case, hist, NOW, 1e9, table_prior=t, strength=k)[1])["loss"]
                     for t in medians] for k in (2, 5, 10, 20)}
        for label, xs in [("LLM single draw", singles), ("LLM median of 5", meds)] + \
                         [(f"LLM median as prior + mem7, k={k}", v) for k, v in mixes.items()]:
            xs = np.array(xs)
            print(f"{label:34} {xs.mean():7.2f}  min {xs.min():6.2f}  max {xs.max():6.2f}  (n={len(xs)})")


def validate(histories: int = 10, size: int = 120) -> None:
    """k = 10 was picked on the lab mem7's history; does it hold on histories it never saw?"""
    ks = (2, 5, 10, 20, 40)
    for name in ["kafka", "internet"]:
        case = Case(name)
        medians = medians_of(name)
        loss = {k: [] for k in ks}
        alone = []
        for h in range(histories):
            hist = draw_history(case, size, 1000 + h, NOW)
            alone.append(expected_loss(case, learn(case, hist, NOW, 1e9)[1])["loss"])
            for k in ks:
                loss[k] += [expected_loss(case, learn(case, hist, NOW, 1e9, table_prior=t, strength=k)[1])["loss"]
                            for t in medians]
        print(f"\n== {name}: {histories} fresh histories of {size} incidents x {len(medians)} LLM tables")
        print(f"mem7 alone: {np.mean(alone):.2f} (min {min(alone):.2f}, max {max(alone):.2f})")
        for k in ks:
            xs = np.array(loss[k])
            print(f"k={k:<3} {xs.mean():6.2f}  p90 {np.percentile(xs, 90):6.2f}  max {xs.max():6.2f}")


K_GRID = np.geomspace(1, 300, 40)


def counts(case: Case, history: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Positives and runs per (test, cause), unweighted."""
    pos = np.zeros((len(case.tests), len(case.causes)))
    run = np.zeros_like(pos)
    for inc in history:
        c = case.causes.index(inc["cause"])
        for t, positive in inc["results"].items():
            run[case.tests.index(t), c] += 1
            pos[case.tests.index(t), c] += positive
    return pos, run


def ebayes_strength(case: Case, history: list[dict], table: np.ndarray) -> float:
    """The k under which the closed incidents are most probable (beta-binomial marginal likelihood).

    Each cell's counts are Binomial(run, p) with p ~ Beta(k q, k (1 - q)), q the LLM's
    value: a k that makes the history likely means the LLM's table fits what was seen.
    """
    from scipy.special import betaln
    pos, run = counts(case, history)
    neg = run - pos

    def loglik(k):
        a, b = k * table, k * (1 - table)
        return float((betaln(pos + a, neg + b) - betaln(a, b)).sum())

    return float(max(K_GRID, key=loglik))


def ebayes(histories: int = 10, size: int = 120) -> None:
    for name in ["kafka", "internet"]:
        case = Case(name)
        rows = []
        for h in range(histories):
            hist = draw_history(case, size, 1000 + h, NOW)
            for t in medians_of(name):
                k = ebayes_strength(case, hist, t)
                eb = expected_loss(case, learn(case, hist, NOW, 1e9, table_prior=t, strength=k)[1])["loss"]
                k10 = expected_loss(case, learn(case, hist, NOW, 1e9, table_prior=t, strength=10)[1])["loss"]
                rows.append((k, eb, k10, expected_loss(case, t)["loss"]))
        r = np.array(rows)
        print(f"\n== {name}: k estimated per history and table ({len(r)} pairs)")
        print(f"k chosen: median {np.median(r[:, 0]):.1f} (p10 {np.percentile(r[:, 0], 10):.1f}, p90 {np.percentile(r[:, 0], 90):.1f})")
        for label, col in [("empirical Bayes k", 1), ("fixed k = 10", 2), ("LLM table alone", 3)]:
            print(f"{label:18} {r[:, col].mean():6.2f}  p90 {np.percentile(r[:, col], 90):6.2f}  max {r[:, col].max():6.2f}")
        print(f"k vs table quality: corr(k, LLM-alone loss) = {np.corrcoef(r[:, 0], r[:, 3])[0, 1]:+.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["draw", "eval", "validate", "ebayes"])
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    {"draw": lambda: draw(a.n), "eval": evaluate, "validate": validate, "ebayes": ebayes}[a.cmd]()


if __name__ == "__main__":
    main()
