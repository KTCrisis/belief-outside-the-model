"""Bayes for belief, least action for choice: a diagnosis agent on a toy Kafka
consumer-lag incident.

The hidden cause is one of six. Each diagnostic costs minutes and returns a
positive or negative signal whose probability depends on the cause. The agent
keeps a posterior over causes (Bayes) and picks its next diagnostic by a rule;
it stops when one cause reaches the target confidence, or when nothing is
left to run, and names the most probable cause.

Three ways to choose and to stop:
- threshold rules: pick by a heuristic (random, cheapest, most informative,
  information per minute) and stop at a fixed confidence;
- least action: an investigation is a trajectory whose action is
  S = sum(cost) - lam * sum(bits gained); the agent takes the step of least
  expected action, and stops when every step left has a positive action, so
  the principle also decides when to stop. lam is the price of a bit;
- Bayes decision: the expected loss of stopping now is M * P(wrong), and the
  agent runs the test that lowers cost + expected loss most, or stops. This
  is the same minimisation with P(wrong) in place of entropy.
Score: loss = minutes + M * (wrong answer), M the cost of a misdiagnosis.

usage: SCENARIO=kafka|internet python sim.py [--episodes 3000] [--seed 7] [--miss M]
"""
import argparse
import os

import numpy as np

from scenarios import SCENARIOS

# The case is chosen by the SCENARIO environment variable (kafka by default),
# so sim.py and exact.py read the same one; see scenarios.py.
SCENARIO = os.environ.get("SCENARIO", "kafka")
CASE = SCENARIOS[SCENARIO]
CAUSES = CASE["causes"]
PRIOR = np.array(CASE["prior"])
TESTS = CASE["tests"]
NAMES = [t[0] for t in TESTS]
COST = np.array([t[1] for t in TESTS])
LIK = np.array([t[2] for t in TESTS])  # tests x causes


def entropy(p: np.ndarray) -> float:
    q = p[p > 0]
    return float(-(q * np.log2(q)).sum())


def update(post: np.ndarray, test: int, positive: bool) -> np.ndarray:
    like = LIK[test] if positive else 1.0 - LIK[test]
    new = post * like
    return new / new.sum()


def outcomes(post: np.ndarray, test: int) -> list[tuple[float, np.ndarray]]:
    """The two possible results of a test: (probability, posterior after it)."""
    p_pos = float((post * LIK[test]).sum())
    return [(p_pos, update(post, test, True)), (1.0 - p_pos, update(post, test, False))]


def info_gain(post: np.ndarray, test: int) -> float:
    """Expected entropy reduction, in bits: the epistemic value of a test."""
    return entropy(post) - sum(p * entropy(q) for p, q in outcomes(post, test) if p > 0)


def step_action(post: np.ndarray, test: int, lam: float) -> float:
    return COST[test] - lam * info_gain(post, test)


def path_action(post: np.ndarray, test: int, left: list[int], lam: float, target: float) -> float:
    """Expected action of running test, then the best next test (two steps)."""
    total = step_action(post, test, lam)
    rest = [t for t in left if t != test]
    for p, q in outcomes(post, test):
        if p == 0 or q.max() >= target or not rest:
            continue
        # the agent may stop instead, at zero action
        total += p * min(0.0, min(step_action(q, t, lam) for t in rest))
    return total


def p_wrong(post: np.ndarray) -> float:
    return 1.0 - float(post.max())


def bern_kl(p: float, q: float) -> float:
    p, q = min(max(p, 1e-9), 1 - 1e-9), min(max(q, 1e-9), 1 - 1e-9)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def chernoff_score(post: np.ndarray, test: int) -> float:
    """Chernoff (1959), deterministic and per unit cost: how well the test separates
    the leading cause from its closest rival, min over rivals of KL(leader || rival)."""
    lead = int(post.argmax())
    return min(bern_kl(LIK[test, lead], LIK[test, j]) for j in range(LIK.shape[1]) if j != lead) / COST[test]


def step_loss(post: np.ndarray, test: int, miss: float) -> float:
    """Cost of the test plus the expected loss of stopping right after it."""
    return COST[test] + sum(p * miss * p_wrong(q) for p, q in outcomes(post, test))


THRESHOLD_RULES = ["random", "cheapest", "max_info", "info_per_cost"]


def choose(rule: str, post: np.ndarray, left: list[int], rng: np.random.Generator, lam: float, miss: float, target: float) -> int | None:
    """The next test, or None to stop."""
    if rule in THRESHOLD_RULES:
        if post.max() >= target:
            return None
        if rule == "random":
            return int(rng.choice(left))
        if rule == "cheapest":
            return min(left, key=lambda t: COST[t])
        if rule == "max_info":
            return max(left, key=lambda t: info_gain(post, t))
        return max(left, key=lambda t: info_gain(post, t) / COST[t])
    if rule == "least_action":
        best = min(left, key=lambda t: step_action(post, t, lam))
        return best if step_action(post, best, lam) < 0 else None
    if rule == "ratio_priced":
        # rank by information per cost (as HESP), stop by the price of a bit:
        # continue only while the best test costs less than lam per bit
        best = max(left, key=lambda t: info_gain(post, t) / COST[t])
        gain = info_gain(post, best)
        return best if gain > 0 and COST[best] / gain < lam else None
    if rule.startswith("chernoff"):
        # selection by Chernoff's separation of the leader from its closest rival;
        # three stops: myopic Bayes (jeu A), priced bits (lam), and Chernoff-style
        # (stop once the risk is below the mean test cost over M)
        best = max(left, key=lambda t: chernoff_score(post, t))
        if rule == "chernoff_bayes":
            return best if step_loss(post, best, miss) < miss * p_wrong(post) else None
        if rule == "chernoff_priced":
            gain = info_gain(post, best)
            return best if gain > 0 and COST[best] / gain < lam else None
        return best if p_wrong(post) > COST.mean() / miss else None
    if rule == "least_action_path":
        best = min(left, key=lambda t: path_action(post, t, left, lam, 1.01))
        return best if path_action(post, best, left, lam, 1.01) < 0 else None
    if rule in ("two_criteria", "two_criteria_2step"):
        # choose by bits (the least S), stop by the price of an error: run the chosen test only if its cost plus
        # the expected M x P(wrong) after it is below M x P(wrong) now. The 2-step variant, before stopping,
        # also asks whether some test followed by the best second test (or a stop) would beat stopping now.
        best = min(left, key=lambda t: step_action(post, t, lam))
        stop_now = miss * p_wrong(post)
        if step_loss(post, best, miss) < stop_now:
            return best
        if rule == "two_criteria_2step":
            for t1 in left:
                rest = [x for x in left if x != t1]
                v = COST[t1]
                for p, q in outcomes(post, t1):
                    if p > 0:
                        v += p * min([miss * p_wrong(q)] + [step_loss(q, t2, miss) for t2 in rest])
                if v < stop_now - 1e-12:
                    return best
        return None
    if rule == "bayes_decision":
        best = min(left, key=lambda t: step_loss(post, t, miss))
        return best if step_loss(post, best, miss) < miss * p_wrong(post) else None
    raise ValueError(rule)


def episode(rule: str, cause: int, rng: np.random.Generator, lam: float, miss: float, target: float) -> tuple[float, int, bool]:
    post = PRIOR.copy()
    left = list(range(len(TESTS)))
    cost, steps = 0.0, 0
    while left:
        t = choose(rule, post, left, rng, lam, miss, target)
        if t is None:
            break
        left.remove(t)
        post = update(post, t, rng.random() < LIK[t, cause])
        cost += COST[t]
        steps += 1
    return cost, steps, int(post.argmax()) == cause


def run(rule: str, episodes: int, seed: int, lam: float, miss: float, target: float) -> dict:
    rng = np.random.default_rng(seed)
    causes = rng.choice(len(CAUSES), size=episodes, p=PRIOR)
    res = np.array([episode(rule, int(c), rng, lam, miss, target) for c in causes], dtype=float)
    cost, steps, acc = res[:, 0].mean(), res[:, 1].mean(), res[:, 2].mean()
    loss = res[:, 0] + miss * (1 - res[:, 2])
    return {"rule": rule, "cost": cost, "steps": steps, "accuracy": acc, "loss": loss.mean(), "se": loss.std(ddof=1) / np.sqrt(len(loss))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--miss", type=float, default=CASE["miss"], help="cost of a wrong diagnosis, in minutes")
    ap.add_argument("--target", type=float, default=0.9, help="confidence the threshold rules stop at")
    ap.add_argument("--lam", type=float, default=12.0, help="price of one bit, in minutes")
    a = ap.parse_args()

    print(f"scenario {SCENARIO}: {a.episodes} incidents, miss {a.miss} min, threshold {a.target}, lam {a.lam} min/bit\n")
    print(f"{'rule':20} {'loss':>13} {'cost':>7} {'steps':>6} {'accuracy':>9}")
    for rule in THRESHOLD_RULES + ["least_action", "ratio_priced", "least_action_path", "bayes_decision", "chernoff_bayes", "chernoff_priced", "chernoff_risk"]:
        r = run(rule, a.episodes, a.seed, a.lam, a.miss, a.target)
        print(f"{r['rule']:20} {r['loss']:7.2f} ±{r['se']:4.2f} {r['cost']:7.2f} {r['steps']:6.2f} {r['accuracy']:9.3f}")

    print("\nlam sweep (least_action):")
    for lam in [2, 4, 8, 12, 20, 40]:
        r = run("least_action", a.episodes, a.seed, lam, a.miss, a.target)
        print(f"  lam {lam:>3}: loss {r['loss']:6.2f}  cost {r['cost']:6.2f}  steps {r['steps']:4.2f}  accuracy {r['accuracy']:.3f}")


if __name__ == "__main__":
    main()
