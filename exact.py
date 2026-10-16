"""Exact expected loss, no sampling: the true optimum by dynamic programming,
and each deterministic rule of sim.py evaluated on the same tree.

A state is the posterior reached plus the tests left. From each state the
optimal agent either stops (loss M * P(wrong)) or runs the test whose cost
plus expected value of the next state is lowest. With n tests of two outcomes
there are at most 3^n states (2187 for 7 tests), so the tree is solved exactly.
This is the reference no rule can beat; sim.py's "bayes_decision" only looks
one step ahead and is not it.

usage: SCENARIO=kafka|internet python exact.py [--miss M] [--lam 12]
"""
import argparse
from functools import lru_cache

import numpy as np

from sim import CASE, COST, NAMES, PRIOR, SCENARIO, THRESHOLD_RULES, choose, outcomes, p_wrong


def key(post: np.ndarray, left: tuple[int, ...]) -> tuple:
    return (tuple(np.round(post, 12)), left)


def optimum(miss: float) -> tuple[float, str]:
    """Expected loss of the optimal policy, and its first move."""
    memo: dict = {}

    def value(post: np.ndarray, left: tuple[int, ...]) -> float:
        k = key(post, left)
        if k in memo:
            return memo[k]
        best = miss * p_wrong(post)  # stop now
        for t in left:
            rest = tuple(x for x in left if x != t)
            v = COST[t] + sum(p * value(q, rest) for p, q in outcomes(post, t) if p > 0)
            best = min(best, v)
        memo[k] = best
        return best

    start = tuple(range(len(NAMES)))
    total = value(PRIOR.copy(), start)
    stop = miss * p_wrong(PRIOR)
    first = "stop"
    for t in start:
        rest = tuple(x for x in start if x != t)
        v = COST[t] + sum(p * value(q, rest) for p, q in outcomes(PRIOR, t) if p > 0)
        if abs(v - total) < 1e-9:
            first = NAMES[t]
    if abs(stop - total) < 1e-9:
        first = "stop"
    return total, first


def evaluate(rule: str, miss: float, lam: float, target: float) -> dict:
    """Exact expected loss, minutes and accuracy of a deterministic rule."""
    rng = np.random.default_rng(0)  # unused by deterministic rules

    def walk(post: np.ndarray, left: list[int]) -> tuple[float, float]:
        t = choose(rule, post, left, rng, lam, miss, target) if left else None
        if t is None:
            return 0.0, 1.0 - p_wrong(post)  # minutes from here, P(right)
        rest = [x for x in left if x != t]
        minutes, right = COST[t], 0.0
        for p, q in outcomes(post, t):
            if p > 0:
                m, r = walk(q, rest)
                minutes += p * m
                right += p * r
        return minutes, right

    minutes, right = walk(PRIOR.copy(), list(range(len(NAMES))))
    return {"rule": rule, "loss": minutes + miss * (1 - right), "minutes": minutes, "accuracy": right}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--miss", type=float, default=CASE["miss"])
    ap.add_argument("--lam", type=float, default=12.0)
    ap.add_argument("--target", type=float, default=0.9)
    a = ap.parse_args()

    best, first = optimum(a.miss)
    print(f"scenario {SCENARIO}: miss {a.miss} min, lam {a.lam} min/bit, threshold {a.target}  (exact expectations, no sampling)\n")
    print(f"{'rule':20} {'loss':>7} {'minutes':>8} {'accuracy':>9} {'gap to optimum':>15}")
    print(f"{'OPTIMUM (DP)':20} {best:7.2f} {'':>8} {'':>9} {'':>15}   first move: {first}")
    for rule in [r for r in THRESHOLD_RULES if r != "random"] + ["least_action", "ratio_priced", "least_action_path", "bayes_decision", "chernoff_bayes", "chernoff_priced", "chernoff_risk", "two_criteria",
                 "two_criteria_2step"]:
        r = evaluate(rule, a.miss, a.lam, a.target)
        print(f"{r['rule']:20} {r['loss']:7.2f} {r['minutes']:8.2f} {r['accuracy']:9.3f} {r['loss'] - best:+15.2f}")

    lams = [a.miss / k for k in (8, 6, 5, 4.5, 4, 3.5, 3, 2)]
    for rule in ["least_action", "ratio_priced"]:
        print(f"\nlam sweep, {rule} (exact; lam = M/8 .. M/2):")
        for lam in lams:
            r = evaluate(rule, a.miss, lam, a.target)
            print(f"  lam {lam:6.1f} (M/{a.miss / lam:.1f}): loss {r['loss']:7.2f}  gap {r['loss'] - best:+.2f}")


if __name__ == "__main__":
    main()
