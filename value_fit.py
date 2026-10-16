"""How well does each one-step rule guess the future? Fit the true optimal
value V (exact.py's dynamic programming) against the two guesses the rules make.

Bellman: V(belief) = min(stop: M * P(wrong), min over tests: cost + E[V(after)]).
- Bayes decision (one step) replaces V(after) by M * P(wrong after).
- Least action replaces V(after) by lam * H(after); minimising
  cost + lam * E[H(after)] is minimising cost - lam * (bits gained).
Both are approximations of the same equation. Fitting V ~ a * H over every
state gives the price of a bit implied by the optimum itself (a, in min/bit),
instead of tuning lam by hand.

usage: SCENARIO=kafka|internet python value_fit.py [--miss M]
"""
import argparse

import numpy as np

from sim import CASE, COST, PRIOR, SCENARIO, entropy, outcomes, p_wrong


def all_values(miss: float) -> tuple[float, np.ndarray]:
    """Optimal value of every state: rows of (V, H, M * P(wrong), tests left)."""
    memo: dict = {}

    def value(post: np.ndarray, left: tuple[int, ...]) -> float:
        k = (tuple(np.round(post, 12)), left)
        if k in memo:
            return memo[k][0]
        best = miss * p_wrong(post)
        for t in left:
            rest = tuple(x for x in left if x != t)
            best = min(best, COST[t] + sum(p * value(q, rest) for p, q in outcomes(post, t) if p > 0))
        memo[k] = (best, entropy(post), miss * p_wrong(post), len(left))
        return best

    v0 = value(PRIOR, tuple(range(len(COST))))
    return v0, np.array(list(memo.values()))


def fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope through the origin and R^2 of y ~ a * x."""
    a = float(x @ y / (x @ x))
    res = y - a * x
    return a, float(1 - res @ res / ((y - y.mean()) @ (y - y.mean())))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--miss", type=float, default=CASE["miss"])
    args = ap.parse_args()
    v0, rows = all_values(args.miss)
    V, H, R, n = rows.T
    print(f"{SCENARIO}  M={args.miss:g}  states={len(V)}  optimum V0={v0:.2f}")
    for label, mask in [("all states", np.ones_like(V, bool)), ("3+ tests left", n >= 3)]:
        a, r2a = fit(H[mask], V[mask])
        b, r2b = fit(R[mask], V[mask])
        print(f"  {label:14s} V ~ a*H: a={a:5.2f} min/bit (M/a={args.miss / a:.2f}) R2={r2a:.3f}"
              f" | V ~ b*M*risk: b={b:.2f} R2={r2b:.3f}")


if __name__ == "__main__":
    main()
