"""Where does lam ~ M/4 come from? A check written before looking at the numbers.

Prediction (08/10/2026, before running): at the beliefs where the optimal agent
stops, a bit of remaining entropy carries a fixed share of error, so
M / lam ~ H / P(wrong) measured at those stops. With two causes left and 10 to
20 % error, H / P(wrong) is 3.6 to 4.7; nearer certainty it grows (5.7 at 5 %),
which would explain why M / lam drifts upward with M. The test passes if the
predicted M / lam tracks the best M / lam of the least_action rule across the
three cases and across M x 1/4 .. x 4, drift included.

Two predictors from the optimal policy's stopping beliefs, weighted by how often
each is reached:
- ratio of means: E[H] / E[P(wrong)];
- binary: h(e) / e with e = E[P(wrong)], as if two causes were left.
Measured: the M / lam that minimises least_action's exact loss (fine sweep), with
the range of M / lam within 1 % of that minimum, since the curve is often flat.

usage: python lambda_check.py      # runs the three scenarios in subprocesses
"""
import json
import os
import subprocess
import sys

import numpy as np

SCALES = (0.25, 0.5, 1, 2, 4)


def h2(e: float) -> float:
    return 0.0 if e <= 0 or e >= 1 else float(-e * np.log2(e) - (1 - e) * np.log2(1 - e))


def one(scale: float) -> dict:
    from exact import evaluate
    from sim import CASE, COST, NAMES, PRIOR, entropy, outcomes, p_wrong

    miss = CASE["miss"] * scale
    memo: dict = {}

    def value(post, left):
        k = (tuple(np.round(post, 12)), left)
        if k in memo:
            return memo[k]
        best, act = miss * p_wrong(post), None
        for t in left:
            rest = tuple(x for x in left if x != t)
            v = COST[t] + sum(p * value(q, rest)[0] for p, q in outcomes(post, t) if p > 0)
            if v < best - 1e-12:
                best, act = v, t
        memo[k] = (best, act)
        return memo[k]

    stops = []  # (reach probability, P(wrong), H)

    def walk(post, left, reach):
        _, act = value(post, left)
        if act is None:
            stops.append((reach, p_wrong(post), entropy(post)))
            return
        rest = tuple(x for x in left if x != act)
        for p, q in outcomes(post, act):
            if p > 0:
                walk(q, rest, reach * p)

    start = tuple(range(len(NAMES)))
    opt = value(PRIOR.copy(), start)[0]
    walk(PRIOR.copy(), start, 1.0)
    w, e, H = np.array(stops).T
    e_mean, h_mean = float(w @ e), float(w @ H)

    ks = np.linspace(2, 9, 141)
    losses = np.array([evaluate("least_action", miss, miss / k, 0.9)["loss"] for k in ks])
    best = losses.min()
    near = ks[losses <= best * 1.01]
    return {"M": miss, "optimum": opt, "err_at_stop": e_mean, "H_at_stop": h_mean,
            "pred_ratio": h_mean / e_mean if e_mean > 0 else float("nan"),
            "pred_binary": h2(e_mean) / e_mean if e_mean > 0 else float("nan"),
            "best_k": float(ks[losses.argmin()]), "k_lo": float(near.min()), "k_hi": float(near.max()),
            "best_gap_pct": 100 * (best - opt) / opt}


def main() -> None:
    if os.environ.get("LAMBDA_CHILD"):
        print(json.dumps([one(s) for s in SCALES]))
        return
    print("M / lam: predicted from the optimal stops vs best measured (least_action, exact)\n")
    print(f"{'case':9} {'M':>7} {'err@stop':>9} {'H@stop':>7} {'pred H/e':>9} {'pred bin':>9} "
          f"{'best M/lam':>11} {'within 1 %':>12} {'gap opt':>8}")
    for sc in ("kafka", "internet", "clinic"):
        env = dict(os.environ, SCENARIO=sc, LAMBDA_CHILD="1")
        out = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=True).stdout
        for r in json.loads(out.strip().splitlines()[-1]):
            print(f"{sc:9} {r['M']:7.1f} {r['err_at_stop']:9.3f} {r['H_at_stop']:7.3f} {r['pred_ratio']:9.2f} "
                  f"{r['pred_binary']:9.2f} {r['best_k']:11.2f} {r['k_lo']:5.2f}-{r['k_hi']:<6.2f} "
                  f"{r['best_gap_pct']:+7.1f}%")


if __name__ == "__main__":
    main()
