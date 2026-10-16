"""Does lambda follow the price of an error? Sweep M on the current scenario.

For each M (a multiple of the scenario's own M), compute exactly:
- the optimum;
- the slope a of V on the entropy over all states (value_fit), so M/a;
- the best lambda for least_action and ratio_priced on a grid of M/k,
  and the loss with the untuned default lambda = M/4.

If lambda behaves like a price set by M, M/lambda stays roughly constant
across M and the default M/4 stays close to the optimum everywhere.

usage: SCENARIO=kafka|internet|clinic python m_sweep.py
"""

import numpy as np

from exact import evaluate, optimum
from sim import CASE, SCENARIO
from value_fit import all_values, fit

FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0]
KS = [8, 7, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2]


def best_lambda(rule: str, miss: float) -> tuple[float, float]:
    """Best lambda on the grid M/k (ties: the median k of the minimisers), and its loss."""
    losses = [(evaluate(rule, miss, miss / k, 0.9)["loss"], k) for k in KS]
    low = min(l for l, _ in losses)
    ks = [k for l, k in losses if l <= low + 1e-9]
    return float(np.median(ks)), low


def main() -> None:
    m0 = CASE["miss"]
    print(f"scenario {SCENARIO}, base M = {m0:g}  (exact; grid lambda = M/k, k in {KS})\n")
    print(f"{'M':>7} {'opt':>8} {'M/a':>6} {'R2(H)':>6} | {'least_action':^30} | {'ratio_priced':^30}")
    print(f"{'':>7} {'':>8} {'':>6} {'':>6} | {'best k':>7} {'loss':>8} {'M/4 loss':>9} {'gap%':>5} | {'best k':>7} {'loss':>8} {'M/4 loss':>9} {'gap%':>5}")
    for f in FACTORS:
        miss = m0 * f
        opt, _ = optimum(miss)
        _, rows = all_values(miss)
        V, H = rows[:, 0], rows[:, 1]
        a, r2 = fit(H, V)
        cells = []
        for rule in ["least_action", "ratio_priced"]:
            k, loss = best_lambda(rule, miss)
            d = evaluate(rule, miss, miss / 4, 0.9)["loss"]
            cells.append(f"{k:7.1f} {loss:8.2f} {d:9.2f} {100 * (d - opt) / opt:5.1f}")
        print(f"{miss:7.1f} {opt:8.2f} {miss / a:6.2f} {r2:6.2f} | {cells[0]} | {cells[1]}")


if __name__ == "__main__":
    main()
