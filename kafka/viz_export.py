"""Export the numbers behind the Bellman / lambda x H page (exploratory, nothing re-run).

- cloud: every belief reachable under the mean mixed table (study 1 counts), its entropy H and its exact
  value V (dynamic programming, binary readings), for M = 15, 60, 240; with the slope a of V ~ a * H and R^2;
- checks: cost and expected bits of each check at the uniform prior (mean mixed table);
- curve: on study 2 (routed reader, the 10 mixed tables), the rule's minutes and error rate for each lambda
  from 1 to 30 (its decisions do not depend on M, so loss(M) = minutes + M x error rate), and the exact
  optimum's loss for each M.

usage: python viz_export.py > results/viz-export.json
"""
import json

import numpy as np

import analyze as A
import analyze2 as B
from optimal_replay import Optimal, replay_opt

MS_CLOUD = [15, 60, 240]
MS_OPT = [2, 4, 8, 15, 30, 60, 120, 240]


def cloud(table, miss):
    memo = {}

    def value(post, left):
        k = (tuple(np.round(post, 10)), left)
        if k in memo:
            return memo[k][0]
        best = miss * A.p_wrong(post)
        for j in left:
            like = table[j]
            p_pos = float(post @ like)
            rest = tuple(x for x in left if x != j)
            ev = sum(p * value(A.update(post, like, r), rest) for r, p in ((1.0, p_pos), (0.0, 1 - p_pos)) if p > 1e-12)
            best = min(best, A.COST[j] + ev)
        memo[k] = (best, A.entropy(post))
        return best

    value(np.full(A.NC, 1 / A.NC), tuple(range(A.NT)))
    V, H = np.array([v for v, _ in memo.values()]), np.array([h for _, h in memo.values()])
    pts = sorted({(round(float(h), 3), round(float(v), 2)) for v, h in memo.values()})  # for display only
    a = float(H @ V / (H @ H))
    res = V - a * H
    return {"points": pts, "a": round(a, 2), "r2": round(float(1 - res @ res / ((V - V.mean()) @ (V - V.mean()))), 2)}


def main() -> None:
    hist, hr, _ = A.load("main")
    hist = [e for e in hist if not A.excluded(e)]
    pos, run = A.counts(hist, hr)
    meds = A.llm_medians()
    mixed = [A.table_mixed(pos, run, m, A.ebayes_k(pos, run, m)) for m in meds]
    table = np.mean(mixed, axis=0)
    out = {"checks": [], "cloud": {}, "curve": [], "optimum": {}}

    u = np.full(A.NC, 1 / A.NC)
    for j, c in enumerate(A.CHECKS):
        out["checks"].append({"check": c, "cost": float(A.COST[j]), "bits": round(float(A.info_gain(u, table[j])), 4)})
    for M in MS_CLOUD:
        out["cloud"][M] = cloud(table, float(M))

    eps, rd, _ = B.load2("study2")
    keep = [e for e in eps if not A.excluded(e)]
    for lam in range(1, 31):
        A.LAM = float(lam)
        mins, wrong = [], []
        for e in keep:
            for t in mixed:
                _, minutes, right, _ = A.replay("voi", e, t, rd, "routed")
                mins.append(minutes)
                wrong.append(0.0 if right else 1.0)
        out["curve"].append({"lam": lam, "minutes": round(float(np.mean(mins)), 3), "wrong": round(float(np.mean(wrong)), 4)})
    A.LAM = 12.0
    for M in MS_OPT:
        A.MISS = float(M)
        opts = [Optimal(t, float(M)) for t in mixed]
        out["optimum"][M] = round(float(np.mean([np.mean([replay_opt(o, e, rd, "routed")[0] for o in opts]) for e in keep])), 2)
    A.MISS = 60.0
    print(json.dumps(out))


if __name__ == "__main__":
    main()
