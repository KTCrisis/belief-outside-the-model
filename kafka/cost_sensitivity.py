"""Check costs, conventional against measured (exploratory, not preregistered; nothing re-run).

The studies price each check by convention, in an engineer's minutes (1 to 8). The lab also recorded how long each
probe actually took on the machine (`probe_seconds` in the episodes). This replays the hybrid of study 2 (routed
reader, the 10 mixed tables counted on study 1, M = 60) under both cost scales:

- conventional: the minutes of the studies;
- measured: the mean probe duration on the lab, in minutes (seconds / 60).

For each scale: running every check, the rule S at lambda = 12 and at the best lambda of a grid, the two-criteria
rule (choose by S, stop by M x P(wrong)), and the exact optimum under the table. Sonnet alone cannot be replayed:
its choices were made under the conventional costs of its prompt.

usage: python cost_sensitivity.py > results/cost-sensitivity.txt
"""
import json

import numpy as np

import analyze as A
import analyze2 as B
from optimal_replay import Optimal, hybrid_replay, replay_opt

GRID = [0.01, 0.03, 0.1, 0.3, 1, 2, 4, 6, 8, 12, 16, 24]


def main() -> None:
    hist, hr, _ = A.load("main")
    hist = [e for e in hist if not A.excluded(e)]
    pos, run = A.counts(hist, hr)
    mixed = [A.table_mixed(pos, run, m, A.ebayes_k(pos, run, m)) for m in A.llm_medians()]
    eps, rd, _ = B.load2("study2")
    keep = [e for e in eps if not A.excluded(e)]
    probes = [json.loads(line) for line in open("results/episodes-study2.jsonl")]
    measured = np.array([np.mean([p["probe_seconds"][c] for p in probes]) / 60 for c in A.CHECKS])
    scales = {"conventional": A.COST.copy(), "measured": measured}
    A.MISS = 60.0

    def mean(f):
        return np.mean([np.mean([f(e, t) for t in mixed]) for e in keep])

    print("check costs (minutes):")
    for c, a, b in zip(A.CHECKS, scales["conventional"], scales["measured"]):
        print(f"  {c:22} conventional {a:4.1f}   measured {b:6.3f}")
    for name, cost in scales.items():
        A.COST = cost
        print(f"\n== {name} costs, study 2, routed reader, M = 60 (mean loss, checks)")
        print(f"  run every check            {cost.sum():7.3f}   (7 checks, no error with this reader)")
        for lam in GRID:
            A.LAM = float(lam)
            loss = mean(lambda e, t: A.replay("voi", e, t, rd, "routed")[0])
            checks = mean(lambda e, t: len(A.replay("voi", e, t, rd, "routed")[3]))
            print(f"  rule S, lambda = {lam:<6}    {loss:7.3f}   {checks:4.2f}")
        A.LAM = 12.0
        two = mean(lambda e, t: hybrid_replay(e, t, rd, "routed")[0])
        print(f"  two criteria, lambda = 12  {two:7.3f}")
        opts = [Optimal(t, A.MISS) for t in mixed]
        opt = np.mean([np.mean([replay_opt(o, e, rd, "routed")[0] for o in opts]) for e in keep])
        print(f"  exact optimum (table)      {opt:7.3f}")


if __name__ == "__main__":
    main()
