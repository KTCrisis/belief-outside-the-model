"""Where lambda comes from, on the real faults (exploratory): two ways to set the price of a bit.

1. From the optimum (volet I, section 7): exact dynamic programming over every belief reachable with the
   table counted on study 1 (seven checks: at most 3^7 states), then the slope a of V ~ a * H through the
   origin. a is the price of a bit the optimum itself implies; it needs no episode, only the table and M.
   The reading is taken as binary (the reference reader), as the table is counted.
2. From the history: replay the rule on study 1's closed incidents (two-fold, reference reader) for a grid
   of lambda, keep the lambda of least loss, then evaluate it on study 2 with the routed reader.

Both are done for M from 15 to 240. Nothing is re-run; this is not a preregistered test.

Reading: the rule S = cost - lambda x bits does not contain M, so its decisions depend on lambda alone; M
only prices the result. V0 is the loss the optimum expects under the counted table, which is noisier than
the faults (smoothed counts): realised losses fall below it.

Measured on 2026-10-09 (grids printed by hand alongside): on study 1 (reference reader, two folds) every
lambda from 6 to at least 24 gives the same loss, 9.33 (9.45 at 30); on study 2 (routed reader) the plateau
runs from 6 to 14, and 15 or more over-investigates (10.53 at 15, 15.99 at 30); 3 or less runs no check, 4 and
5 too few (33.67 and 17.67 on study 1 at M = 60).

usage: python lambda_fit.py > results/lambda-fit.txt
"""
import numpy as np

import analyze as A
import analyze2 as B

MS = [15, 30, 60, 120, 240]
GRID = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 24, 30, 36, 48]


def slope(table: np.ndarray, miss: float) -> tuple[float, float, float]:
    """Exact optimum over the table: (V0, slope a of V ~ a * H, R^2)."""
    memo: dict = {}

    def value(post, left):
        k = (tuple(np.round(post, 10)), left)
        if k in memo:
            return memo[k][0]
        best = miss * A.p_wrong(post)
        for j in left:
            like = table[j]
            p_pos = float(post @ like)
            rest = tuple(x for x in left if x != j)
            ev = 0.0
            for r, p in ((1.0, p_pos), (0.0, 1 - p_pos)):
                if p > 1e-12:
                    ev += p * value(A.update(post, like, r), rest)
            best = min(best, A.COST[j] + ev)
        memo[k] = (best, A.entropy(post))
        return best

    v0 = value(np.full(A.NC, 1 / A.NC), tuple(range(A.NT)))
    V, H = np.array([v for v, _ in memo.values()]), np.array([h for _, h in memo.values()])
    a = float(H @ V / (H @ H))
    res = V - a * H
    return v0, a, float(1 - res @ res / ((V - V.mean()) @ (V - V.mean())))


def main() -> None:
    hist, hist_readings, _ = A.load("main")
    hist = [e for e in hist if not A.excluded(e)]
    pos, run = A.counts(hist, hist_readings)
    meds = A.llm_medians()
    mixed = [A.table_mixed(pos, run, m, A.ebayes_k(pos, run, m)) for m in meds]
    table = np.mean(mixed, axis=0)

    n_blocks = max(e["episode"] for e in hist) // 6 + 1
    folds = {}
    for f in (0, 1):
        other = [e for e in hist if A.fold_of(e, n_blocks) != f]
        p_, r_ = A.counts(other, hist_readings)
        folds[f] = [A.table_mixed(p_, r_, m, A.ebayes_k(p_, r_, m)) for m in meds]

    episodes2, readings2, _ = B.load2("study2")
    keep2 = [e for e in episodes2 if not A.excluded(e)]

    def loss1():
        return np.mean([np.mean([A.replay("voi", e, t, hist_readings, "reference")[0]
                                 for t in folds[A.fold_of(e, n_blocks)]]) for e in hist])

    def loss2():
        return np.mean([np.mean([A.replay("voi", e, t, readings2, "routed")[0] for t in mixed]) for e in keep2])

    print("M     optimum V0   slope a (min/bit)   R2    M/a  |  lambda of least loss on study 1   "
          "study 2 with it   study 2 at lambda = a   study 2 at lambda = 12")
    for M in MS:
        A.MISS = float(M)
        v0, a, r2 = slope(table, M)
        best, best_loss = None, None
        for lam in GRID:
            A.LAM = float(lam)
            l1 = loss1()
            if best_loss is None or l1 < best_loss - 1e-9:
                best, best_loss = lam, l1
        A.LAM = float(best)
        s2_best = loss2()
        A.LAM = a
        s2_a = loss2()
        A.LAM = 12.0
        s2_12 = loss2()
        print(f"{M:<5} {v0:8.2f}     {a:8.2f}          {r2:5.2f}  {M / a:4.1f}  |  {best:>4} ({best_loss:6.2f})"
              f"                    {s2_best:6.2f}            {s2_a:6.2f}                {s2_12:6.2f}")


if __name__ == "__main__":
    main()
