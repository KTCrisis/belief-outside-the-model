"""The exact Bayes-optimal policy replayed on the faults (exploratory, not preregistered).

For each table, exact dynamic programming over the belief and the checks left gives, at every step, the action of
least expected loss under that table: stop and name the likeliest cause (M x P(wrong)), or run the check whose
cost plus the expected loss after it is least. That policy is replayed on the recorded outputs exactly like the
rule S = cost - lambda x bits, with the same tables, readers and loss; the gap between the two is what the one-step
approximation costs on these faults. The optimum is optimal under the table, not under the faults: a table that
is wrong about the faults can make the "optimal" policy lose.

Readings of the routed reader are probabilities, not 0 or 1: the policy is evaluated at the belief they actually
produce (the value function is computed on demand for any belief), with outcomes treated as binary in the look-ahead.

A second part sweeps the price of an error M (2 to 240) and compares, on the same faults: guessing at once, the
rule at lambda = 12, the rule at lambda = M/4, the myopic Bayes decision that stops by M x P(wrong) (game A), a
hybrid that chooses the check by S but stops by game A's test on that check, and the optimum.

usage: python optimal_replay.py > results/optimal-replay.txt
"""
import numpy as np

import analyze as A
import analyze2 as B

RNG = np.random.default_rng(7)


class Optimal:
    """Exact optimum under one table: value(post, left) and the best action."""

    def __init__(self, table: np.ndarray, miss: float):
        self.t, self.miss, self.memo = table, miss, {}

    def value(self, post, left):
        k = (tuple(np.round(post, 9)), left)
        if k not in self.memo:
            best, act = self.miss * A.p_wrong(post), None
            for j in left:
                like = self.t[j]
                p_pos = float(post @ like)
                rest = tuple(x for x in left if x != j)
                v = A.COST[j]
                for r, p in ((1.0, p_pos), (0.0, 1 - p_pos)):
                    if p > 1e-12:
                        v += p * self.value(A.update(post, like, r), rest)[0]
                if v < best - 1e-12:
                    best, act = v, j
            self.memo[k] = (best, act)
        return self.memo[k]


def replay_opt(opt: Optimal, ep, readings, reader):
    post = np.full(A.NC, 1 / A.NC)
    left, minutes, run = tuple(range(A.NT)), 0.0, []
    while True:
        _, j = opt.value(post, left)
        if j is None:
            break
        left = tuple(x for x in left if x != j)
        run.append(j)
        minutes += A.COST[j]
        post = A.update(post, opt.t[j], A.reading(readings, ep, j, reader))
    right = A.CAUSES[int(post.argmax())] == ep["cause"]
    return minutes + (0.0 if right else A.MISS), minutes, right, run


def boot(d):
    idx = RNG.integers(0, len(d), size=(10_000, len(d)))
    m = d[idx].mean(axis=1)
    return d.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)


def main() -> None:
    hist, hr, _ = A.load("main")
    hist = [e for e in hist if not A.excluded(e)]
    pos, run = A.counts(hist, hr)
    meds = A.llm_medians()
    mixed = [A.table_mixed(pos, run, m, A.ebayes_k(pos, run, m)) for m in meds]
    opts = [Optimal(t, A.MISS) for t in mixed]
    eps, rd, _ = B.load2("study2")
    keep = [e for e in eps if not A.excluded(e)]

    print(f"study 2, {len(keep)} faults, the 10 mixed tables counted on study 1, M = {A.MISS:g}, lambda = {A.LAM:g}")
    print("reader       rule      loss   right  checks  minutes")
    res = {}
    for reader in ("reference", "routed"):
        for rule in ("voi", "optimal"):
            rows = []
            for e in keep:
                per = [replay_opt(o, e, rd, reader) if rule == "optimal" else A.replay("voi", e, t, rd, reader)
                       for o, t in zip(opts, mixed)]
                rows.append([np.mean([x[0] for x in per]), np.mean([x[2] for x in per]),
                             np.mean([len(x[3]) for x in per]), np.mean([x[1] for x in per])])
            rows = np.array(rows)
            res[(reader, rule)] = rows[:, 0]
            print(f"{reader:12} {rule:8} {rows[:, 0].mean():6.2f} {rows[:, 1].mean():6.1%} {rows[:, 2].mean():6.2f}"
                  f"  {rows[:, 3].mean():6.2f}")
    for reader in ("reference", "routed"):
        paths = losses = 0
        for e in keep:
            for o, t in zip(opts, mixed):
                a, b = replay_opt(o, e, rd, reader), A.replay("voi", e, t, rd, reader)
                paths += a[3] != b[3]
                losses += abs(a[0] - b[0]) > 1e-9
        print(f"{reader}: paths differ in {paths} of {len(keep) * len(mixed)} episode-table pairs, losses in {losses}")
    for reader in ("reference", "routed"):
        m, lo, hi = boot(res[(reader, "voi")] - res[(reader, "optimal")])
        print(f"{reader}: rule minus optimum {m:+.2f} [{lo:+.2f}, {hi:+.2f}] (95 % paired bootstrap, uncorrected)")
    m_sweep(mixed, keep, rd)


def hybrid_replay(ep, table, readings, reader):
    """Choose by S (bits, lambda = 12), stop by game A's test: run the chosen check only if its cost plus the
    expected M x P(wrong) after it is below M x P(wrong) now."""
    post = np.full(A.NC, 1 / A.NC)
    left, minutes, run = list(range(A.NT)), 0.0, []
    while left:
        j = min(left, key=lambda k: A.COST[k] - A.LAM * A.info_gain(post, table[k]))
        if not A.COST[j] + A.step_loss(post, table[j]) < A.MISS * A.p_wrong(post):
            break
        left.remove(j)
        run.append(j)
        minutes += A.COST[j]
        post = A.update(post, table[j], A.reading(readings, ep, j, reader))
    right = A.CAUSES[int(post.argmax())] == ep["cause"]
    return minutes + (0.0 if right else A.MISS), minutes, right, run


def m_sweep(mixed, keep, rd) -> None:
    print("\nprice of an error M, routed reader, study 2 (mean loss, minutes)")
    print("M     guess  rule l=12  rule l=M/4  game A  hybrid S/A  optimum")
    for M in (2, 4, 8, 15, 30, 60, 120, 240):
        A.MISS = float(M)

        def mean(f):
            return np.mean([np.mean([f(e, t) for t in mixed]) for e in keep])
        A.LAM = 12.0
        r12 = mean(lambda e, t: A.replay("voi", e, t, rd, "routed")[0])
        game_a = mean(lambda e, t: A.replay("bayes", e, t, rd, "routed")[0])
        hyb = mean(lambda e, t: hybrid_replay(e, t, rd, "routed")[0])
        A.LAM = M / 4
        rm4 = mean(lambda e, t: A.replay("voi", e, t, rd, "routed")[0])
        A.LAM = 12.0
        opts = [Optimal(t, float(M)) for t in mixed]
        opt = np.mean([np.mean([replay_opt(o, e, rd, "routed")[0] for o in opts]) for e in keep])
        print(f"{M:<5} {M * 5 / 6:5.2f}  {r12:9.2f}  {rm4:10.2f}  {game_a:6.2f}  {hyb:10.2f}  {opt:7.2f}")
    A.MISS = 60.0


if __name__ == "__main__":
    main()
