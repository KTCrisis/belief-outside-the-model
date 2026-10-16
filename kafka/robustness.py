"""Two robustness checks asked for after the studies (exploratory, not preregistered).

1. The reader's implicit prior. The update r*l + (1 - r)*(1 - l) treats a reading r as virtual
   evidence whose likelihood ratio is r / (1 - r), i.e. as if the reader started from 0.5. If the
   reader behaves as if the signal had prior pi, the ratio it carries is odds(r) / odds(pi). We
   estimate pi against the reference: logit P(signal | r) = logit r - logit pi + logit rho, with
   rho the base rate of the signal, a logistic fit whose slope is fixed at 1. A free slope (Platt)
   is printed as a calibration diagnostic. The corrected reading sigmoid(logit r - logit pi) is
   then replayed, with pi fitted out of sample (other fold in study 1, study 1 for study 2).

2. The order of the checks. Outputs were recorded in a fixed order, the restart last, so a
   replayed policy that runs the restart before another check sees that check's pre-restart
   output. We count how often this happens per arm.

Usage: python robustness.py > results/robustness.txt
"""

from __future__ import annotations

import numpy as np

import analyze as A
import analyze2 as B

RESTART = A.CHECKS.index("restart_consumer")
logit = lambda x: np.log(x / (1 - x))  # noqa: E731
sigmoid = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


# --------------------------------------------------------------------------- #
#  1. reader prior
# --------------------------------------------------------------------------- #
def pairs(episodes, readings, reader):
    """(clipped reading, reference) over every check of every episode, plus episode ids."""
    r, s, ids = [], [], []
    for e in episodes:
        for j in range(A.NT):
            row = readings.get((e["episode"], A.CHECKS[j]))
            if row is None or reader not in row:
                continue
            r.append(A.reading(readings, e, j, reader))
            s.append(row["reference"])
            ids.append(e["episode"])
    return np.array(r), np.array(s, float), np.array(ids)


def fit(z, s, slope_free=False):
    """Logistic fit of s on z, Newton; returns (slope, intercept), slope fixed at 1 unless freed."""
    a, b = 1.0, 0.0
    for _ in range(50):
        p = sigmoid(a * z + b)
        w = p * (1 - p) + 1e-12
        if slope_free:
            X = np.stack([z, np.ones_like(z)], 1)
            H = X.T @ (X * w[:, None])
            step = np.linalg.solve(H, X.T @ (s - p))
            a, b = a + step[0], b + step[1]
        else:
            b += (s - p).sum() / w.sum()
    return a, b


def pi_of(r, s):
    rho = s.mean()
    _, b = fit(logit(r), s)
    return float(sigmoid(logit(rho) - b))


def pi_boot(r, s, ids, n=2000):
    eps = np.unique(ids)
    rows = {e: np.flatnonzero(ids == e) for e in eps}
    out = []
    for _ in range(n):
        pick = np.concatenate([rows[e] for e in A.RNG.choice(eps, len(eps))])
        out.append(pi_of(r[pick], s[pick]))
    return np.percentile(out, [2.5, 97.5])


def corrected(readings, episodes, reader, pi, key):
    """Copy of the readings with a `key` column holding sigmoid(logit r - logit pi)."""
    out = {k: dict(v) for k, v in readings.items()}
    for e in episodes:
        for j in range(A.NT):
            row = out.get((e["episode"], A.CHECKS[j]))
            if row is not None and reader in row:
                row[key] = float(sigmoid(logit(A.reading(readings, e, j, reader)) - logit(pi)))
    return out


def describe(label, episodes, readings, reader):
    r, s, ids = pairs(episodes, readings, reader)
    pi = pi_of(r, s)
    lo, hi = pi_boot(r, s, ids)
    a, b = fit(logit(r), s, slope_free=True)
    print(f"  {label:8} {reader:8} n={len(r):4}  base rate {s.mean():.3f}  pi {pi:.3f} [{lo:.3f}, {hi:.3f}]  "
          f"Platt slope {a:.2f}, intercept {b:+.2f}")
    return pi


def study1_tables():
    episodes, readings, _ = A.load("main")
    n_blocks = max(e["episode"] for e in episodes) // 6 + 1
    keep = [e for e in episodes if not A.excluded(e)]
    meds = A.llm_medians()
    folds = [[e for e in keep if A.fold_of(e, n_blocks) == f] for f in (0, 1)]
    T = {}
    for f in (0, 1):
        pos, run = A.counts(folds[1 - f], readings)
        ks = [A.ebayes_k(pos, run, m) for m in meds]
        T[f] = {"mixed_eb": [A.table_mixed(pos, run, m, k) for m, k in zip(meds, ks)],
                "eps": [A.table_eps(pos, run)]}
    return keep, readings, folds, n_blocks, T


def study2_tables():
    hist, hist_readings, _ = A.load("main")
    hist = [e for e in hist if not A.excluded(e)]
    pos, run = A.counts(hist, hist_readings)
    meds = A.llm_medians()
    ks = [A.ebayes_k(pos, run, m) for m in meds]
    mixed = [A.table_mixed(pos, run, m, k) for m, k in zip(meds, ks)]
    episodes, readings, sonnet = B.load2("study2")
    keep = [e for e in episodes if not A.excluded(e)]
    return keep, readings, sonnet, mixed, hist, hist_readings


def mean_loss(episodes, tables_of, readings, reader, rule="voi"):
    return np.array([np.mean([A.replay(rule, e, t, readings, reader)[0] for t in tables_of(e)]) for e in episodes])


def reader_prior() -> None:
    print("== 1. The reader's implicit prior (pi = 0.5 assumed by the update)")
    keep1, rd1, folds, n_blocks, T = study1_tables()
    keep2, rd2, _, mixed, hist, hist_rd = study2_tables()
    print("\nEstimates against the reference (95% bootstrap over episodes):")
    for reader in ("jev", "haiku"):
        describe("study 1", keep1, rd1, reader)
    for reader in ("jev", "haiku", "routed"):
        describe("study 2", keep2, rd2, reader)

    print("\nJev per check, study 1:")
    for j, c in enumerate(A.CHECKS):
        r, s, _ = pairs(keep1, {k: v for k, v in rd1.items() if k[1] == c}, "jev")
        if s.min() == s.max():
            print(f"  {c:22} signal always {int(s[0])}: pi not identifiable")
            continue
        print(f"  {c:22} base rate {s.mean():.2f}  pi {pi_of(r, s):.3f}")

    print("\nReplay with the corrected reading (mean loss, minutes; 95% paired bootstrap, uncorrected):")
    for reader in ("jev", "haiku"):
        rd = rd1
        for f in (0, 1):  # pi from the other fold
            r, s, _ = pairs(folds[1 - f], rd1, reader)
            rd = corrected(rd, folds[f], reader, pi_of(r, s), reader + "_pi")
        tab = lambda e: T[A.fold_of(e, n_blocks)]["mixed_eb"]  # noqa: E731
        base = mean_loss(keep1, tab, rd, reader)
        corr = mean_loss(keep1, tab, rd, reader + "_pi")
        m, lo, hi = A.boot(corr - base, 0.95)
        print(f"  study 1  {reader:8} {base.mean():6.2f} -> {corr.mean():6.2f}   diff {m:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    for reader in ("jev", "haiku", "routed"):
        if reader == "routed":  # no routed reading in study 1: pi from the first half of study 2
            first = [e for e in keep2 if e["episode"] < 60]
            r, s, _ = pairs(first, rd2, reader)
            eps = [e for e in keep2 if e["episode"] >= 60]
            note = " (pi from episodes 0-59, replay on 60-119)"
        else:
            r, s, _ = pairs(hist, hist_rd, reader)
            eps, note = keep2, ""
        rd = corrected(rd2, eps, reader, pi_of(r, s), reader + "_pi")
        base = mean_loss(eps, lambda e: mixed, rd, reader)
        corr = mean_loss(eps, lambda e: mixed, rd, reader + "_pi")
        m, lo, hi = A.boot(corr - base, 0.95)
        print(f"  study 2  {reader:8} {base.mean():6.2f} -> {corr.mean():6.2f}   diff {m:+.2f} [{lo:+.2f}, {hi:+.2f}]{note}")


# --------------------------------------------------------------------------- #
#  2. order of the checks
# --------------------------------------------------------------------------- #
def restart_early(run) -> bool:
    return RESTART in run and run.index(RESTART) < len(run) - 1


def share(runs) -> str:
    n = sum(map(restart_early, runs))
    return f"{n:4} of {len(runs):4} ({100 * n / max(len(runs), 1):5.1f}%)"


def order() -> None:
    print("\n== 2. Restart run before another check (that check's output was recorded before the restart)")
    keep1, rd1, _, n_blocks, T = study1_tables()
    keep2, rd2, sonnet2, mixed, _, _ = study2_tables()
    _, _, sonnet1 = A.load("main")
    tab = lambda e, k: T[A.fold_of(e, n_blocks)][k]  # noqa: E731
    print("  episode-table pairs, study 1:")
    for rule, tk, reader in (("voi", "mixed_eb", "jev"), ("voi", "mixed_eb", "reference"),
                             ("hesp", "eps", "jev"), ("voi", "mixed_eb", "haiku")):
        runs = [A.replay(rule, e, t, rd1, reader)[3] for e in keep1 for t in tab(e, tk)]
        print(f"    {rule:5} {tk:9} {reader:10} {share(runs)}")
    print("  episode-table pairs, study 2:")
    for reader in ("routed", "reference", "jev"):
        runs = [A.replay("voi", e, t, rd2, reader)[3] for e in keep2 for t in mixed]
        print(f"    voi   mixed_eb  {reader:10} {share(runs)}")
    for label, son in (("study 1", sonnet1), ("study 2", sonnet2)):
        runs = [[A.CHECKS.index(c) for c in r["path"] if c in A.CHECKS] for r in son.values()]
        print(f"  Sonnet alone, {label}: episodes {share(runs)}")


if __name__ == "__main__":
    reader_prior()
    order()
