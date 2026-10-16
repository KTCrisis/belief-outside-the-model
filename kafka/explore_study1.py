"""Exploratory checks on study 1 (not preregistered; the frozen analysis is analyze.py).

1. Fold by fold: H1, H2, H3 differences per fold, then inside fold B before and
   during the healthy-lag drift noted in results/run-notes.md (01:40-05:40 CEST).
2. Empirical k: analyze.py searches k on a grid starting at 1 and lands on 1.0
   in both folds. Profile the marginal likelihood below 1, look at how pure the
   counted cells are, and replay H1 at several fixed k.

Usage: python explore_study1.py --tag main
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
from scipy.special import betaln

import analyze as A

DRIFT_START = datetime(2026, 10, 5, 23, 40, tzinfo=timezone.utc)  # 01:40 CEST
DRIFT_END = datetime(2026, 10, 6, 3, 40, tzinfo=timezone.utc)      # 05:40 CEST
LEVEL = 0.95


def loglik(pos, run, q, k):
    qq = np.clip(q, 0.01, 0.99)
    a, b = k * qq, k * (1 - qq)
    return float((betaln(pos + a, run - pos + b) - betaln(a, b)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    tag = ap.parse_args().tag
    episodes, readings, sonnet = A.load(tag)
    n_blocks = max(e["episode"] for e in episodes) // 6 + 1
    keep = [e for e in episodes if not A.excluded(e)]
    folds = [[e for e in keep if A.fold_of(e, n_blocks) == f] for f in (0, 1)]
    meds = A.llm_medians()

    counted = {}
    for f in (0, 1):
        pos, run = A.counts(folds[1 - f], readings)
        ks = [A.ebayes_k(pos, run, m) for m in meds]
        counted[f] = (pos, run, ks)

    def table_set(f, k=None):
        pos, run, ks = counted[f]
        return [A.table_mixed(pos, run, m, k if k is not None else kk) for m, kk in zip(meds, ks)]

    def losses(eps, rule, kind, reader, k=None):
        out = []
        for e in eps:
            f = A.fold_of(e, n_blocks)
            if kind == "mixed":
                ts = table_set(f, k)
            elif kind == "llm":
                ts = meds
            else:  # eps
                pos, run, _ = counted[f]
                ts = [A.table_eps(pos, run)]
            out.append(np.mean([A.replay(rule, e, t, readings, reader)[0] for t in ts]))
        return np.array(out)

    def sonnet_loss(eps):
        return np.array([sonnet[e["episode"]]["loss"] if e["episode"] in sonnet
                         and not sonnet[e["episode"]].get("failed") else np.nan for e in eps])

    def report(label, eps):
        h1a, h1b = losses(eps, "voi", "mixed", "jev"), losses(eps, "voi", "llm", "jev")
        h2a, h2b = losses(eps, "ratio_priced", "eps", "jev"), losses(eps, "hesp", "eps", "jev")
        s = sonnet_loss(eps)
        ok = ~np.isnan(s)
        lags = [e["healthy_lag"] for e in eps]
        print(f"  {label:26} n={len(eps):3}  healthy lag median {np.median(lags):6.0f}")
        for name, d in [("H1 mixed - LLM", h1a - h1b), ("H2 priced - HESP", h2a - h2b),
                        ("H3 hybrid - Sonnet", h1a[ok] - s[ok])]:
            m, lo, hi = A.boot(d, LEVEL)
            print(f"      {name:20} {m:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]")
        ref = losses(eps, "voi", "mixed", "reference")
        print(f"      {'hybrid, reference':20} {ref.mean():6.2f}   Sonnet {np.nanmean(s):6.2f}")

    print(f"== {tag}: exploratory, {LEVEL:.0%} paired bootstrap intervals\n")
    print("1. Fold by fold, and the healthy-lag drift inside fold B")
    report("all", keep)
    report("fold A", folds[0])
    report("fold B", folds[1])
    when = lambda e: datetime.fromisoformat(e["injected"])  # noqa: E731
    in_drift = [e for e in folds[1] if DRIFT_START <= when(e) <= DRIFT_END]
    out_drift = [e for e in folds[1] if not DRIFT_START <= when(e) <= DRIFT_END]
    report("fold B, outside the drift", out_drift)
    report("fold B, during the drift", in_drift)
    print("  per cause, during the drift: " + ", ".join(
        f"{c} {sum(e['cause'] == c for e in in_drift)}" for c in A.CAUSES))

    print("\n2. Empirical k below the grid floor")
    grid = np.concatenate([np.geomspace(0.01, 1, 15), A.K_GRID[1:]])
    for f in (0, 1):
        pos, run, ks = counted[f]
        rate = np.divide(pos, run, out=np.full_like(pos, np.nan), where=run > 0)
        pure = np.isclose(rate, 0) | np.isclose(rate, 1)
        print(f"  fold {'AB'[f]} (tables counted on the other fold): {pure.sum()}/{pure.size} cells pure "
              f"(all 0 or all 1); mixed cells: "
              + (", ".join(f"{A.CHECKS[j]}|{A.CAUSES[c]} {pos[j, c]:.0f}/{run[j, c]:.0f}"
                           for j, c in zip(*np.where(~pure))) or "none"))
        kstar = [float(grid[np.argmax([loglik(pos, run, m, k) for k in grid])]) for m in meds]
        ll = np.array([[loglik(pos, run, m, k) for k in (0.01, 0.1, 1, 10)] for m in meds]).mean(axis=0)
        print(f"      k* on an extended grid: median {np.median(kstar):.2f} (min {min(kstar):.2f}, max {max(kstar):.2f})")
        print("      mean log-likelihood at k = 0.01, 0.1, 1, 10: " + ", ".join(f"{v:.1f}" for v in ll))
    print("  H1 loss (voi, Jev reader) with k fixed, mixed table vs LLM table "
          f"{losses(keep, 'voi', 'llm', 'jev').mean():.2f}:")
    for k in (0.01, 0.1, 0.5, 1, 3, 10, 30):
        print(f"      k = {k:5}: jev {losses(keep, 'voi', 'mixed', 'jev', k).mean():6.2f}   "
              f"reference {losses(keep, 'voi', 'mixed', 'reference', k).mean():6.2f}")


if __name__ == "__main__":
    main()
