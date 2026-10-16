"""Sonnet alone, repeated on study 2 (exploratory, not preregistered).

The preregistered run of Sonnet alone is `sonnet-study2.jsonl`; `sonnet.py --run N` replays the same
episodes into `sonnet-study2-runN.jsonl`. This script reports each run, the spread across runs per
episode, and H5 recomputed against each run and against the per-episode mean of all runs.

Usage: python sonnet_repeats.py > results/sonnet-repeats.txt
"""

from __future__ import annotations

import json

import numpy as np

import analyze as A
import analyze2 as B

LEVEL = 0.975  # Bonferroni level of study 2
MARGIN = 2.0


def runs() -> dict[str, dict[int, dict]]:
    out = {}
    for path in sorted(A.OUT.glob("sonnet-study2*.jsonl")):
        label = "preregistered" if path.stem == "sonnet-study2" else path.stem.split("-")[-1]
        out[label] = {r["episode"]: r for r in map(json.loads, path.open())}
    return out


def main() -> None:
    hist, hist_readings, _ = A.load("main")
    hist = [e for e in hist if not A.excluded(e)]
    pos, run = A.counts(hist, hist_readings)
    meds = A.llm_medians()
    mixed = [A.table_mixed(pos, run, m, A.ebayes_k(pos, run, m)) for m in meds]
    episodes, readings, _ = B.load2("study2")
    keep = [e for e in episodes if not A.excluded(e)]
    hybrid = np.array([np.mean([A.replay("voi", e, t, readings, "routed")[0] for t in mixed]) for e in keep])

    R = runs()
    complete = {k: v for k, v in R.items() if all(e["episode"] in v and not v[e["episode"]].get("failed") for e in keep)}
    print(f"== Sonnet alone on study 2: runs found {sorted(R)}, complete {sorted(complete)}")
    print(f"   routed hybrid {hybrid.mean():.2f} min\n")
    print(f"  {'run':14} {'loss':>6} {'right':>7} {'checks':>7} {'$':>7}   H5 diff [{LEVEL:.1%} CI]")
    L = {}
    for k, v in complete.items():
        rows = [v[e["episode"]] for e in keep]
        L[k] = np.array([r["loss"] for r in rows])
        m, lo, hi = A.boot(hybrid - L[k], LEVEL)
        ok = "holds" if hi < MARGIN else "fails"
        print(f"  {k:14} {L[k].mean():6.2f} {sum(r['right'] for r in rows):3}/{len(rows)} "
              f"{np.mean([len(r['path']) for r in rows]):7.2f} {sum(r['dollars'] for r in rows):7.2f}   "
              f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}] {ok}")
    if len(L) < 2:
        return
    M = np.stack(list(L.values()))
    mean = M.mean(axis=0)
    m, lo, hi = A.boot(hybrid - mean, LEVEL)
    print(f"  {'mean of runs':14} {mean.mean():6.2f}{'':27}{m:+.2f} [{lo:+.2f}, {hi:+.2f}] "
          f"{'holds' if hi < MARGIN else 'fails'}")
    varies = (M.max(axis=0) != M.min(axis=0))
    paths = [len({tuple(complete[k][e['episode']]['path']) for k in complete}) > 1 for e in keep]
    answers = [len({complete[k][e['episode']]['answer'] for k in complete}) > 1 for e in keep]
    print(f"\n  episodes whose loss differs across runs: {varies.sum()} of {len(keep)}; "
          f"path differs: {sum(paths)}; named cause differs: {sum(answers)}")
    print(f"  mean per-episode standard deviation of the loss: {M.std(axis=0, ddof=1).mean():.2f} min; "
          f"spread of the run means: {M.mean(axis=1).min():.2f} to {M.mean(axis=1).max():.2f}")


if __name__ == "__main__":
    main()
