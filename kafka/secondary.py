"""Preregistered secondary endpoints that analyze.py / analyze2.py print only in part.

Exploratory, uncorrected. The frozen analysis scripts stay untouched; this one imports
their functions and completes three endpoints:

- study 1: HESP's own metric (share correct) for every rule and table, not only four pairs;
- study 1: the designer table against the counted table, cell by cell (which hand-written
  values reality contradicts);
- study 2: the pull toward the suspected cause per check and per named cause.

Usage: python secondary.py > results/secondary.txt
"""

from __future__ import annotations

import numpy as np

import analyze as A

RULES = ["voi", "ratio_priced", "hesp", "bayes", "cheapest", "most_informative"]
TABLES = ["designer", "llm", "counted", "eps", "mixed10", "mixed_eb"]


def study1() -> None:
    episodes, readings, _ = A.load("main")
    n_blocks = max(e["episode"] for e in episodes) // 6 + 1
    keep = [e for e in episodes if not A.excluded(e)]
    meds = A.llm_medians()
    folds = [[e for e in keep if A.fold_of(e, n_blocks) == f] for f in (0, 1)]
    T, counted = {}, {}
    for f in (0, 1):
        pos, run = A.counts(folds[1 - f], readings)
        ks = [A.ebayes_k(pos, run, m) for m in meds]
        counted[f] = (pos, run)
        T[f] = {"designer": A.DESIGNER, "counted": np.clip((pos + 1) / (run + 2), 0.01, 0.99),
                "eps": A.table_eps(pos, run), "llm": meds,
                "mixed10": [A.table_mixed(pos, run, m, 10) for m in meds],
                "mixed_eb": [A.table_mixed(pos, run, m, k) for m, k in zip(meds, ks)]}

    print("== study 1 (main), Jev reader")
    print("\nHESP's own metric (share of episodes ending on the true cause), every rule x table")
    print("  " + " " * 16 + "".join(f"{t:>10}" for t in TABLES))
    for rule in RULES:
        row = []
        for tk in TABLES:
            acc = []
            for e in keep:
                t = T[A.fold_of(e, n_blocks)][tk]
                ts = t if isinstance(t, list) else [t]
                acc.append(np.mean([A.replay(rule, e, ti, readings, "jev")[2] for ti in ts]))
            row.append(f"{np.mean(acc):10.3f}")
        print(f"  {rule:16}" + "".join(row))

    # Designer against reality: counted on all 120 episodes with the reference reader,
    # so that the comparison is about the system, not about a reader's errors.
    pos = sum(counted[f][0] for f in (0, 1))
    run = sum(counted[f][1] for f in (0, 1))
    obs = pos / np.maximum(run, 1)
    print("\nDesigner table against the counted table (all 120 episodes, reference reading)")
    print("  cells where the designer and the observed frequency fall on opposite sides of 0.5:")
    n = 0
    for j, c in enumerate(A.CHECKS):
        for i, cause in enumerate(A.CAUSES):
            d, o = A.DESIGNER[j, i], obs[j, i]
            if (d - 0.5) * (o - 0.5) < 0:
                n += 1
                print(f"    {c:22} | {cause:18} designer {d:.2f}  observed {o:.2f}  ({int(pos[j, i])}/{int(run[j, i])})")
    gap = np.abs(A.DESIGNER - obs)
    print(f"  {n} of {gap.size} cells contradicted; mean absolute gap {gap.mean():.3f}, "
          f"cells off by more than 0.3: {(gap > 0.3).sum()}")


def study2() -> None:
    import analyze2 as B

    # the study 2 tables, built exactly as in analyze2.main (counted on study 1)
    hist, hist_readings, _ = A.load("main")
    hist = [e for e in hist if not A.excluded(e)]
    pos, run = A.counts(hist, hist_readings)
    meds = A.llm_medians()
    ks = [A.ebayes_k(pos, run, m) for m in meds]
    mean_table = np.mean([A.table_mixed(pos, run, m, k) for m, k in zip(meds, ks)], axis=0)
    episodes, readings, _ = B.load2("study2")
    keep = [e for e in episodes if not A.excluded(e)]
    print("\n== study 2: pull toward the suspected cause, per check and per named cause")
    for informed, blind, cause in [("haiku_context", "haiku", "haiku_context_cause"),
                                   ("haiku_seq", "haiku_seq_blind", "haiku_seq_cause"),
                                   ("haiku_seqn", "haiku_seqn_blind", "haiku_seqn_cause")]:
        for ref in (0, 1):
            by_check, by_cause = {}, {}
            for e in keep:
                for j, c in enumerate(A.CHECKS):
                    row = readings.get((e["episode"], c))
                    if not row or row["reference"] != ref or blind not in row or informed not in row:
                        continue
                    named = A.CAUSES.index(row[cause])
                    v = (row[informed] - row[blind]) * (2 * mean_table[j, named] - 1)
                    by_check.setdefault(c, []).append(v)
                    by_cause.setdefault(row[cause], []).append(v)
            label = "present" if ref else "absent"
            print(f"  {informed} | reference {label}")
            for name, d in (("check", by_check), ("named cause", by_cause)):
                for k, vals in sorted(d.items()):
                    m, lo, hi = A.boot(np.array(vals), 0.95)
                    print(f"    {name:11} {k:22} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  (n={len(vals)})")


if __name__ == "__main__":
    study1()
    study2()
