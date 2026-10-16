"""Numbers quoted in the paper that no other results file prints (exploratory, offline).

Nothing is re-run and no model is called: everything is recomputed from the recorded files.

1. Appendix C, setting lambda from the history: the lambda grids behind lambda_fit.py's docstring (the
   plateau of equal loss on study 1, reference reader, two folds; and on study 2, routed reader), at M = 60.
2. Appendix D, cost per diagnosis: the routed hybrid with Haiku 4.5 and Haiku 5.5 extractions (recorded
   *_usd of the derived checks run, on the mean table, as analyze2.py does), plus the bound for Jev's
   readings (three calls at most, under 0.00004 $ each, figures.py), against the arms acting alone.
3. Section 3.1, 3.4 and Appendix C, companion note: the simulated bench figures, as percentages of the
   exact optimum, from results-exact.txt, results-internet-exact.txt, results-clinic-exact.txt,
   results-m-sweep.txt and exp2/results/tables-summary.txt.

usage: python paper_numbers.py > results/paper-numbers.txt
"""
import json
import re
from pathlib import Path

import numpy as np

import analyze as A
import analyze2 as B

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 18, 20, 24, 30, 36, 48]
JEV_CALL = 0.00004  # figures.py: bound per Jev call, at most three per routed diagnosis


def lambda_grids() -> None:
    hist, hist_readings, _ = A.load("main")
    hist = [e for e in hist if not A.excluded(e)]
    pos, run = A.counts(hist, hist_readings)
    meds = A.llm_medians()
    mixed = [A.table_mixed(pos, run, m, A.ebayes_k(pos, run, m)) for m in meds]
    n_blocks = max(e["episode"] for e in hist) // 6 + 1
    folds = {}
    for f in (0, 1):
        other = [e for e in hist if A.fold_of(e, n_blocks) != f]
        p_, r_ = A.counts(other, hist_readings)
        folds[f] = [A.table_mixed(p_, r_, m, A.ebayes_k(p_, r_, m)) for m in meds]
    episodes2, readings2, _ = B.load2("study2")
    keep2 = [e for e in episodes2 if not A.excluded(e)]

    A.MISS = 60.0
    print("== 1. Appendix C: loss of the rule against lambda, M = 60")
    print("   study 1: reference reader, two folds (as lambda_fit.py); study 2: routed reader, tables of study 1")
    print(f"   {'lambda':>6} {'study 1':>8} {'checks':>6} {'study 2':>8} {'checks':>6}")
    for lam in GRID:
        A.LAM = float(lam)
        r1 = [[A.replay("voi", e, t, hist_readings, "reference") for t in folds[A.fold_of(e, n_blocks)]] for e in hist]
        r2 = [[A.replay("voi", e, t, readings2, "routed") for t in mixed] for e in keep2]
        l1 = np.mean([np.mean([x[0] for x in xs]) for xs in r1])
        c1 = np.mean([np.mean([len(x[3]) for x in xs]) for xs in r1])
        l2 = np.mean([np.mean([x[0] for x in xs]) for xs in r2])
        c2 = np.mean([np.mean([len(x[3]) for x in xs]) for xs in r2])
        print(f"   {lam:>6} {l1:8.2f} {c1:6.2f} {l2:8.2f} {c2:6.2f}")
    A.LAM = 12.0


def costs() -> None:
    hist, hist_readings, _ = A.load("main")
    hist = [e for e in hist if not A.excluded(e)]
    pos, run = A.counts(hist, hist_readings)
    meds = A.llm_medians()
    mean_table = np.mean([A.table_mixed(pos, run, m, A.ebayes_k(pos, run, m)) for m in meds], axis=0)
    episodes, readings, sonnet = B.load2("study2")
    keep = [e for e in episodes if not A.excluded(e)]
    A.MISS, A.LAM = 60.0, 12.0

    # the exploratory file repeats routed_h55 on the rows that carry routed_oss: check the copies agree
    seen, clash = {}, 0
    for l in (HERE / "results/readings2x-study2.jsonl").open():
        r = json.loads(l)
        if "routed_h55_usd" in r:
            k = (r["episode"], r["check"])
            clash += k in seen and seen[k] != (r["routed_h55"], r["routed_h55_usd"])
            seen[k] = (r["routed_h55"], r["routed_h55_usd"])
    print("\n== 2. Appendix D: dollars per diagnosis, study 2, 120 faults")
    print(f"   routed_h55 rows with a cost: {len(seen)} (episode, check) pairs, {clash} disagreeing copies")

    def extraction(reader):
        per = []
        for e in keep:
            _, _, path, _ = B.replay_post(e, mean_table, readings, reader)
            per.append(sum(readings[(e["episode"], A.CHECKS[j])].get(f"{reader}_usd", 0.0) for j in path))
        return float(np.mean(per)), float(np.max(per))

    h45, h45_max = extraction("routed")
    h55, h55_max = extraction("routed_h55")
    alone = {}
    for name, f in [("Sonnet 5.5 alone", "sonnet-study2.jsonl"), ("Haiku 5.5 alone", "claude-haiku-5-5-study2.jsonl"),
                    ("Haiku 4.5 alone", "claude-haiku-4-5-study2.jsonl")]:
        rows = [json.loads(l) for l in (HERE / "results" / f).open()]
        alone[name] = (float(np.mean([r.get("dollars", 0.0) for r in rows])), float(np.mean([r["loss"] for r in rows])))
    jev = 3 * JEV_CALL
    print(f"   routed, Haiku 4.5 extraction: mean {h45:.5f} $ (max {h45_max:.5f}), with Jev bound < {h45 + jev:.5f} $")
    print(f"   routed, Haiku 5.5 extraction: mean {h55:.5f} $ (max {h55_max:.5f}), with Jev bound < {h55 + jev:.5f} $")
    for name, (usd, loss) in alone.items():
        print(f"   {name:18} mean {usd:.5f} $   loss {loss:.2f}")
    s, h = alone["Sonnet 5.5 alone"][0], alone["Haiku 5.5 alone"][0]
    print(f"   extraction Haiku 4.5 / Haiku 5.5: {h45 / h55:.1f}x  (bounds with Jev: {(h45 + jev) / (h55 + jev):.1f}x)")
    print(f"   Haiku 5.5 alone / routed Haiku 5.5: {h / (h55 + jev):.1f}x with the Jev bound, {h / h55:.1f}x without")
    print(f"   Sonnet alone / routed Haiku 5.5: {s / (h55 + jev):.0f}x with the Jev bound, {s / h55:.0f}x without")


def bench() -> None:
    print("\n== 3. Companion note (simulated bench, exact expectations)")
    for f in ["results-exact.txt", "results-internet-exact.txt", "results-clinic-exact.txt"]:
        txt = (ROOT / f).read_text()
        opt = float(re.search(r"OPTIMUM \(DP\)\s+([\d.]+)", txt).group(1))
        la = float(re.search(r"^least_action\s+([\d.]+)", txt, re.M).group(1))
        head = txt.splitlines()[0]
        print(f"   {f:28} {head.split('(')[0].strip()}: least_action {la:.2f} vs optimum {opt:.2f}"
              f"  gap {100 * (la - opt) / opt:+.1f}%")
    txt = (ROOT / "results-m-sweep.txt").read_text()
    print("   results-m-sweep.txt, lambda = M/4 (least_action), gap to the optimum at each M:")
    for block in txt.split("scenario ")[1:]:
        name, m0 = re.match(r"(\w+), base M = ([\d.]+)", block).groups()
        cells = []
        for row in re.findall(r"^\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+ \|\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+) \|", block, re.M):
            cells.append(f"M = {float(row[0]):g} ({round(float(row[0]) / float(m0), 2):g}x): {row[1]}%")
        print(f"     {name:9} " + ", ".join(cells))
    txt = (ROOT / "exp2/results/tables-summary.txt").read_text()
    for name in ["kafka", "internet"]:
        part = txt.split(f"== {name}")[1]
        truth = float(re.search(r"truth\s+([\d.]+)", part).group(1))
        lo, hi = map(float, re.search(r"LLM median of 5\s+[\d.]+\s+min\s+([\d.]+)\s+max\s+([\d.]+)", part).groups())
        one = float(re.search(r"LLM single draw\s+[\d.]+\s+min\s+[\d.]+\s+max\s+([\d.]+)", part).group(1))
        print(f"   tables-summary.txt {name:8}: true table {truth:.2f}; worst LLM median-of-5 table {hi:.2f} "
              f"({100 * (hi - truth) / truth:+.0f}% vs true table, {100 * (hi - lo) / lo:+.0f}% vs best median "
              f"{lo:.2f}); worst single draw {one:.2f} ({100 * (one - truth) / truth:+.0f}%)")


if __name__ == "__main__":
    lambda_grids()
    costs()
    bench()
