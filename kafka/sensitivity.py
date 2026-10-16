"""Sensitivity of the five verdicts to the price of an error M and the price of a bit lambda (exploratory).

Both studies are replayed offline on their recorded outputs, unchanged, with M and lambda varied; nothing
is re-run. Two regimes: lambda = M/5 (the preregistered ratio, 12 at M = 60) and lambda fixed at 12 (a
rule tuned once and left alone while the cost of an error moves). Sonnet alone keeps the decisions it took
under its brief (which stated M = 60); only its loss is recomputed, minutes + M x wrong. H5 keeps its
preregistered 2-minute margin at every M. Intervals are 95 %
paired bootstrap, uncorrected: this is not a preregistered test.

usage: python sensitivity.py > results/sensitivity.txt
"""
import numpy as np

import analyze as A
import analyze2 as B

MS = [15, 30, 60, 120, 240]
RNG = np.random.default_rng(7)


def boot(diff):
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    idx = RNG.integers(0, len(diff), size=(10_000, len(diff)))
    means = diff[idx].mean(axis=1)
    return diff.mean(), np.quantile(means, 0.025), np.quantile(means, 0.975)


def fmt(name, a, b, margin=None):
    m, lo, hi = boot(a - b)
    if margin is None:
        tag = "holds" if hi < 0 else ("reversed" if lo > 0 else "not separated")
    else:
        tag = "holds" if hi < margin else "fails"
    return f"{name} {np.nanmean(a):6.2f} vs {np.nanmean(b):6.2f}  diff {m:+6.2f} [{lo:+.2f}, {hi:+.2f}] {tag}"


def study1():
    episodes, readings, _ = A.load("main")
    n_blocks = max(e["episode"] for e in episodes) // 6 + 1
    keep = [e for e in episodes if not A.excluded(e)]
    meds = A.llm_medians()
    T = {}
    for f in (0, 1):
        fold = [e for e in keep if A.fold_of(e, n_blocks) != f]
        pos, run = A.counts(fold, readings)
        ks = [A.ebayes_k(pos, run, m) for m in meds]
        T[f] = {"llm": meds, "eps": [A.table_eps(pos, run)],
                "mixed_eb": [A.table_mixed(pos, run, m, k) for m, k in zip(meds, ks)]}

    def loss(rule, tk):
        return np.array([np.mean([A.replay(rule, e, t, readings, "jev")[0] for t in T[A.fold_of(e, n_blocks)][tk]])
                         for e in keep])
    return loss


def study2():
    hist, hist_readings, _ = A.load("main")
    hist = [e for e in hist if not A.excluded(e)]
    pos, run = A.counts(hist, hist_readings)
    meds = A.llm_medians()
    mixed = [A.table_mixed(pos, run, m, A.ebayes_k(pos, run, m)) for m in meds]
    episodes, readings, sonnet = B.load2("study2")
    keep = [e for e in episodes if not A.excluded(e)]

    def loss(reader):
        return np.array([np.mean([A.replay("voi", e, t, readings, reader)[0] for t in mixed]) for e in keep])

    def sonnet_loss():
        out = []
        for e in keep:
            s = sonnet.get(e["episode"])
            out.append(np.nan if s is None or s.get("failed") else s["minutes"] + (0 if s["right"] else A.MISS))
        return np.array(out)
    return loss, sonnet_loss


def main() -> None:
    s1 = study1()
    s2, sonnet = study2()
    for regime in ("lambda = M/5", "lambda = 12"):
        print(f"\n== {regime}")
        for M in MS:
            A.MISS = float(M)
            A.LAM = M / 5 if regime == "lambda = M/5" else 12.0
            print(f"-- M = {M}, lambda = {A.LAM:g}")
            print("  " + fmt("H1 mixed vs LLM table   ", s1("voi", "mixed_eb"), s1("voi", "llm")))
            print("  " + fmt("H2 priced vs HESP stop  ", s1("ratio_priced", "eps"), s1("hesp", "eps")))
            routed = s2("routed")
            print("  " + fmt("H4 routed vs Jev        ", routed, s2("jev")))
            print("  " + fmt("H5 routed vs Sonnet (margin 2)", routed, sonnet(), margin=2.0))


if __name__ == "__main__":
    main()
