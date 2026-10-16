"""Analysis of study 2 (PREREGISTRATION-2.md, sections 3 to 7).

Tables are counted on the 120 episodes of study 1 (reference readings of
readings-main.jsonl), with the empirical k estimated there; every episode of
study 2 is evaluated with them. The hybrid is the value-of-information rule
(lambda = 12, priced stop) on the mixed table with empirical k, averaged over
the 10 LLM medians; only the reader changes from arm to arm.

usage: python analyze2.py --tag study2
       python analyze2.py --tag pilot --tables main     # trial on the pilot
"""
import argparse
import json
from pathlib import Path

import numpy as np

import analyze as A

OUT = Path(__file__).resolve().parent / "results"
READERS = ["routed", "jev", "haiku", "reference", "sonnet_narrow", "haiku_context", "haiku_seq", "haiku_seqn"]
RNG = np.random.default_rng(2027)


def load2(tag: str):
    episodes = [json.loads(l) for l in (OUT / f"episodes-{tag}.jsonl").open()]
    readings = {}
    # the preregistered readings, then the exploratory readers' file beside it (new fields only)
    for p in (OUT / f"readings2-{tag}.jsonl", OUT / f"readings2x-{tag}.jsonl"):
        if p.exists():
            for l in p.open():
                r = json.loads(l)
                readings.setdefault((r["episode"], r["check"]), {}).update(r)
    sonnet = {}
    p = OUT / f"sonnet-{tag}.jsonl"
    if p.exists():
        for l in p.open():
            r = json.loads(l)
            sonnet[r["episode"]] = r
    return episodes, readings, sonnet


def boot(diff, level):
    diff = np.asarray(diff, float)
    idx = RNG.integers(0, len(diff), size=(10_000, len(diff)))
    means = diff[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(diff.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def has(readings, ep, reader):
    return all(reader in readings.get((ep["episode"], c), {}) for c in A.CHECKS)


def replay_post(ep, table, readings, reader):
    """The hybrid on one episode, with its final belief: (loss, right, checks run, posterior)."""
    post = np.full(A.NC, 1 / A.NC)
    left, minutes, run = list(range(A.NT)), 0.0, []
    while (j := A.choose("voi", post, left, table, {})) is not None:
        left.remove(j)
        run.append(j)
        minutes += A.COST[j]
        post = A.update(post, table[j], A.reading(readings, ep, j, reader))
    right = A.CAUSES[int(post.argmax())] == ep["cause"]
    return minutes + (0.0 if right else A.MISS), right, run, post


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tables", default="main", help="the study whose episodes count the tables")
    a = ap.parse_args()

    # tables from study 1
    hist, hist_readings, _ = A.load(a.tables)
    hist = [e for e in hist if not A.excluded(e)]
    pos, run = A.counts(hist, hist_readings)
    meds = A.llm_medians()
    ks = [A.ebayes_k(pos, run, m) for m in meds]
    mixed = [A.table_mixed(pos, run, m, k) for m, k in zip(meds, ks)]
    mean_table = np.mean(mixed, axis=0)
    eps_table = A.table_eps(pos, run)
    print(f"== tables counted on {a.tables}: {len(hist)} episodes; empirical k median {np.median(ks):.1f} "
          f"(min {min(ks):.1f}, max {max(ks):.1f})")

    episodes, readings, sonnet = load2(a.tag)
    keep, dropped = [], {}
    for e in episodes:
        if A.excluded(e):
            dropped[e["cause"]] = dropped.get(e["cause"], 0) + 1
        else:
            keep.append(e)
    print(f"== {a.tag}: {len(episodes)} episodes, {len(keep)} kept, excluded per cause: {dropped or 'none'}")

    def losses(reader, eps=None):
        eps = keep if eps is None else eps
        return np.array([np.mean([A.replay("voi", e, t, readings, reader)[0] for t in mixed]) for e in eps])

    arms = [r for r in READERS if all(has(readings, e, r) for e in keep)]
    missing = [r for r in READERS if r not in arms]
    if missing:
        print(f"(readers not complete, left out: {missing})")
    L = {r: losses(r) for r in arms}
    s_loss = np.array([sonnet[e["episode"]]["loss"] if e["episode"] in sonnet and not sonnet[e["episode"]].get("failed")
                       else np.nan for e in keep])

    level = 1 - 0.05 / 2
    print(f"\nPrimary endpoints, paired bootstrap, {level:.1%} intervals (Bonferroni for two)")
    if "routed" in L and "jev" in L:
        m, lo, hi = boot(L["routed"] - L["jev"], level)
        v = "confirmed" if hi < 0 else ("contradicted" if lo > 0 else "not confirmed")
        print(f"H4 routed {L['routed'].mean():6.2f} vs Jev {L['jev'].mean():6.2f}   diff {m:+6.2f} [{lo:+.2f}, {hi:+.2f}]  {v}")
    ok = ~np.isnan(s_loss)
    if "routed" in L and ok.any():
        m, lo, hi = boot(L["routed"][ok] - s_loss[ok], level)
        v = ("confirmed, and better" if hi < 0 else "confirmed (not worse by 2 min)") if hi < 2 else "not confirmed"
        print(f"H5 routed {L['routed'][ok].mean():6.2f} vs Sonnet {s_loss[ok].mean():6.2f}  diff {m:+6.2f} "
              f"[{lo:+.2f}, {hi:+.2f}]  margin +2  {v}  (n={ok.sum()})")

    print("\nSecondary (exploratory): the hybrid per reader, on the mean table (calibration) and averaged tables (loss)")
    print(f"  {'reader':14} {'loss':>6} {'right':>6} {'checks':>6} {'Brier':>6} {'logS':>6} {'readBrier':>9} {'$/diag':>7}")
    for r in arms:
        rows = [replay_post(e, mean_table, readings, r) for e in keep]
        right = np.mean([x[1] for x in rows])
        checks = np.mean([len(x[2]) for x in rows])
        p_true = np.array([x[3][A.CAUSES.index(e["cause"])] for x, e in zip(rows, keep)])
        onehot = [np.eye(A.NC)[A.CAUSES.index(e["cause"])] for e in keep]
        brier = np.mean([((x[3] - y) ** 2).sum() for x, y in zip(rows, onehot)])
        logs = float(-np.log(np.clip(p_true, 1e-9, 1)).mean())
        rb = [(A.reading(readings, e, j, r) - readings[(e["episode"], A.CHECKS[j])]["reference"]) ** 2
              for x, e in zip(rows, keep) for j in x[2]]
        usd = np.mean([sum(readings[(e["episode"], A.CHECKS[j])].get(f"{r}_usd", 0.0) for j in x[2])
                       for x, e in zip(rows, keep)])
        print(f"  {r:14} {L[r].mean():6.2f} {right:6.1%} {checks:6.2f} {brier:6.3f} {logs:6.3f} {np.mean(rb):9.3f} {usd:7.4f}")
    if ok.any():
        acc = np.mean([sonnet[e["episode"]]["right"] for e in keep if e["episode"] in sonnet and not sonnet[e["episode"]].get("failed")])
        usd = np.mean([sonnet[e["episode"]].get("dollars", 0) for e in keep if e["episode"] in sonnet])
        print(f"  {'Sonnet alone':14} {np.nanmean(s_loss):6.2f} {acc:6.1%} "
              f"{np.mean([len(sonnet[e['episode']]['path']) for e in keep if e['episode'] in sonnet]):6.2f}"
              f" {'':>6} {'':>6} {'':>9} {usd:7.4f}")

        # supported: Sonnet's own path, replayed with the reference reader and the study 1 table
        sup = []
        for e in keep:
            s = sonnet.get(e["episode"])
            if not s or s.get("failed") or not s.get("answer"):
                continue
            post = np.full(A.NC, 1 / A.NC)
            for name in s["path"]:
                j = A.CHECKS.index(name)
                post = A.update(post, mean_table[j], A.reading(readings, e, j, "reference"))
            sup.append((post[A.CAUSES.index(s["answer"])] >= 0.8, s["right"]))
        if sup:
            sup = np.array(sup)
            print(f"  Sonnet alone: conclusion supported by its own path (posterior >= 0.8) {sup[:, 0].mean():.1%};"
                  f" right but unsupported {np.mean(sup[:, 1] & ~sup[:, 0]):.1%}")

    print("\nPull toward the suspected cause (shift x (2 P(signal | named cause) - 1)), within reference strata")
    for informed, blind, cause in [("haiku_context", "haiku", "haiku_context_cause"),
                                   ("haiku_seq", "haiku_seq_blind", "haiku_seq_cause"),
                                   ("haiku_seqn", "haiku_seqn_blind", "haiku_seqn_cause")]:
        if informed not in arms:
            continue
        for ref in (0, 1):
            vals = []
            for e in keep:
                for j, c in enumerate(A.CHECKS):
                    row = readings[(e["episode"], c)]
                    if row["reference"] != ref or blind not in row:
                        continue
                    shift = row[informed] - row[blind]
                    named = A.CAUSES.index(row[cause])
                    vals.append(shift * (2 * mean_table[j, named] - 1))
            if vals:
                m, lo, hi = boot(vals, 0.95)
                print(f"  {informed:14} reference {'present' if ref else 'absent '}  {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  (n={len(vals)})")

    print("\nReadings on the wrong side of the reference, per check")
    print("  " + f"{'check':22}" + "".join(f"{r[:13]:>14}" for r in arms if r != "reference"))
    for c in A.CHECKS:
        cells = []
        for r in arms:
            if r == "reference":
                continue
            rs = [(readings[(e["episode"], c)][r], readings[(e["episode"], c)]["reference"]) for e in keep]
            p, y = np.array(rs, float).T
            cells.append(f"{((p > 0.5) != (y > 0.5)).mean():14.1%}")
        print(f"  {c:22}" + "".join(cells))
    if "routed" in arms:
        derived = ["restart_consumer", "partition_lag_spread", "producer_rate_graph", "consumer_cpu_profile"]
        fails = [readings[(e["episode"], c)].get("routed_failed", 0) for e in keep for c in derived]
        print(f"  routed: failed extractions {sum(fails)} of {3 * len(fails)}")

    print("\nStudy 1 repeated on the new series (tables counted on study 1, Jev reader)")
    if "jev" in arms:
        h1a, h1b = L["jev"], np.array([np.mean([A.replay("voi", e, t, readings, "jev")[0] for t in meds]) for e in keep])
        m, lo, hi = boot(h1a - h1b, 0.95)
        print(f"  H1 mixed empirical k {h1a.mean():6.2f} vs LLM {h1b.mean():6.2f}  diff {m:+6.2f} [{lo:+.2f}, {hi:+.2f}]")
        h2a = np.array([A.replay("ratio_priced", e, eps_table, readings, "jev")[0] for e in keep])
        h2b = np.array([A.replay("hesp", e, eps_table, readings, "jev")[0] for e in keep])
        m, lo, hi = boot(h2a - h2b, 0.95)
        print(f"  H2 priced stop {h2a.mean():6.2f} vs HESP {h2b.mean():6.2f}  diff {m:+6.2f} [{lo:+.2f}, {hi:+.2f}]")

    shifted = OUT / f"episodes-{a.tag}-shifted.jsonl"
    if shifted.exists() and (OUT / f"readings2-{a.tag}-shifted.jsonl").exists():
        eps_s, rd_s, _ = load2(f"{a.tag}-shifted")
        eps_s = [e for e in eps_s if not A.excluded(e)]
        print("\nFormat shift (exploratory): the hybrid on the shifted copy")
        for r in ("reference", "routed"):
            if all(has(rd_s, e, r) for e in eps_s):
                ls = np.array([np.mean([A.replay("voi", e, t, rd_s, r)[0] for t in mixed]) for e in eps_s])
                print(f"  {r:10} {ls.mean():6.2f}  (unshifted {L[r].mean():6.2f})" if r in L else f"  {r:10} {ls.mean():6.2f}")


if __name__ == "__main__":
    main()
