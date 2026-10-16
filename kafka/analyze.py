"""Analysis of the real-fault series (PREREGISTRATION.md, sections 5 to 7 and addenda).

Every rule is replayed offline on the recorded episodes: it picks a check,
receives that check's reading, updates its belief, and continues or
concludes. Loss = minutes of checks run + 60 if the concluded cause is wrong.

Tables are counted with two-fold cross-fitting: fold A = blocks 1 to 10
(episodes 0 to 59), fold B = blocks 11 to 20; an episode is always evaluated
with tables counted on the other fold. Counted tables use the deterministic
reference's readings of the counting fold.

usage: python analyze.py --tag main
       python analyze.py --tag pilot      # smoke test: one fold only, tables fall back to their priors
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import betaln

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "exp2"))
from scenarios import SCENARIOS  # noqa: E402

OUT = HERE / "results"
SC = SCENARIOS["kafka"]
CAUSES = SC["causes"]
CHECKS = [t[0] for t in SC["tests"]]
COST = np.array([t[1] for t in SC["tests"]])
DESIGNER = np.array([t[2] for t in SC["tests"]])  # checks x causes
MISS = SC["miss"]
LAM = 12.0
NC, NT = len(CAUSES), len(CHECKS)
TEXTUAL = ["rebalance_logs", "jvm_gc_logs", "consumer_cpu_profile"]
NUMERIC = ["partition_lag_spread", "producer_rate_graph", "broker_quota_metrics", "restart_consumer"]
K_GRID = np.geomspace(1, 300, 40)
RNG = np.random.default_rng(2026)


# --------------------------------------------------------------------------- #
#  data
# --------------------------------------------------------------------------- #
def load(tag: str):
    episodes = [json.loads(l) for l in (OUT / f"episodes-{tag}.jsonl").open()]
    readings = {}
    for l in (OUT / f"readings-{tag}.jsonl").open():
        r = json.loads(l)
        readings.setdefault((r["episode"], r["check"]), {}).update(r)  # later lines complete earlier ones
    sonnet = {}
    p = OUT / f"sonnet-{tag}.jsonl"
    if p.exists():
        for l in p.open():
            r = json.loads(l)
            sonnet[r["episode"]] = r
    return episodes, readings, sonnet


def excluded(ep: dict) -> str | None:
    """The two exclusion rules of section 3 (a failed reset stops the runner, so only the second can show)."""
    lags = ep.get("lag_after_injection") or []
    if not lags or lags[-1] < 2 * max(ep["healthy_lag"], 1):
        return "fault did not take"
    return None


def fold_of(ep: dict, n_blocks: int) -> int:
    return 0 if ep["episode"] // 6 < n_blocks / 2 else 1


def reading(readings, ep: dict, j: int, reader: str) -> float:
    v = readings[(ep["episode"], CHECKS[j])][reader]
    return float(v) if reader == "reference" else min(max(float(v), 0.01), 0.99)


# --------------------------------------------------------------------------- #
#  tables
# --------------------------------------------------------------------------- #
def counts(history: list[dict], readings, checks_run: dict | None = None):
    """Positives and runs per (check, cause) from the reference readings of `history`."""
    pos = np.zeros((NT, NC))
    run = np.zeros((NT, NC))
    for ep in history:
        c = CAUSES.index(ep["cause"])
        for j in (checks_run[ep["episode"]] if checks_run else range(NT)):
            run[j, c] += 1
            pos[j, c] += readings[(ep["episode"], CHECKS[j])]["reference"]
    return pos, run


def table_mixed(pos, run, q, k):
    return np.clip((pos + k * q) / (run + k), 0.01, 0.99)


def table_eps(pos, run, eps=0.01):
    """HESP's counted table: counts smoothed with eps (section 4.1 of the paper); 0.5 where nothing was seen."""
    return (pos + eps) / (run + 2 * eps)


def ebayes_k(pos, run, q) -> float:
    neg = run - pos
    qq = np.clip(q, 0.01, 0.99)

    def loglik(k):
        a, b = k * qq, k * (1 - qq)
        return float((betaln(pos + a, neg + b) - betaln(a, b)).sum())

    return float(max(K_GRID, key=loglik))


def llm_medians() -> list[np.ndarray]:
    from tables import medians_of
    return [np.clip(np.asarray(m), 0.01, 0.99) for m in medians_of("kafka")]


# --------------------------------------------------------------------------- #
#  belief and rules
# --------------------------------------------------------------------------- #
def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def info_gain(post, like):
    p_pos = float(post @ like)
    h = 0.0
    for pr, lk in ((p_pos, like), (1 - p_pos, 1 - like)):
        if pr > 0:
            q = post * lk
            h += pr * entropy(q / q.sum())
    return entropy(post) - h


def p_wrong(post):
    return 1.0 - float(post.max())


def step_loss(post, like):
    p_pos = float(post @ like)
    out = 0.0
    for pr, lk in ((p_pos, like), (1 - p_pos, 1 - like)):
        if pr > 0:
            q = post * lk
            out += pr * MISS * p_wrong(q / q.sum())
    return out


def update(post, like, r):
    q = post * (r * like + (1 - r) * (1 - like))
    return q / q.sum()


def choose(rule, post, left, table, state):
    if not left:
        return None
    if rule == "voi":
        s = {j: COST[j] - LAM * info_gain(post, table[j]) for j in left}
        j = min(s, key=s.get)
        return j if s[j] < 0 else None
    if rule == "ratio_priced":
        g = {j: info_gain(post, table[j]) for j in left}
        j = max(left, key=lambda j: g[j] / COST[j])
        return j if g[j] > 0 and COST[j] / g[j] < LAM else None
    if rule == "hesp":
        lead = int(post.argmax())
        if post[lead] >= 0.8 and lead in state["supported"]:
            return None
        return max(left, key=lambda j: info_gain(post, table[j]) / COST[j])
    if rule == "bayes":
        s = {j: COST[j] + step_loss(post, table[j]) for j in left}
        j = min(s, key=s.get)
        return j if s[j] < MISS * p_wrong(post) else None
    if rule in ("cheapest", "most_informative"):
        if post.max() >= 0.9:
            return None
        if rule == "cheapest":
            return min(left, key=lambda j: COST[j])
        return max(left, key=lambda j: info_gain(post, table[j]))
    raise ValueError(rule)


def replay(rule, ep, table, readings, reader, prior=None):
    """One episode under one rule: (loss, minutes, right, checks run)."""
    hesp = rule == "hesp"
    if hesp:  # six causes plus an explicit `other`, uniform prior, `other` row unseen = 0.5
        table = np.hstack([table, np.full((NT, 1), 0.5)])
        post = np.full(NC + 1, 1 / (NC + 1))
    else:
        post = (prior if prior is not None else np.full(NC, 1 / NC)).copy()
    left, minutes, run, state = list(range(NT)), 0.0, [], {"supported": set()}
    while (j := choose(rule, post, left, table, state)) is not None:
        left.remove(j)
        run.append(j)
        minutes += COST[j]
        before = post.copy()
        post = update(post, table[j], reading(readings, ep, j, reader))
        state["supported"] |= {h for h in range(len(post)) if post[h] > before[h]}
    answer = int(post.argmax())
    right = answer < NC and CAUSES[answer] == ep["cause"]
    return minutes + (0.0 if right else MISS), minutes, right, run


# --------------------------------------------------------------------------- #
#  statistics
# --------------------------------------------------------------------------- #
def boot(diff, level):
    diff = np.asarray(diff, float)
    idx = RNG.integers(0, len(diff), size=(10_000, len(diff)))
    means = diff[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(diff.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def verdict(lo, hi):
    return "confirmed" if hi < 0 else ("contradicted" if lo > 0 else "inconclusive")


# --------------------------------------------------------------------------- #
#  the run
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    episodes, readings, sonnet = load(a.tag)
    n_blocks = max(e["episode"] for e in episodes) // 6 + 1
    keep, dropped = [], {}
    for e in episodes:
        why = excluded(e)
        if why:
            dropped[e["cause"]] = dropped.get(e["cause"], 0) + 1
        else:
            keep.append(e)
    print(f"== {a.tag}: {len(episodes)} episodes, {len(keep)} kept, excluded per cause: {dropped or 'none'}")
    meds = llm_medians()
    folds = [[e for e in keep if fold_of(e, n_blocks) == f] for f in (0, 1)]

    # tables for each fold, counted on the other one
    T = {}
    for f in (0, 1):
        hist = folds[1 - f]
        pos, run = counts(hist, readings)
        qbar = np.median(np.array(meds), axis=0)
        ks = [ebayes_k(pos, run, m) for m in meds]
        T[f] = {"designer": DESIGNER, "counted": np.clip((pos + 1) / (run + 2), 0.01, 0.99),
                "eps": table_eps(pos, run), "llm": meds,
                "mixed10": [table_mixed(pos, run, m, 10) for m in meds],
                "mixed_eb": [table_mixed(pos, run, m, k) for m, k in zip(meds, ks)],
                "k": ks, "history": len(hist), "qbar": qbar}
        print(f"fold {'AB'[f]}: {len(folds[f])} evaluated, tables counted on {len(hist)}; empirical k: "
              f"median {np.median(ks):.1f} (min {min(ks):.1f}, max {max(ks):.1f})")

    def loss_of(rule, table_key, reader, avg=True):
        """Per-episode losses over the kept episodes (list tables averaged over the 10 medians)."""
        out = []
        for e in keep:
            t = T[fold_of(e, n_blocks)][table_key]
            if isinstance(t, list):
                ls = [replay(rule, e, ti, readings, reader)[0] for ti in t]
                out.append(np.mean(ls) if avg else ls)
            else:
                out.append(replay(rule, e, t, readings, reader)[0])
        return np.array(out)

    level = 1 - 0.05 / 3
    print(f"\nPrimary endpoints, paired bootstrap, {level:.2%} intervals (Bonferroni for three)")
    h1a, h1b = loss_of("voi", "mixed_eb", "jev"), loss_of("voi", "llm", "jev")
    m, lo, hi = boot(h1a - h1b, level)
    print(f"H1 table  mixed empirical k {h1a.mean():6.2f}  vs LLM {h1b.mean():6.2f}   diff {m:+6.2f} [{lo:+.2f}, {hi:+.2f}]  {verdict(lo, hi)}")
    h2a, h2b = loss_of("ratio_priced", "eps", "jev"), loss_of("hesp", "eps", "jev")
    m, lo, hi = boot(h2a - h2b, level)
    print(f"H2 stop   priced            {h2a.mean():6.2f}  vs HESP {h2b.mean():6.2f}  diff {m:+6.2f} [{lo:+.2f}, {hi:+.2f}]  {verdict(lo, hi)}")
    s_loss = np.array([sonnet[e["episode"]]["loss"] if e["episode"] in sonnet and not sonnet[e["episode"]].get("failed")
                       else np.nan for e in keep])
    ok = ~np.isnan(s_loss)
    if ok.any():
        m, lo, hi = boot(h1a[ok] - s_loss[ok], level)
        print(f"H3 hybrid                   {h1a[ok].mean():6.2f}  vs Sonnet {s_loss[ok].mean():6.2f} "
              f"diff {m:+6.2f} [{lo:+.2f}, {hi:+.2f}]  {verdict(lo, hi)}  (n={ok.sum()})")
    else:
        print("H3: no Sonnet runs yet")

    print("\nSecondary (exploratory): mean loss by rule x table, Jev reader")
    for rule in ["voi", "ratio_priced", "hesp", "bayes", "cheapest", "most_informative"]:
        row = []
        for tk in ["designer", "llm", "counted", "eps", "mixed10", "mixed_eb"]:
            row.append(f"{tk} {loss_of(rule, tk, 'jev').mean():6.2f}")
        print(f"  {rule:16} " + " | ".join(row))
    print("  readers, voi + mixed_eb: " + " | ".join(
        f"{r} {loss_of('voi', 'mixed_eb', r).mean():6.2f}" for r in ["reference", "jev", "haiku"]))

    print("\nHESP's own metric (share correct), Jev reader")
    for rule, tk in [("voi", "mixed_eb"), ("ratio_priced", "eps"), ("hesp", "eps"), ("bayes", "mixed_eb")]:
        acc = []
        for e in keep:
            t = T[fold_of(e, n_blocks)][tk]
            ts = t if isinstance(t, list) else [t]
            acc.append(np.mean([replay(rule, e, ti, readings, "jev")[2] for ti in ts]))
        print(f"  {rule:14} {tk:9} {np.mean(acc):.3f}")
    if ok.any():
        acc = np.mean([sonnet[e["episode"]]["right"] for e in keep if e["episode"] in sonnet and not sonnet[e["episode"]].get("failed")])
        cost = sum(sonnet[e["episode"]].get("dollars", 0) for e in keep if e["episode"] in sonnet)
        print(f"  Sonnet alone   {acc:.3f}   ({cost:.2f} $)")

    print("\nReading agreement with the reference (Brier, share on the wrong side)")
    for reader in ["jev", "haiku"]:
        groups = {"textual": [], "numeric": []}
        for c in CHECKS:
            rs = [(readings[(e["episode"], c)][reader], readings[(e["episode"], c)]["reference"]) for e in keep
                  if reader in readings[(e["episode"], c)]]
            if not rs:
                continue
            p, y = np.array(rs, float).T
            brier, wrong = float(((p - y) ** 2).mean()), float(((p > 0.5) != (y > 0.5)).mean())
            groups["textual" if c in TEXTUAL else "numeric"].append(brier)
            print(f"  {reader:6} {c:22} Brier {brier:.3f}  wrong side {wrong:.1%}")
        print(f"  {reader:6} textual {np.mean(groups['textual']):.3f} vs numeric {np.mean(groups['numeric']):.3f}"
              f"  (prediction: textual lower)")

    print("\nInformative censoring (addendum 2): tables from a self-chosen history, voi + Jev on the other fold")
    for label in ["llm k=10", "no prior"]:
        losses_self, losses_full = [], []
        for f in (0, 1):
            hist, evals = folds[1 - f], folds[f]
            priors = meds if label.startswith("llm") else [np.full((NT, NC), 0.5)]
            k = 10.0 if label.startswith("llm") else 2.0
            for q in priors:
                chosen = {}
                for e in sorted(hist, key=lambda e: e["episode"]):
                    pos, run = counts([h for h in hist if h["episode"] in chosen], readings, chosen)
                    tab = table_mixed(pos, run, q, k)
                    _, _, _, ran = replay("voi", e, tab, readings, "reference")
                    chosen[e["episode"]] = ran
                pos_s, run_s = counts(hist, readings, chosen)
                pos_f, run_f = counts(hist, readings)
                ts, tf = table_mixed(pos_s, run_s, q, k), table_mixed(pos_f, run_f, q, k)
                losses_self += [replay("voi", e, ts, readings, "jev")[0] for e in evals]
                losses_full += [replay("voi", e, tf, readings, "jev")[0] for e in evals]
                checks = np.mean([len(v) for v in chosen.values()]) if chosen else 0
        print(f"  {label:9} self-chosen {np.mean(losses_self):6.2f} vs complete {np.mean(losses_full):6.2f}"
              f"  (checks per history episode, last prior: {checks:.2f})")


if __name__ == "__main__":
    main()
