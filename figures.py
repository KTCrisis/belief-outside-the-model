"""Figures of the paper and of the companion note, as SVG.

Every figure is recomputed from the code and the recorded results, never typed
by hand: the optimal value V from exact dynamic programming (value_fit.py), the
M sweep from results-m-sweep.txt, study 1 from the recorded episodes, readings
and Sonnet runs (kafka/analyze.py, replayed read-only).

usage: python figures.py [--out figures] [name ...]   |   python figures.py --paper
names: value, msweep, arms, readers, hist, study2 (default: all)
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent

# Reference palette (dataviz skill, light mode); slots in fixed order.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "svg.fonttype": "none",
})


def save(fig, out: Path, name: str) -> None:
    path = out / f"bayes-{name}.svg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
#  volet I: the optimal value V against the two guesses of the one-step rules
# --------------------------------------------------------------------------- #
def dump_values() -> None:
    """Child process: SCENARIO is read by sim.py at import, so one process per case."""
    sys.path.insert(0, str(ROOT))
    from sim import CASE  # noqa: PLC0415
    from value_fit import all_values  # noqa: PLC0415
    v0, rows = all_values(CASE["miss"])
    np.save(sys.argv[sys.argv.index("--dump-values") + 1], rows)


def fig_value(out: Path) -> None:
    cases = [("clinic", "Clinique (euros)", "€"), ("kafka", "Kafka (minutes)", "min")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey="row")
    for r, (sc, title, unit) in enumerate(cases):
        tmp = out / f".v-{sc}.npy"
        subprocess.run([sys.executable, __file__, "--dump-values", str(tmp)], check=True,
                       env={**os.environ, "SCENARIO": sc}, cwd=ROOT)
        V, H, R, n = np.load(tmp).T
        tmp.unlink()
        for c, (x, xl, col) in enumerate([(H, "entropie de la croyance H (bits)", S1),
                                         (R, f"M × risque d'erreur ({unit})", S2)]):
            ax = axes[r, c]
            a = float(x @ V / (x @ x))
            res = V - a * x
            r2 = 1 - res @ res / ((V - V.mean()) @ (V - V.mean()))
            ax.scatter(x, V, s=6, color=col, alpha=0.25, linewidths=0)
            xs = np.linspace(0, x.max(), 50)
            ax.plot(xs, a * xs, color=INK, linewidth=1.5)
            slope = f"pente {a:.1f} {unit}/bit" if c == 0 else f"pente {a:.2f}"
            ax.text(0.03, 0.95, f"{slope}\nR² = {r2:.2f}", transform=ax.transAxes, va="top", color=INK2, fontsize=8)
            ax.set_xlabel(xl)
            if c == 0:
                ax.set_ylabel(f"V optimale ({unit})")
            ax.set_title(f"{title}, {len(V)} états" if c == 0 else "")
    fig.suptitle("La valeur optimale V contre les deux approximations des règles à un coup",
                 x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out, "value")


# --------------------------------------------------------------------------- #
#  volet I: how far lambda = M/4 strays from the optimum when M moves
# --------------------------------------------------------------------------- #
def fig_msweep(out: Path) -> None:
    txt = (ROOT / "results-m-sweep.txt").read_text()
    blocks = re.split(r"scenario (\w+), base M = ([\d.]+)", txt)[1:]
    names = {"kafka": "Kafka", "internet": "Box internet", "clinic": "Clinique"}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True)
    for ax, (sc, base, body) in zip(axes, zip(blocks[0::3], blocks[1::3], blocks[2::3])):
        rows = [list(map(float, ln.replace("|", " ").split())) for ln in body.splitlines()
                if re.match(r"\s+\d", ln)]
        rows = np.array(rows)
        mult = rows[:, 0] / float(base)
        opt = rows[:, 1]
        gap = lambda col: 100 * (rows[:, col] - opt) / opt  # noqa: E731
        series = [("moindre action, λ = M/4", gap(6), S1, "o"),
                  ("prix du bit (ratio_priced), λ = M/4", gap(10), S2, "s"),
                  ("moindre action, meilleur λ", gap(5), S3, "^")]
        for label, y, col, mk in series:
            ax.plot(mult, y, color=col, linewidth=2, marker=mk, markersize=5, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 0.5, 1, 2, 4], ["×¼", "×½", "×1", "×2", "×4"])
        ax.set_title(names[sc])
        ax.set_xlabel("prix de l'erreur M")
    axes[0].set_ylabel("écart à l'optimum (%)")
    axes[0].legend(loc="upper left", fontsize=7.5, handlelength=1.6)
    fig.suptitle("λ = M/4 tient près du M de référence et s'éloigne quand l'erreur devient chère",
                 x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out, "msweep")


# --------------------------------------------------------------------------- #
#  volet II: study 1 (real faults), replayed read-only from kafka/analyze.py
# --------------------------------------------------------------------------- #
def study1():
    sys.path.insert(0, str(ROOT / "kafka"))
    old = sys.argv
    sys.argv = ["analyze.py"]
    import analyze as A  # noqa: PLC0415
    sys.argv = old
    episodes, readings, sonnet = A.load("main")
    n_blocks = max(e["episode"] for e in episodes) // 6 + 1
    keep = [e for e in episodes if not A.excluded(e)]
    meds = A.llm_medians()
    folds = [[e for e in keep if A.fold_of(e, n_blocks) == f] for f in (0, 1)]
    T = {}
    for f in (0, 1):
        pos, run = A.counts(folds[1 - f], readings)
        ks = [A.ebayes_k(pos, run, m) for m in meds]
        T[f] = {"llm": meds, "eps": A.table_eps(pos, run),
                "mixed_eb": [A.table_mixed(pos, run, m, k) for m, k in zip(meds, ks)]}

    def loss(rule, tk, reader):
        out = []
        for e in keep:
            t = T[A.fold_of(e, n_blocks)][tk]
            ts = t if isinstance(t, list) else [t]
            out.append(np.mean([A.replay(rule, e, ti, readings, reader)[0] for ti in ts]))
        return np.array(out)

    s = np.array([sonnet[e["episode"]]["loss"] for e in keep])
    return keep, loss, s


def fig_arms(out: Path, keep, loss, s) -> None:
    arms = [("Sonnet seul", s.mean()),
            ("hybride, lecteur de référence", loss("voi", "mixed_eb", "reference").mean()),
            ("hybride, lecteur Haiku", loss("voi", "mixed_eb", "haiku").mean()),
            ("hybride, lecteur Jev", loss("voi", "mixed_eb", "jev").mean()),
            ("arrêt façon HESP (Jev)", loss("hesp", "eps", "jev").mean()),
            ("table du modèle seule (Jev)", loss("voi", "llm", "jev").mean())]
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    y = np.arange(len(arms))[::-1]
    vals = [v for _, v in arms]
    ax.barh(y, vals, height=0.62, color=S1)
    for yi, v in zip(y, vals):
        ax.text(v + 0.4, yi, f"{v:.1f}", va="center", color=INK, fontsize=8.5)
    ax.set_yticks(y, [a for a, _ in arms])
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("perte moyenne par panne (minutes + 60 si mauvaise cause)")
    ax.set_xlim(0, max(vals) * 1.12)
    ax.set_title(f"Étude 1, {len(keep)} pannes réelles : le goulot est la lecture, pas la règle")
    fig.tight_layout()
    save(fig, out, "study1-arms")


def fig_readers(out: Path) -> None:
    txt = (ROOT / "kafka/results/analysis-main.txt").read_text()
    rows = re.findall(r"^\s+(jev|haiku)\s+(\w+)\s+Brier [\d.]+\s+wrong side ([\d.]+)%", txt, re.M)
    labels = {"rebalance_logs": "journaux de rebalancing", "jvm_gc_logs": "journaux GC",
              "broker_quota_metrics": "quotas du broker", "restart_consumer": "redémarrage d'un consommateur",
              "partition_lag_spread": "lag par partition", "consumer_cpu_profile": "profil CPU",
              "producer_rate_graph": "débit du producteur"}
    order = list(labels)
    val = {(r, c): float(p) for r, c, p in rows}
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    y = np.arange(len(order))[::-1]
    h = 0.36
    for off, (r, name, col) in zip([h / 2 + 0.02, -h / 2 - 0.02], [("jev", "Jev", S1), ("haiku", "Haiku", S2)]):
        v = [val[(r, c)] for c in order]
        ax.barh(y + off, v, height=h, color=col, label=name)
        for yi, vi in zip(y + off, v):
            ax.text(vi + 0.6, yi, f"{vi:.1f} %" if vi else "0", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [labels[c] for c in order])
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("lectures du mauvais côté de 0,5 par rapport à la référence (%)")
    ax.legend(loc="upper right")
    ax.set_title("Texte lu sans faute, nombres à comparer : aucun lecteur n'est bon partout")
    fig.tight_layout()
    save(fig, out, "readers")


def fig_hist(out: Path, keep, loss, s) -> None:
    series = [("Sonnet seul", s, S1), ("hybride, lecteur de référence", loss("voi", "mixed_eb", "reference"), S3),
              ("hybride, lecteur Jev", loss("voi", "mixed_eb", "jev"), S2)]
    bins = np.arange(0, 85, 4)
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 4.6), sharex=True, sharey=True)
    for ax, (name, v, col) in zip(axes, series):
        ax.hist(v, bins=bins, color=col, rwidth=0.88)
        miss = int((v >= 60).sum())
        ax.set_title(f"{name} : moyenne {v.mean():.1f}, {miss} pannes ≥ 60",
                     fontsize=8.5, fontweight="normal", color=INK2)
        ax.axvline(60, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_ylabel("pannes")
    axes[-1].set_xlabel("perte par panne (minutes ; au-delà de 60, la cause annoncée était fausse)")
    fig.suptitle(f"Étude 1, distribution des pertes sur {len(keep)} pannes",
                 x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out, "study1-hist")


def fig_study2(out: Path) -> None:
    """Parsed from the frozen analysis of study 2 (analysis-study2.txt), nothing recomputed."""
    txt = (ROOT / "kafka/results/analysis-study2.txt").read_text()
    names = {"routed": "hybride, lecteur aiguillé", "reference": "hybride, lecteur de référence",
             "sonnet_narrow": "hybride, Sonnet lecteur étroit", "haiku_context": "hybride, Haiku avec contexte",
             "haiku": "hybride, Haiku", "jev": "hybride, Jev", "haiku_seqn": "hybride, Haiku séquentiel neutre",
             "haiku_seq": "hybride, Haiku séquentiel orienté"}
    rows = {m.group(1): (float(m.group(2)), float(m.group(3)))
            for m in re.finditer(r"^\s{2}(\w+)\s+([\d.]+)\s+[\d.]+%.*?([\d.]+)$", txt, re.M) if m.group(1) in names}
    so = re.search(r"Sonnet alone\s+([\d.]+)\s+[\d.]+%\s+[\d.]+\s+([\d.]+)", txt)
    arms = [("Sonnet seul", float(so.group(1)), float(so.group(2)))]
    arms += [(names[k], v[0], v[1]) for k, v in sorted(rows.items(), key=lambda kv: kv[1][0])]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    y = np.arange(len(arms))[::-1]
    for yi, (lab, loss, usd) in zip(y, arms):
        ax.barh(yi, loss, height=0.62, color=S2 if lab == "Sonnet seul" else S1)
        cost = (f"{1000 * usd:.1f} $ / 1000 diag." if usd else
                "Jev, facturé à part, non compté" if "Jev" in lab else "code, sans modèle")
        ax.text(loss + 0.3, yi, f"{loss:.2f}   ({cost})", va="center", fontsize=7.8, color=INK2)
    ax.set_yticks(y, [a[0] for a in arms])
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(a[1] for a in arms) * 1.45)
    ax.set_xlabel("perte moyenne par panne (minutes + 60 si mauvaise cause), 120 pannes neuves")
    ax.set_title("Étude 2 : le lecteur aiguillé rejoint Sonnet seul pour six fois moins cher")
    fig.tight_layout()
    save(fig, out, "study2-arms")


# --------------------------------------------------------------------------- #
#  the paper (studies 1 and 2), in English, one column wide (3.35 in)
# --------------------------------------------------------------------------- #
COL = 3.35


#: The paper's language; French labels for the French version (paper-fr-*.svg).
LANG = "en"
FR = {
    "Sonnet alone": "Sonnet seul", "hybrid, reference reader": "hybride, lecteur de référence",
    "hybrid, Haiku reader": "hybride, lecteur Haiku", "hybrid, Jev reader (H1, H3)": "hybride, lecteur Jev (H1, H3)",
    "HESP stop, Jev (H2)": "arrêt HESP, Jev (H2)", "model's table only, Jev (H1)": "table LLM seule, Jev (H1)",
    "mean loss per fault (minutes, +60 if wrong)": "perte moyenne (min, +60 si faux)",
    "rebalance logs (read)": "journaux de rebalancing (lu)", "GC logs (read)": "journaux GC (lu)",
    "quota metrics (read)": "quotas du broker (lu)", "restart (derived)": "redémarrage (calculé)",
    "lag spread (derived)": "lag par partition (calculé)", "CPU profile (derived)": "profil CPU (calculé)",
    "producer rate (derived)": "débit du producteur (calculé)",
    "readings on the wrong side of the reference (%)": "lectures du mauvais côté de la référence (%)",
    "hybrid, Jev reader": "hybride, lecteur Jev", "faults": "pannes",
    "loss per fault (min; beyond 60, wrong cause)": "perte par panne (min ; au-delà de 60, cause fausse)",
    "{name}: mean {m}, {n} wrong": "{name} : moyenne {m}, {n} fausses",
    "routed": "aiguillé", "reference (code)": "référence (code)", "Sonnet, narrow": "Sonnet, question étroite",
    "Haiku + context": "Haiku + contexte", "Haiku, narrow": "Haiku, question étroite",
    "Haiku, sequential neutral": "Haiku, séquentiel neutre", "Haiku, sequential leading": "Haiku, séquentiel orienté",
    "mean loss per fault (min, +60 if wrong)": "perte moyenne (min, +60 si faux)",
    "Haiku, seq. neutral": "Haiku, séq. neutre", "Haiku, seq. leading": "Haiku, séq. orienté",
    "$ per 1,000 diagnoses (log scale)": "$ pour 1 000 diagnostics (échelle log)",
    "mean loss per fault": "perte moyenne par panne", "hybrid, ": "hybride, ",
    "reachable beliefs": "croyances atteignables", "best fit a × H (a = {a})": "meilleure droite a × H (a = {a})",
    "λ × H (λ = 12)": "λ × H (λ = 12)", "entropy H of the belief (bits)": "entropie H de la croyance (bits)",
    "exact value V (minutes)": "valeur exacte V (minutes)", "λ (minutes per bit)": "λ (minutes par bit)",
    "rule S, mean loss": "règle S, perte moyenne", "exact optimum": "optimum exact",
    "plateau 6 to 14": "plateau 6 à 14", "A. Approximating V": "A. Approcher V", "B. Deciding": "B. Décider",
    "a ≈ 15": "a ≈ 15",
}


def t(text: str) -> str:
    return FR.get(text, text) if LANG == "fr" else text


def num(x: float, digits: int) -> str:
    """A number as the paper's language writes it: 8.63 in English, 8,63 in French."""
    out = f"{x:.{digits}f}"
    return out.replace(".", ",") if LANG == "fr" else out


def paper_style():
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                         "svg.fonttype": "path"})   # glyphs drawn: no font swap widens a label past the frame


def save_paper(fig, out: Path, name: str) -> None:
    path = out / (f"paper-{name}.svg" if LANG == "en" else f"paper-{LANG}-{name}.svg")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def paper_study1(out: Path, keep, loss, s) -> None:
    arms = [(t("Sonnet alone"), s.mean(), S2),
            (t("hybrid, reference reader"), loss("voi", "mixed_eb", "reference").mean(), S3),
            (t("hybrid, Haiku reader"), loss("voi", "mixed_eb", "haiku").mean(), S1),
            (t("hybrid, Jev reader (H1, H3)"), loss("voi", "mixed_eb", "jev").mean(), S1),
            (t("priced stop, HESP's table, Jev (H2)"), loss("voi", "eps", "jev").mean(), MUTED),
            (t("HESP stop, Jev (H2)"), loss("hesp", "eps", "jev").mean(), MUTED),
            (t("model's table only, Jev (H1)"), loss("voi", "llm", "jev").mean(), MUTED)]
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    y = np.arange(len(arms))[::-1]
    for yi, (_, v, c) in zip(y, arms):
        ax.barh(yi, v, height=0.62, color=c)
        ax.text(v + 0.4, yi, num(v, 1), va="center", color=INK, fontsize=6.5)
    ax.set_yticks(y, [a[0] for a in arms])
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(a[1] for a in arms) * 1.14)
    ax.set_xlabel(t("mean loss per fault (min, +60 if wrong)"))
    fig.tight_layout()
    save_paper(fig, out, "study1-arms")


def hypothesis_rows():
    """Differences and intervals of H1 to H5, read from the recorded analysis outputs (never typed by hand)."""
    res = ROOT / "kafka" / "results"
    a1 = (res / "analysis-main.txt").read_text()
    a2 = (res / "analysis-study2.txt").read_text()
    rep = (res / "sonnet-repeats.txt").read_text()

    def diff(text, start):
        line = next(ln for ln in text.splitlines() if ln.startswith(start))
        m = re.search(r"diff\s+([+-][\d.]+)\s+\[([+-][\d.]+),\s*([+-][\d.]+)\]", line)
        return tuple(float(g) for g in m.groups())

    def run(label):
        line = next(ln for ln in rep.splitlines() if ln.strip().startswith(label))
        m = re.search(r"([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", line)
        return tuple(float(g) for g in m.groups())

    return [(t("H1  mixed table vs LLM table"), *diff(a1, "H1 "), True),
            (t("H2  priced stop vs HESP's stop"), *diff(a1, "H2 "), True),
            (t("H3  hybrid (Jev) vs Sonnet alone"), *diff(a1, "H3 "), True),
            (t("H4  routed reader vs Jev"), *diff(a2, "H4 "), True),
            (t("H5  routed hybrid vs Sonnet alone"), *diff(a2, "H5 "), True),
            (t("H5 vs Sonnet, run 2 (exploratory)"), *run("run2"), False),
            (t("H5 vs Sonnet, run 3 (exploratory)"), *run("run3"), False),
            (t("H5 vs mean of 3 runs (exploratory)"), *run("mean of runs"), False)]


def paper_forest(out: Path) -> None:
    """The preregistered hypotheses at a glance: each difference with its corrected interval, zero, and H5's margin."""
    rows = hypothesis_rows()
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    y = np.arange(len(rows))[::-1]
    for yi, (_, m, lo, hi, prereg) in zip(y, rows):
        c = INK if prereg else MUTED
        ax.plot([lo, hi], [yi, yi], color=c, linewidth=1.3, solid_capstyle="butt")
        ax.plot(m, yi, "o", color=c, markersize=3.6, markeredgecolor=SURFACE, markeredgewidth=0.6)
        ax.text(max(hi, 2.4) + 0.6, yi, f"{num(m, 2)} [{num(lo, 2)}; {num(hi, 2)}]", va="center", fontsize=5.6,
                color=INK2)
    ax.axvline(0, color=AXIS, linewidth=0.9, zorder=0)
    h5 = [yi for yi, r in zip(y, rows) if r[0].startswith("H5")]
    ax.plot([2, 2], [min(h5) - 0.45, max(h5) + 0.45], color=INK2, linewidth=0.9, linestyle=(0, (3, 2)))
    ax.text(2, max(h5) + 0.5, t("margin of H5"), ha="center", va="bottom", fontsize=5.6, color=INK2)
    ax.axhline(y[2] - 0.5, color=GRID, linewidth=0.8)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-16, 27)
    ax.set_xlabel(t("difference in mean loss (min); below 0, the first arm loses less"), loc="right")
    fig.tight_layout()
    save_paper(fig, out, "hypotheses")


def paper_readers(out: Path) -> None:
    txt = (ROOT / "kafka/results/analysis-main.txt").read_text()
    rows = re.findall(r"^\s+(jev|haiku)\s+(\w+)\s+Brier [\d.]+\s+wrong side ([\d.]+)%", txt, re.M)
    labels = {"rebalance_logs": t("rebalance logs (read)"), "jvm_gc_logs": t("GC logs (read)"),
              "broker_quota_metrics": t("quota metrics (read)"), "restart_consumer": t("restart (derived)"),
              "partition_lag_spread": t("lag spread (derived)"), "consumer_cpu_profile": t("CPU profile (derived)"),
              "producer_rate_graph": t("producer rate (derived)")}
    order = list(labels)
    val = {(r, c): float(p) for r, c, p in rows}
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    y = np.arange(len(order))[::-1]
    h = 0.36
    for off, (r, name, col) in zip([h / 2 + 0.02, -h / 2 - 0.02], [("jev", "Jev", S1), ("haiku", "Haiku", INK2)]):
        v = [val[(r, c)] for c in order]
        ax.barh(y + off, v, height=h, color=col, label=name)
        for yi, vi in zip(y + off, v):
            ax.text(vi + 0.6, yi, num(vi, 1) if vi else "0", va="center", fontsize=5.8, color=INK2)
    ax.set_yticks(y, [labels[c] for c in order])
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(t("readings on the wrong side of the reference (%)"))
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_paper(fig, out, "readers")


def paper_hist(out: Path, keep, loss, s) -> None:
    series = [(t("Sonnet alone"), s, S2), (t("hybrid, reference reader"), loss("voi", "mixed_eb", "reference"), S3),
              (t("hybrid, Jev reader"), loss("voi", "mixed_eb", "jev"), S1)]
    bins = np.arange(0, 85, 4)
    fig, axes = plt.subplots(3, 1, figsize=(COL, 2.9), sharex=True, sharey=True)
    for ax, (name, v, col) in zip(axes, series):
        ax.hist(v, bins=bins, color=col, rwidth=0.88)
        ax.set_title(t("{name}: mean {m}, {n} wrong").format(name=name, m=num(v.mean(), 1), n=int((v >= 60).sum())),
                     fontsize=6.8, fontweight="normal", color=INK2)
        ax.axvline(60, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.set_ylabel(t("faults"))
    axes[-1].set_xlabel(t("loss per fault (min; beyond 60, wrong cause)"))
    fig.tight_layout()
    save_paper(fig, out, "study1-hist")


def study2_rows():
    txt = (ROOT / "kafka/results/analysis-study2.txt").read_text()
    rows = {m.group(1): (float(m.group(2)), float(m.group(3)))
            for m in re.finditer(r"^\s{2}(\w+)\s+([\d.]+)\s+[\d.]+%.*?([\d.]+)$", txt, re.M)}
    so = re.search(r"Sonnet alone\s+([\d.]+)\s+[\d.]+%\s+[\d.]+\s+([\d.]+)", txt)
    return rows, (float(so.group(1)), float(so.group(2)))


#: Jev's tokens are not in the recorded readings; its cost is bounded from its published price (0.042 $ per
#: million input tokens, output free) and the input sizes measured on our calls (640 to 950 tokens):
#: under 0.00004 $ a call, at most three calls per routed diagnosis.
JEV_CALL = 0.00004


def paper_study2(out: Path) -> None:
    rows, (so_loss, so_usd) = study2_rows()
    names = {"routed": t("routed"), "reference": t("reference (code)"), "sonnet_narrow": t("Sonnet, narrow"),
             "haiku_context": t("Haiku + context"), "haiku": t("Haiku, narrow"), "jev": "Jev",
             "haiku_seqn": t("Haiku, sequential neutral"), "haiku_seq": t("Haiku, sequential leading")}
    arms = [(t("Sonnet alone"), so_loss, S2)] + [(names[k], v[0], S3 if k == "routed" else S1)
                                              for k, v in sorted(rows.items(), key=lambda kv: kv[1][0]) if k in names]
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    y = np.arange(len(arms))[::-1]
    for yi, (lab, v, c) in zip(y, arms):
        ax.barh(yi, v, height=0.62, color=c)
        ax.text(v + 0.3, yi, num(v, 2), va="center", fontsize=6.3, color=INK)
    ax.set_yticks(y, [(t("hybrid, ") + a[0]) if a[0] != t("Sonnet alone") else a[0] for a in arms])
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(a[1] for a in arms) * 1.15)
    ax.set_xlabel(t("mean loss per fault (min, +60 if wrong)"))
    fig.tight_layout()
    save_paper(fig, out, "study2-arms")


def paper_cost(out: Path) -> None:
    """Loss against dollars per diagnosis: the routed reader in the frontier's corner, at a fraction of its cost."""
    rows, (so_loss, so_usd) = study2_rows()
    pts = [(t("Sonnet alone"), so_loss, so_usd, S2),
           (t("routed"), rows["routed"][0], rows["routed"][1] + 3 * JEV_CALL, S3),
           (t("Sonnet, narrow"), *rows["sonnet_narrow"], S1), (t("Haiku + context"), *rows["haiku_context"], S1),
           (t("Haiku, narrow"), *rows["haiku"], S1), ("Jev", rows["jev"][0], 4 * JEV_CALL, S1),
           (t("Haiku, seq. neutral"), *rows["haiku_seqn"], MUTED), (t("Haiku, seq. leading"), *rows["haiku_seq"], MUTED)]
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    for lab, loss_, usd, c in pts:
        ax.scatter(usd * 1000, loss_, s=22, color=c, edgecolors=SURFACE, linewidths=0.8, zorder=3)
        dy = {t("Haiku, narrow"): -7, t("Haiku + context"): 4, t("Sonnet, narrow"): -6}.get(lab, 2)   # three points crowd
        ax.annotate(lab, (usd * 1000, loss_), xytext=(4, dy), textcoords="offset points", fontsize=5.8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel(t("$ per 1,000 diagnoses (log scale)"))
    ax.set_ylabel(t("mean loss per fault"))
    ax.set_xlim(0.4, 80)
    fig.tight_layout()
    save_paper(fig, out, "cost")


LOOP_TEXT = {
    "en": {"agent": "Agent: a library that holds the loop, calling models as sensors",
           "choose": "Choose or stop", "choose1": "S = cost − λ · E[bits]",
           "choose2": "run the lowest S while S &lt; 0,", "choose3": "else name the most probable cause",
           "world": "World", "world1": "7 checks:", "world2": "logs,", "world3": "metrics,", "world4": "a restart",
           "run": "run check", "raw": "raw output",
           "router": "Router: is the verdict read or derived?",
           "readoff": "read", "derived": "derived",
           "jev": "Jev", "jev1": "reads the verdict",
           "extract": "Extraction", "extract1": "copies the values",
           "compare": "Comparison", "compare1": "stated in code",
           "p": "p(signal)", "bayes": "Bayes update", "bayes1": "posterior over the six causes",
           "table": "Likelihood", "table1": "table", "table2": "counted on", "table3": "closed", "table4": "incidents",
           "post": "posterior", "lm": "model, used as a sensor", "code": "code"},
    "fr": {"agent": "Agent : une bibliothèque tient la boucle, les modèles ne font que lire",
           "choose": "Choisir ou s'arrêter", "choose1": "S = coût − λ · E[bits]",
           "choose2": "lancer le plus petit S si S &lt; 0,", "choose3": "sinon nommer la cause la plus probable",
           "world": "Monde", "world1": "7 vérifications :", "world2": "journaux,", "world3": "métriques,", "world4": "redémarrage",
           "run": "lancer", "raw": "sortie brute",
           "router": "Aiguilleur : verdict à lire ou à calculer ?",
           "readoff": "à lire", "derived": "à calculer",
           "jev": "Jev", "jev1": "lit le verdict",
           "extract": "Extraction", "extract1": "recopie les valeurs",
           "compare": "Comparaison", "compare1": "écrite dans le code",
           "p": "p(signal)", "bayes": "Mise à jour bayésienne", "bayes1": "postérieure sur les six causes",
           "table": "Table de", "table1": "vraisemblances", "table2": "comptée sur", "table3": "les incidents", "table4": "clos",
           "post": "postérieure", "lm": "modèle, simple capteur", "code": "code"},
}


def paper_loop(out: Path) -> None:
    """The hybrid loop as a hand-written SVG: what was measured, nothing of the target architecture."""
    T = LOOP_TEXT[LANG]
    FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
    MODEL_FILL, MODEL_LINE = "#e9f1fb", S1
    WORLD_FILL = "#f1f0eb"

    def box(x, y, w, h, lines, kind="code", bold_first=True):
        fill, line = {"code": ("#ffffff", INK2), "model": (MODEL_FILL, MODEL_LINE), "world": (WORLD_FILL, MUTED)}[kind]
        out_ = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="{line}" stroke-width="1"/>']
        n = len(lines)
        for i, text in enumerate(lines):
            ty = y + h / 2 + (i - (n - 1) / 2) * 11 + 3.5
            weight = "700" if (i == 0 and bold_first) else "400"
            size = 9 if i == 0 else 8
            out_.append(f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" font-size="{size}" font-weight="{weight}" fill="{INK}">{text}</text>')
        return "\n".join(out_)

    def arrow(points, label=None, lx=None, ly=None, anchor="middle"):
        d = " ".join(f"{'M' if i == 0 else 'L'} {x} {y}" for i, (x, y) in enumerate(points))
        out_ = [f'<path d="{d}" fill="none" stroke="{INK2}" stroke-width="1" marker-end="url(#head)"/>']
        if label:
            out_.append(f'<text x="{lx}" y="{ly}" text-anchor="{anchor}" font-size="7.5" font-style="italic" fill="{INK2}">{label}</text>')
        return "\n".join(out_)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 352" font-family="{FONT}">',
        f'<defs><marker id="head" viewBox="0 0 8 8" refX="7.5" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
        f'<path d="M 0 0 L 8 4 L 0 8 z" fill="{INK2}"/></marker></defs>',
        f'<rect x="0" y="0" width="400" height="352" fill="{SURFACE}"/>',
        # the agent frame
        f'<rect x="8" y="22" width="296" height="300" rx="6" fill="none" stroke="{MUTED}" stroke-dasharray="4 3"/>',
        f'<text x="14" y="15" font-size="8.5" font-weight="700" fill="{INK}">{T["agent"]}</text>',
        box(28, 34, 196, 56, [T["choose"], T["choose1"], T["choose2"], T["choose3"]]),
        box(314, 30, 82, 66, [T["world"], T["world1"], T["world2"], T["world3"], T["world4"]], "world"),
        arrow([(224, 52), (314, 52)], T["run"], 269, 47),
        arrow([(355, 96), (355, 128), (224, 128)], T["raw"], 290, 123),
        box(28, 112, 196, 32, [T["router"]], bold_first=False),
        arrow([(76, 144), (76, 176)], T["readoff"], 80, 163, "start"),
        arrow([(176, 144), (176, 168)], T["derived"], 180, 159, "start"),
        box(28, 176, 96, 36, [T["jev"], T["jev1"]], "model"),
        box(128, 168, 96, 30, [T["extract"], T["extract1"]], "model"),
        arrow([(176, 198), (176, 208)]),
        box(128, 208, 96, 30, [T["compare"], T["compare1"]]),
        arrow([(76, 212), (76, 262)], T["p"], 80, 240, "start"),
        arrow([(176, 238), (176, 262)], T["p"], 180, 253, "start"),
        box(28, 262, 196, 40, [T["bayes"], T["bayes1"]]),
        box(232, 250, 68, 64, [T["table"], T["table1"], T["table2"], T["table3"], T["table4"]]),
        arrow([(232, 282), (224, 282)]),
        # the loop back: posterior to the choice
        arrow([(28, 282), (16, 282), (16, 62), (28, 62)], T["post"], 23, 175, "middle"),
        # legend
        f'<rect x="20" y="332" width="12" height="9" rx="2" fill="{MODEL_FILL}" stroke="{MODEL_LINE}"/>',
        f'<text x="37" y="340" font-size="8" fill="{INK2}">{T["lm"]}</text>',
        f'<rect x="250" y="332" width="12" height="9" rx="2" fill="#ffffff" stroke="{INK2}"/>',
        f'<text x="267" y="340" font-size="8" fill="{INK2}">{T["code"]}</text>',
        "</svg>",
    ]
    svg = "\n".join(parts)
    # the loop-back label reads upward along its line
    svg = svg.replace(f'<text x="23" y="175" text-anchor="middle"', f'<text x="23" y="175" transform="rotate(-90 23 175)" text-anchor="middle"')
    path = out / (f"paper-loop.svg" if LANG == "en" else f"paper-{LANG}-loop.svg")
    path.write_text(svg)
    print(f"wrote {path}")


def paper_lambda(out: Path) -> None:
    """Approximating V and deciding are two things: on the Kafka table V is far from a x H (left), yet the rule
    reaches the exact optimum on the whole plateau of lambda (right); the best-fitting slope falls just past it."""
    d = json.loads((ROOT / "kafka" / "results" / "viz-export.json").read_text())
    c = d["cloud"]["60"]
    h, v = np.array([p[0] for p in c["points"]]), np.array([p[1] for p in c["points"]])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(2 * COL, 2.4))
    a1.scatter(h, v, s=3, color=S1, alpha=0.25, linewidths=0, label=t("reachable beliefs"))
    xs = np.array([0, 2.65])
    a1.plot(xs, c["a"] * xs, color=S2, lw=1.4, ls="--", label=t("best fit a × H (a = {a})").format(a=num(c["a"], 2)))
    a1.plot(xs, 12 * xs, color=INK, lw=1.2, label=t("λ × H (λ = 12)"))
    a1.set_xlabel(t("entropy H of the belief (bits)"))
    a1.set_ylabel(t("exact value V (minutes)"))
    a1.set_xlim(0, 2.7)
    a1.set_ylim(0, None)
    a1.set_title(t("A. Approximating V") + f"   (M = 60, R² = {num(c['r2'], 2)})")
    a1.legend(loc="upper left", fontsize=6)
    lam = np.array([p["lam"] for p in d["curve"]])
    loss = np.array([p["minutes"] + 60 * p["wrong"] for p in d["curve"]])
    a2.axvspan(6, 14, color=S1, alpha=0.08, lw=0, label=t("plateau 6 to 14"))
    a2.axhline(d["optimum"]["60"], color=S3, lw=1.6, label=t("exact optimum"))
    a2.plot(lam, loss, color=S1, lw=1.4, marker="o", ms=2.2, label=t("rule S, mean loss"))
    for x, lab in ((12, "λ = 12"), (15, t("a ≈ 15"))):
        y = float(loss[lam == x][0])
        a2.scatter([x], [y], s=22, color=S2 if x == 15 else INK, zorder=4)
        a2.annotate(f"{lab} : {num(y, 2)}" if LANG == "fr" else f"{lab}: {num(y, 2)}", (x, y),
                    xytext=(-34, -14) if x == 12 else (2, 14), textcoords="offset points", fontsize=6, color=INK2)
    a2.set_ylim(0, 20)
    a2.set_xlim(0, 30)
    a2.set_xlabel(t("λ (minutes per bit)"))
    a2.set_ylabel(t("mean loss per fault"))
    a2.set_title(t("B. Deciding") + "   (M = 60)")
    a2.set_yticks([0, 5, 10, 15, 20])
    a2.legend(loc="lower right", fontsize=6)
    if LANG == "fr":
        from matplotlib.ticker import FuncFormatter
        a1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
    fig.tight_layout()
    save_paper(fig, out, "lambda")


def paper(out: Path, lang: str = "en") -> None:
    global LANG
    LANG = lang
    paper_style()
    paper_loop(out)
    keep, loss, s = study1()
    paper_study1(out, keep, loss, s)
    paper_hist(out, keep, loss, s)
    paper_readers(out)
    paper_study2(out)
    paper_forest(out)
    paper_cost(out)
    paper_lambda(out)


def main() -> None:
    if "--dump-values" in sys.argv:
        dump_values()
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str("figures"))
    ap.add_argument("names", nargs="*", default=["value", "msweep", "arms", "readers", "hist", "study2"])
    ap.add_argument("--paper", action="store_true", help="the figures of the paper (studies 1 and 2)")
    ap.add_argument("--lang", default="en", choices=["en", "fr"], help="language of the paper's figures")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if a.paper:
        paper(out, a.lang)
        return
    if "value" in a.names:
        fig_value(out)
    if "msweep" in a.names:
        fig_msweep(out)
    if "study2" in a.names:
        fig_study2(out)
    if "readers" in a.names:
        fig_readers(out)
    if {"arms", "hist"} & set(a.names):
        keep, loss, s = study1()
        if "arms" in a.names:
            fig_arms(out, keep, loss, s)
        if "hist" in a.names:
            fig_hist(out, keep, loss, s)


if __name__ == "__main__":
    main()
