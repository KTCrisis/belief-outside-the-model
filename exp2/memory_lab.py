"""Closed investigations kept in mem7, and what an agent learns from them.

seed_history writes past incidents into a lab mem7 (never the real one): one
fact per incident, dated at its closing day (valid_from), with its true cause
and the results of the checks that were run then (a few, not all, as in real
investigations). learn reads them back and counts, weighting each incident
by its age (half-life in days), with a Beta(1, 1) / Dirichlet(1) prior so an
unseen pair never reads as 0 or 1:
  P(positive | test, cause) = (w_pos + 1) / (w_run + 2)
  P(cause) = (w_cause + 1) / (w_total + n_causes)

usage: python memory_lab.py seed [--n 120] [--seed 5]   # writes the history
"""
import argparse
import json
import re
from datetime import datetime, timedelta, timezone

import numpy as np
from mem7 import Mem7

LAB_URL = "http://127.0.0.1:9077"  # the lab instance, MEM7_DIR=~/bayes-action/mem7-lab
AGENT = "bayes-lab"


def lab() -> Mem7:
    return Mem7(LAB_URL)


def draw_history(case, n: int, seed: int, now: datetime, checks_per_incident: int = 4) -> list[dict]:
    """n closed incidents over the past year, each with a few checks run, in read_history's shape."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        cause = int(rng.choice(len(case.causes), p=case.prior))
        run = rng.choice(len(case.tests), size=min(checks_per_incident, len(case.tests)), replace=False)
        results = {case.tests[j]: bool(rng.random() < case.lik[j, cause]) for j in sorted(run)}
        closed = now - timedelta(days=int(rng.integers(1, 365)))
        out.append({"cause": case.causes[cause], "results": results, "closed": closed.strftime("%Y-%m-%d")})
    return out


def seed_history(case, n: int, seed: int, now: datetime, m: Mem7, checks_per_incident: int = 4) -> None:
    """Writes draw_history's incidents into the lab mem7, dated at their closing day."""
    for i, inc in enumerate(draw_history(case, n, seed, now, checks_per_incident)):
        value = json.dumps({"cause": inc["cause"], "results": inc["results"]})
        m.store(f"incident/{case.name}/{i:04d}", value, tags=[case.name, "closed-incident"],
                agent=AGENT, valid_from=inc["closed"])


def read_history(case, m: Mem7, as_of: str | None = None) -> list[dict]:
    """The closed incidents of this scenario, as mem7 holds them (or held then)."""
    text = m.recall(tags=[case.name, "closed-incident"], limit=10000, **({"as_of": as_of} if as_of else {}))
    out = []
    for block in text.split("## ")[1:]:
        lines = block.strip().splitlines()
        if len(lines) < 2 or not lines[1].startswith("{"):
            continue
        valid = next((ln for ln in lines if ln.startswith("Valid: ")), "")
        day = re.search(r"Valid: (\d{4}-\d{2}-\d{2})", valid)
        out.append({**json.loads(lines[1]), "closed": day.group(1) if day else None})
    return out


def learn(case, history: list[dict], now: datetime, half_life_days: float = 180.0,
          table_prior: np.ndarray | None = None, strength: float = 2.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Prior over causes and P(positive | test, cause), from weighted counts.

    Each cell starts from Beta(strength * q, strength * (1 - q)), q from
    table_prior: by default q = 0.5 and strength 2, the uniform Beta(1, 1).
    Passing the LLM's table as table_prior makes it the starting belief that
    the counts correct, worth `strength` incidents.
    """
    nt, nc = len(case.tests), len(case.causes)
    w_cause = np.zeros(nc)
    w_run = np.zeros((nt, nc))
    w_pos = np.zeros((nt, nc))
    for inc in history:
        age = (now - datetime.fromisoformat(inc["closed"]).replace(tzinfo=timezone.utc)).days if inc["closed"] else 0
        w = 0.5 ** (age / half_life_days)
        c = case.causes.index(inc["cause"])
        w_cause[c] += w
        for t, positive in inc["results"].items():
            j = case.tests.index(t)
            w_run[j, c] += w
            w_pos[j, c] += w * positive
    prior = (w_cause + 1) / (w_cause.sum() + nc)
    q = np.full((nt, nc), 0.5) if table_prior is None else table_prior
    table = (w_pos + strength * q) / (w_run + strength)
    stats = {"incidents": len(history), "weight": round(float(w_cause.sum()), 1),
             "min_runs_per_cell": round(float(w_run.min()), 1)}
    return prior, table, stats


def summary_for_llm(case, history: list[dict]) -> str:
    """The same history as plain counts, for the LLM agent given memory."""
    lines = [f"Past closed incidents ({len(history)}), per check: how often it came out positive, by true cause."]
    for t in case.tests:
        cells = []
        for c in case.causes:
            runs = [inc["results"][t] for inc in history if inc["cause"] == c and t in inc["results"]]
            cells.append(f"{c} {sum(runs)}/{len(runs)}")
        lines.append(f"- {t}: " + ", ".join(cells))
    return "\n".join(lines)


def main() -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from exp2 import Case

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["seed"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    now = datetime(2026, 10, 4, tzinfo=timezone.utc)
    m = lab()
    for name in ["kafka", "internet"]:
        case = Case(name)
        seed_history(case, a.n, a.seed + (0 if name == "kafka" else 1), now, m)
        hist = read_history(case, m)
        prior, table, stats = learn(case, hist, now)
        print(f"{name}: read back {stats}, table MAE vs truth {np.abs(table - case.lik).mean():.3f}, "
              f"prior {np.round(prior, 2).tolist()}")


if __name__ == "__main__":
    main()
