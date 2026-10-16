"""Informative censoring: does a table learned from the agent's own choices go wrong?

So far the lab histories ran four checks drawn at random per incident
(memory_lab.draw_history). A real history only holds the checks the agent
chose to run, and it chooses them with the table it is learning: a check it
judges useless is rarely run, so never learned, so still judged useless.

Four ways to fill a history of closed incidents, then one table learned from
it (the LLM's median table as prior, strength k = 10, as in the series) and
evaluated exactly (tables.expected_loss, perfect reader):

- complete : every check run on every incident (no censoring);
- random4  : four checks at random (the lab histories so far);
- policy   : the agent runs least action with the table learned so far,
             incident after incident; only its own checks enter the history;
- thompson : the same, but each incident acts on a table drawn from the
             Beta posterior of each cell (exploration).

Also the same four with no LLM prior (Beta(1, 1), mem7 alone).

usage: python censoring.py   # no API call
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp2 import NOW, Case, least_action_next  # noqa: E402
from memory_lab import learn  # noqa: E402
from tables import counts, expected_loss, medians_of  # noqa: E402

K = 10.0
SIZE = 120
HISTORIES = 5


def run_incident(case: Case, cause: int, table: np.ndarray, rng: np.random.Generator) -> dict:
    """One incident investigated with `table`; the results of the checks run, from the truth."""
    post, left, results = case.prior.copy(), list(case.tests), {}
    while (t := least_action_next(case, post, left, table)) is not None:
        left.remove(t)
        j = case.tests.index(t)
        positive = bool(rng.random() < case.lik[j, cause])
        results[t] = positive
        like = table[j] if positive else 1 - table[j]
        post = post * like / (post * like).sum()
    return results


def fill(case: Case, mode: str, prior_table: np.ndarray | None, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    q = np.full_like(case.lik, 0.5) if prior_table is None else prior_table
    k = K if prior_table is not None else 2.0
    history: list[dict] = []
    for _ in range(SIZE):
        cause = int(rng.choice(len(case.causes), p=case.prior))
        if mode == "complete":
            results = {t: bool(rng.random() < case.lik[j, cause]) for j, t in enumerate(case.tests)}
        elif mode == "random4":
            run = rng.choice(len(case.tests), size=4, replace=False)
            results = {case.tests[j]: bool(rng.random() < case.lik[j, cause]) for j in sorted(run)}
        else:
            pos, n = counts(case, history)
            if mode == "policy":
                table = (pos + k * q) / (n + k)
            else:  # thompson
                table = rng.beta(pos + k * q, n - pos + k * (1 - q))
            results = run_incident(case, cause, table, rng)
        history.append({"cause": case.causes[cause], "results": results, "closed": None})
    return history


def main() -> None:
    for name in ["kafka", "internet"]:
        case = Case(name)
        llm = medians_of(name)
        truth = expected_loss(case, case.lik)["loss"]
        print(f"\n== {name}: {HISTORIES} histories of {SIZE} incidents x {len(llm)} LLM tables, "
              f"exact loss with the learned table (truth {truth:.2f})")
        print(f"{'prior':10} {'history':10} {'loss':>7} {'p90':>7} {'max':>7} {'MAE':>6} {'checks/inc':>10} {'thinnest cell':>13}")
        for prior_label, priors in [("LLM, k=10", llm), ("none", [None])]:
            for mode in ["complete", "random4", "policy", "thompson"]:
                losses, maes, checks, thin = [], [], [], []
                for h in range(HISTORIES):
                    for i, t in enumerate(priors):
                        hist = fill(case, mode, t, seed=7000 + 100 * h + i)
                        kw = {"table_prior": t, "strength": K} if t is not None else {}
                        table = learn(case, hist, NOW, 1e9, **kw)[1]
                        losses.append(expected_loss(case, table)["loss"])
                        maes.append(np.abs(table - case.lik).mean())
                        checks.append(np.mean([len(x["results"]) for x in hist]))
                        thin.append(counts(case, hist)[1].min())
                xs = np.array(losses)
                print(f"{prior_label:10} {mode:10} {xs.mean():7.2f} {np.percentile(xs, 90):7.2f} {xs.max():7.2f} "
                      f"{np.mean(maes):6.3f} {np.mean(checks):10.2f} {np.mean(thin):13.1f}")
            if prior_label.startswith("LLM"):
                alone = np.array([expected_loss(case, t)["loss"] for t in llm])
                print(f"{'LLM, k=10':10} {'(no hist.)':10} {alone.mean():7.2f} {np.percentile(alone, 90):7.2f} {alone.max():7.2f}")


if __name__ == "__main__":
    main()
