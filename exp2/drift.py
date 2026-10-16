"""Drift: the world changes under the table learned from closed incidents.

A history of 240 incidents over a year. Halfway, one check changes its
signature (a log format, a box model): its P(positive | main cause) drops.
The table is learned from the whole history, then evaluated exactly against
the world as it is now (after the drift):

- all history, no forgetting (half-life infinite);
- forgetting, half-lives of 90 and 30 days (memory_lab.learn's weighting);
- only the incidents after the drift (an oracle that knows when it changed).

Each with the LLM's median table as prior (k = 10) and with none. The LLM's
table describes the world as it was usually described, not the drifted one.
Checks run per incident: four at random (as the lab histories).

usage: python drift.py   # no API call
"""
import copy
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp2 import NOW, Case  # noqa: E402
from memory_lab import learn  # noqa: E402
from tables import expected_loss, medians_of  # noqa: E402

SIZE = 240
HISTORIES = 5
K = 10.0
# (check, cause, new P(positive | cause)): the main signature of a check weakens.
DRIFT = {
    "kafka": ("jvm_gc_logs", "gc_pauses", 0.30),  # GC logging turned down
    "internet": ("look_at_box_lights", "box_frozen", 0.15),  # new box model, lights stay green
}


def drifted(case: Case) -> Case:
    test, cause, value = DRIFT[case.name]
    new = copy.deepcopy(case)
    new.lik = case.lik.copy()
    new.lik[case.tests.index(test), case.causes.index(cause)] = value
    return new


def history(old: Case, new: Case, seed: int) -> list[dict]:
    """Incidents spread over the past year; those closed in the last six months follow the new world."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(SIZE):
        age = int(rng.integers(1, 365))
        world = new if age < 182 else old
        cause = int(rng.choice(len(old.causes), p=old.prior))
        run = rng.choice(len(old.tests), size=4, replace=False)
        results = {old.tests[j]: bool(rng.random() < world.lik[j, cause]) for j in sorted(run)}
        out.append({"cause": old.causes[cause], "results": results,
                    "closed": (NOW - timedelta(days=age)).strftime("%Y-%m-%d")})
    return out


def main() -> None:
    for name in ["kafka", "internet"]:
        old = Case(name)
        new = drifted(old)
        test, cause, value = DRIFT[name]
        llm = medians_of(name)
        print(f"\n== {name}: {test} under {cause}: {old.lik[old.tests.index(test), old.causes.index(cause)]:.2f} -> {value:.2f}"
              f" halfway; {HISTORIES} histories of {SIZE} x {len(llm)} LLM tables; loss in the drifted world")
        print(f"  {'agent uses':38} {'loss':>7} {'p90':>7}  {'cell learned':>12}")
        print(f"  {'true table, old world':38} {expected_loss(new, old.lik)['loss']:7.2f}")
        print(f"  {'true table, new world':38} {expected_loss(new, new.lik)['loss']:7.2f}")
        cell = (new.tests.index(test), new.causes.index(cause))
        for prior_label, priors in [("LLM, k=10", llm), ("none", [None])]:
            for label, half, recent_only in [("all, no forgetting", 1e9, False), ("half-life 90 days", 90, False),
                                             ("half-life 30 days", 30, False), ("after the drift only", 1e9, True)]:
                losses, cells = [], []
                for h in range(HISTORIES):
                    hist = history(old, new, 9000 + h)
                    if recent_only:
                        hist = [x for x in hist if (NOW - timedelta(days=182)).strftime("%Y-%m-%d") < x["closed"]]
                    for t in priors:
                        kw = {"table_prior": t, "strength": K} if t is not None else {}
                        table = learn(new, hist, NOW, half, **kw)[1]
                        losses.append(expected_loss(new, table)["loss"])
                        cells.append(table[cell])
                xs = np.array(losses)
                print(f"  {prior_label + ', ' + label:38} {xs.mean():7.2f} {np.percentile(xs, 90):7.2f}  {np.mean(cells):12.2f}")
            if prior_label.startswith("LLM"):
                xs = np.array([expected_loss(new, t)["loss"] for t in llm])
                print(f"  {'LLM table alone':38} {xs.mean():7.2f} {np.percentile(xs, 90):7.2f}  {np.mean([t[cell] for t in llm]):12.2f}")


if __name__ == "__main__":
    main()
