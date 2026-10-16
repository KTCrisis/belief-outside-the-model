"""Baseline without an LLM: a decision tree learned from the closed incidents.

The tree is fit on the same histories as the memory (120 incidents, 4 of the
checks run in each, the others missing), with check results as features and
the cause as label. As a policy: the feature a node splits on is the next
check to run (its cost is paid), the result picks the branch, a leaf
concludes on its majority cause. Standard trees ignore check costs; that is
part of what the comparison shows. Evaluated exactly, like tables.py: every
branch weighted by its true probability, perfect reader.

usage: python tree.py    # no API call
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp2 import NOW, Case  # noqa: E402
from memory_lab import draw_history, learn  # noqa: E402
from tables import ebayes_strength, expected_loss, medians_of  # noqa: E402


def fit(case: Case, history: list[dict], depth: int | None) -> DecisionTreeClassifier:
    X = np.full((len(history), len(case.tests)), np.nan)
    y = np.array([case.causes.index(inc["cause"]) for inc in history])
    for i, inc in enumerate(history):
        for t, positive in inc["results"].items():
            X[i, case.tests.index(t)] = float(positive)
    return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=3, random_state=0).fit(X, y)


def tree_loss(case: Case, clf: DecisionTreeClassifier) -> dict:
    """Exact expected loss of the tree used as a policy, under the true table."""
    tr = clf.tree_

    def walk(node, weight, minutes):
        if tr.children_left[node] == -1:
            answer = int(clf.classes_[tr.value[node][0].argmax()])
            mass = weight.sum()
            return np.array([mass * minutes + case.miss * (mass - weight[answer]), weight[answer], mass * minutes])
        j = tr.feature[node]
        like = case.lik[j]
        # results are 0/1; split at threshold 0.5 sends 0 (negative) left, 1 (positive) right
        return (walk(tr.children_left[node], weight * (1 - like), minutes + case.cost[case.tests[j]])
                + walk(tr.children_right[node], weight * like, minutes + case.cost[case.tests[j]]))

    loss, right, minutes = walk(0, case.prior.copy(), 0.0)
    return {"loss": float(loss), "accuracy": float(right), "minutes": float(minutes)}


def main(histories: int = 10, size: int = 120) -> None:
    for name in ["kafka", "internet"]:
        case = Case(name)
        print(f"\n== {name}: exact expected loss, {histories} histories of {size} incidents "
              f"(truth {expected_loss(case, case.lik)['loss']:.2f})")
        rows = {}
        for h in range(histories):
            hist = draw_history(case, size, 1000 + h, NOW)
            for depth in (1, 2, 3, 4, 5, None):
                r = tree_loss(case, fit(case, hist, depth))
                rows.setdefault(f"tree depth {depth or 'free'}", []).append((r["loss"], r["minutes"], r["accuracy"]))
            r = expected_loss(case, learn(case, hist, NOW, 1e9)[1])
            rows.setdefault("hybrid, mem7 table alone", []).append((r["loss"], r["minutes"], r["accuracy"]))
            for t in medians_of(name):
                k = ebayes_strength(case, hist, t)
                r = expected_loss(case, learn(case, hist, NOW, 1e9, table_prior=t, strength=k)[1])
                rows.setdefault("hybrid, LLM table + mem7 (k empirical)", []).append((r["loss"], r["minutes"], r["accuracy"]))
        print(f"{'policy':40} {'loss':>6} {'max':>6} {'minutes':>8} {'accuracy':>9}")
        for label, xs in rows.items():
            a = np.array(xs)
            print(f"{label:40} {a[:, 0].mean():6.2f} {a[:, 0].max():6.2f} {a[:, 1].mean():8.2f} {a[:, 2].mean():9.3f}")


if __name__ == "__main__":
    main()
