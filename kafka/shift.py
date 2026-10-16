"""The shifted copy of study 2 (PREREGISTRATION-2.md, section 7).

Same episodes, same raw outputs, three changes of format that a parser written
by hand does not expect:
- rates in messages per minute instead of per second;
- CPU as a fraction of one CPU (0.91) instead of a percentage (91%);
- the columns of the lag table reordered and its headers renamed.

usage: python shift.py --tag study2      # writes results/episodes-study2-shifted.jsonl
"""
import argparse
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results"
HEADERS = {"GROUP": "CONSUMER_GROUP", "TOPIC": "STREAM", "PARTITION": "PART", "CURRENT-OFFSET": "COMMITTED",
           "LOG-END-OFFSET": "HEAD", "LAG": "BACKLOG", "CONSUMER-ID": "MEMBER", "HOST": "ADDRESS", "CLIENT-ID": "CLIENT"}
ORDER = ["PARTITION", "LAG", "CURRENT-OFFSET", "LOG-END-OFFSET", "CONSUMER-ID", "CLIENT-ID", "HOST", "GROUP", "TOPIC"]


def rates(text: str) -> str:
    text = text.replace("incoming messages per second", "incoming messages per minute")
    return re.sub(r"(\d+) msg/s", lambda m: f"{int(m.group(1)) * 60} msg/min", text)


def cpu(text: str) -> str:
    return re.sub(r"cpu (\d+)% over", lambda m: f"cpu {int(m.group(1)) / 100:.2f} over", text)


def lag_table(text: str) -> str:
    out = []
    head = None
    for line in text.splitlines():
        f = line.split()
        if f[:3] == ["GROUP", "TOPIC", "PARTITION"]:
            head = f
            out.append("  ".join(f"{HEADERS[h]:<14}" for h in ORDER).rstrip())
        elif head and len(f) == len(head):
            row = dict(zip(head, f))
            out.append("  ".join(f"{row[h]:<14}" for h in ORDER).rstrip())
        else:
            out.append(line)
    return "\n".join(out)


def shift(ep: dict) -> dict:
    p = dict(ep["probes"])
    p["producer_rate_graph"] = rates(p["producer_rate_graph"])
    p["consumer_cpu_profile"] = cpu(p["consumer_cpu_profile"])
    p["partition_lag_spread"] = lag_table(p["partition_lag_spread"])
    return {**ep, "probes": p, "shifted": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    src = OUT / f"episodes-{a.tag}.jsonl"
    dst = OUT / f"episodes-{a.tag}-shifted.jsonl"
    eps = [json.loads(l) for l in src.open()]
    with dst.open("w") as f:
        for e in eps:
            f.write(json.dumps(shift(e)) + "\n")
    print(f"{dst.name}: {len(eps)} episodes")


if __name__ == "__main__":
    main()
