"""Real incidents on kafka-lab: inject a fault, run every check, record it all.

Each episode: reset the lab (fresh topic and group), let it run healthy,
inject one of the six faults, wait for the lag to build, then run the seven
checks of the Kafka scenario and store their raw output with the true cause.

Every check is run in every episode, so any policy (least action, HESP's
ranking, an LLM alone) can be replayed offline on the same outputs: an agent
only "sees" the checks it chose. Reading the outputs (Jev, Haiku) is a
separate step on the stored text.

The lab lives in ../kafka-lab; it must be up:
    docker compose -p stack-kafka-lab -f ../kafka-lab/compose.yaml up -d

usage: python episodes.py --per-cause 1 --tag pilot      # 6 episodes, ~27 min
       python episodes.py --per-cause 20 --tag main      # 120 episodes, ~9 h
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LAB = Path(os.environ.get("KAFKA_LAB", Path(__file__).resolve().parent.parent / "kafka-lab")).expanduser()
OUT = Path(__file__).resolve().parent / "results"

CAUSES = ["rebalancing", "slow_consumer", "partition_skew", "broker_throttling", "gc_pauses", "upstream_burst"]
# The checks of scenarios.py "kafka", mapped to their scripts. restart_consumer
# changes the state of the group, so it runs last.
PROBES = {
    "rebalance_logs": "rebalance_logs.sh",
    "partition_lag_spread": "partition_lag_spread.sh",
    "producer_rate_graph": "producer_rate_graph.sh",
    "broker_quota_metrics": "broker_quota_metrics.sh",
    "jvm_gc_logs": "jvm_gc_logs.sh",
    "consumer_cpu_profile": "consumer_cpu_profile.sh",
    "restart_consumer": "restart_consumer.sh",
}
WARMUP_S = 130   # healthy run after a reset: the start-up joins leave rebalance_logs' 2-minute window
SYMPTOM_S = 60   # time between injection and the alert


def sh(*args: str, timeout: int = 120) -> tuple[int, str]:
    r = subprocess.run(args, cwd=LAB, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip()


def total_lag() -> int:
    code, out = sh("bash", "-c", "source lib.sh && total_lag")
    return int(out) if code == 0 and out.isdigit() else -1


def episode(i: int, cause: str) -> dict:
    t0 = time.time()
    code, out = sh("./reset.sh", timeout=180)
    if code != 0:
        raise RuntimeError(f"reset failed: {out}")
    time.sleep(WARMUP_S)
    healthy = total_lag()
    sh(f"./inject/{cause}.sh")
    injected = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lags = []
    for _ in range(SYMPTOM_S // 15):
        time.sleep(15)
        lags.append(total_lag())
    probes, seconds = {}, {}
    for name, script in PROBES.items():
        s = time.time()
        code, out = sh(f"./probes/{script}", timeout=180)
        probes[name] = out if code == 0 else f"(check failed, exit {code})\n{out}"
        seconds[name] = round(time.time() - s, 1)
    sh("./inject/clear.sh")
    return {"episode": i, "cause": cause, "injected": injected, "healthy_lag": healthy,
            "lag_after_injection": lags, "probes": probes, "probe_seconds": seconds,
            "duration_s": round(time.time() - t0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cause", type=int, default=1)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    # balanced blocks: each block holds every cause once, in a shuffled order
    rng = random.Random(a.seed)
    order = []
    for _ in range(a.per_cause):
        block = CAUSES[:]
        rng.shuffle(block)
        order += block

    path = OUT / f"episodes-{a.tag}.jsonl"
    done = sum(1 for _ in path.open()) if path.exists() else 0
    if done:
        print(f"resuming {path.name} after {done} episodes", flush=True)
    for i, cause in enumerate(order[done:], start=done):
        rec = episode(i, cause)
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{i + 1}/{len(order)} {cause:18} healthy {rec['healthy_lag']:>6} -> "
              f"{rec['lag_after_injection']}  ({rec['duration_s']} s)", flush=True)
    sh("./reset.sh", timeout=180)


if __name__ == "__main__":
    sys.exit(main())
