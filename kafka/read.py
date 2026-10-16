"""Read the raw outputs of the real-fault episodes (preregistration, section 4).

Three readers, one question per check, frozen in PREREGISTRATION.md:
- reference: deterministic rules on the raw text, written on the pilot only;
- jev: Jev's probability through Cloudflare Workers AI (exp2/jev_reader.py, cached);
- haiku: claude-haiku-4-5, median of 5 answers.

Each line of results/readings-<tag>.jsonl is one (episode, check) with every
reader asked. The run resumes: pairs already read are skipped.

usage: python read.py --tag pilot --readers reference          # free
       python read.py --tag main --readers reference,jev,haiku # API spend
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp2"))
OUT = HERE / "results"

# check -> (what the check shows, the signal, its absence); the questions of section 4
QUESTIONS = {
    "rebalance_logs": (
        "group coordination lines from the consumers' logs over the last two minutes",
        "the consumers joined, left or rebalanced the group within the window",
        "no consumer joined, left or rebalanced the group within the window"),
    "partition_lag_spread": (
        "the consumer group's lag per partition (kafka-consumer-groups --describe)",
        "the lag is concentrated on one or two partitions rather than spread",
        "the lag is spread over the partitions, none carries most of it"),
    "producer_rate_graph": (
        "the incoming messages per second on the topic, per partition, over 10 seconds",
        "the incoming rate is clearly above its usual level of about 350 messages per second",
        "the incoming rate is at its usual level of about 350 messages per second"),
    "broker_quota_metrics": (
        "the client quotas set on the broker, and the consumers' fetch throttle time",
        "a client quota is set, or the consumers suffer a notable throttle time",
        "no client quota is set and the consumers are not throttled"),
    "jvm_gc_logs": (
        "the last garbage collection events of each consumer's JVM",
        "repeated full GC pauses caused by a lack of memory appear",
        "no repeated full GC pauses caused by a lack of memory appear"),
    "consumer_cpu_profile": (
        "each consumer's CPU use over 5 seconds and where its main thread is",
        "the consumers saturate the CPU in their message processing",
        "the consumers do not saturate the CPU in their message processing"),
    "restart_consumer": (
        "the total lag before restarting one consumer and 10, 20 and 30 seconds after",
        "the total lag decreases after the restart",
        "the total lag does not decrease after the restart"),
}


# --------------------------------------------------------------------------- #
#  the deterministic reference
# --------------------------------------------------------------------------- #
def _lags(text: str) -> list[int]:
    out = []
    for line in text.splitlines():
        f = line.split()
        if len(f) > 5 and f[0] == "lab" and f[1] == "orders" and f[5].isdigit():
            out.append(int(f[5]))
    return out


def reference(check: str, text: str) -> int:
    if check == "rebalance_logs":
        return int(bool(re.search(r"joined|Revok|rejoin|\(Re-\)joining", text)))
    if check == "partition_lag_spread":
        lags = sorted(_lags(text), reverse=True)
        return int(len(lags) >= 2 and lags[0] > 3 * statistics.median(lags[1:]))
    if check == "producer_rate_graph":
        m = re.search(r"total: (\d+) msg/s", text)
        return int(bool(m) and int(m.group(1)) >= 600)
    if check == "broker_quota_metrics":
        throttle = [int(x) for x in re.findall(r"fetch-throttle-time-avg (\d+)", text)]
        return int("consumer_byte_rate" in text or (bool(throttle) and statistics.mean(throttle) > 100))
    if check == "jvm_gc_logs":
        return int(text.count("Pause Full (Allocation Failure)") >= 3)
    if check == "consumer_cpu_profile":
        busy = 0
        for block in re.split(r"(?m)^\$ ", text)[1:]:
            cpu = re.search(r"cpu (\d+)%", block)
            if cpu and int(cpu.group(1)) >= 80 and "LabConsumer.enrichOrder" in block:
                busy += 1
        return int(busy >= 2)
    if check == "restart_consumer":
        before = re.search(r"before restart: (\d+)", text)
        after = re.search(r"30 s after restart: (\d+)", text)
        return int(bool(before and after) and int(after.group(1)) < int(before.group(1)))
    raise ValueError(check)


# --------------------------------------------------------------------------- #
#  the model readers
# --------------------------------------------------------------------------- #
def jev(check: str, text: str) -> float:
    import jev_reader
    what, signal, absence = QUESTIONS[check]
    return jev_reader.read_signal(what, text, signal, absence)


HAIKU = "claude-haiku-4-5"
HAIKU_SAMPLES = 5
haiku_usage = {"in": 0, "out": 0}


def haiku(check: str, text: str) -> float:
    import anthropic
    client = anthropic.Anthropic()
    what, signal, absence = QUESTIONS[check]
    draws = []
    for _ in range(HAIKU_SAMPLES):
        r = client.messages.create(
            model=HAIKU,
            max_tokens=200,
            messages=[{"role": "user", "content": (
                f"An on-call engineer ran a check on a Kafka platform. The check shows {what}.\n"
                f"It returned:\n\"\"\"\n{text}\n\"\"\"\n\n"
                f"What is the probability that this output shows the signal: {signal}? "
                f"(The opposite would be: {absence}.)"
            )}],
            output_config={"format": {"type": "json_schema", "schema": {
                "type": "object", "properties": {"probability": {"type": "number"}},
                "required": ["probability"], "additionalProperties": False}}},
        )
        haiku_usage["in"] += r.usage.input_tokens
        haiku_usage["out"] += r.usage.output_tokens
        draws.append(float(json.loads(next(b.text for b in r.content if b.type == "text"))["probability"]))
    return min(max(statistics.median(draws), 0.01), 0.99)


READERS = {"reference": reference, "jev": jev, "haiku": haiku}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--readers", default="reference")
    a = ap.parse_args()
    readers = a.readers.split(",")
    episodes = [json.loads(l) for l in (OUT / f"episodes-{a.tag}.jsonl").open()]
    path = OUT / f"readings-{a.tag}.jsonl"
    done = {}
    if path.exists():
        for l in path.open():
            r = json.loads(l)
            done[(r["episode"], r["check"])] = r
    with path.open("a") as f:
        for e in episodes:
            for check, text in e["probes"].items():
                row = done.get((e["episode"], check), {"episode": e["episode"], "cause": e["cause"], "check": check})
                missing = [r for r in readers if r not in row]
                if not missing:
                    continue
                for r in missing:
                    row[r] = READERS[r](check, text)
                f.write(json.dumps(row) + "\n")
    print(f"{path.name}: {len(episodes)} episodes x {len(QUESTIONS)} checks, readers {readers}")
    if "haiku" in readers:
        print(f"haiku tokens: {haiku_usage}")


if __name__ == "__main__":
    main()
