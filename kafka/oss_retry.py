"""Re-read, with gpt-oss, the readings where an extraction failed (exploratory, not preregistered).

On study 2, gpt-oss 20B sometimes answers {"partitions": []}: valid JSON the schema allows, so the sample counts as
failed (0.5). Each failed sample is asked again, up to 5 tries, with the same prompt and schema; the result is a
new reader, routed_oss_r, equal to routed_oss wherever nothing failed. routed_oss is left as the night run wrote it.

usage: python oss_retry.py --tag study2
"""
import argparse
import json
import statistics

import httpx

import read2 as R

TRIES = 5


def sample(check, text, q):
    what, schema = R.EXTRACT[check]
    prompt = (f"An on-call engineer ran a check on a Kafka platform. The check shows {q[check][0]}.\n"
              f"It returned:\n\"\"\"\n{text}\n\"\"\"\n\n"
              f"Copy from this output {what}. Copy the values exactly as written; do not interpret or judge them.")
    for i in range(TRIES):
        try:
            d, _ = R.call("ollama:gpt-oss:20b", [{"role": "user", "content": prompt}], schema, max_tokens=1500)
            v = R.verdict(check, d)
        except (ValueError, KeyError, TypeError):
            v = R.FAILED
        if v != R.FAILED:
            return v, i + 1
    return R.FAILED, TRIES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    path = f"results/readings2x-{a.tag}.jsonl"
    rows = [json.loads(line) for line in open(path)]
    eps = {e["episode"]: e for e in map(json.loads, open(f"results/episodes-{a.tag}.jsonl"))}
    todo = [r for r in rows if r.get("routed_oss_failed")]
    print(f"{len(todo)} readings to redo", flush=True)
    for n, r in enumerate(todo, 1):
        text = eps[r["episode"]]["probes"][r["check"]]
        # the samples that succeeded are not stored: all three are drawn again
        vs, tries = [], 0
        for _ in range(3):
            v, t = sample(r["check"], text, R.QUESTIONS)
            vs.append(v)
            tries += t
        r["routed_oss_r"] = statistics.median(vs)
        r["routed_oss_r_failed"] = sum(v == R.FAILED for v in vs)
        r["routed_oss_r_tries"] = tries
        print(f"{n:3} ep {r['episode']:3} {r['check']:22} {r['routed_oss']} -> {r['routed_oss_r']} "
              f"(failed {r['routed_oss_r_failed']}, tries {tries})", flush=True)
    for r in rows:
        if "routed_oss" in r and "routed_oss_r" not in r:
            r["routed_oss_r"], r["routed_oss_r_failed"] = r["routed_oss"], 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    httpx.post(f"{R.OLLAMA}/api/generate", json={"model": "gpt-oss:20b", "keep_alive": 0})


if __name__ == "__main__":
    main()
