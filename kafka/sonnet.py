"""Claude Sonnet alone on the real-fault episodes (preregistration, H3 and addendum 3).

The model gets the same brief as the hybrid (the six causes and what each
means, the seven checks with their cost and what they show, the 60-minute
price of a wrong diagnosis), asks for one check at a time and receives that
check's recorded raw output, then concludes. Default settings: adaptive
thinking (parameter omitted), default effort, no fallback model.

usage: python sonnet.py --tag pilot      # 6 episodes, about 0.5 $
       python sonnet.py --tag main       # 120 episodes
       python sonnet.py --tag study2 --run 2   # a repeat of Sonnet alone, exploratory
       python sonnet.py --tag study2 --model claude-haiku-5-5   # exploratory arm, not preregistered
       python sonnet.py --tag study2 --model ollama:gpt-oss:20b  # local, through Ollama (run at night)

A local model gets the same brief and the same two tools through Ollama's /api/chat, with its default
settings. Its tool calls are not schema-constrained as Claude's are: a call outside the two tools or their
values is answered with a reminder and counted (`invalid`).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import anthropic
import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from read import QUESTIONS  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402

OUT = HERE / "results"
MODEL = "claude-sonnet-5-5"
# dollars per token: Sonnet from the documentation of 2026-09-25; Haiku 5.5 at its rate for prompts of 100K
# tokens or fewer, which every turn here is
PRICES = {"claude-sonnet-5-5": (2.0 / 1e6, 10.0 / 1e6), "claude-haiku-5-5": (0.10 / 1e6, 0.50 / 1e6),
          "claude-haiku-4-5": (1.0 / 1e6, 5.0 / 1e6)}
PRICE_IN, PRICE_OUT = PRICES[MODEL]
OLLAMA = "http://localhost:11434"
SC = SCENARIOS["kafka"]
CAUSES = SC["causes"]
COST = {t[0]: t[1] for t in SC["tests"]}
CHECKS = [t[0] for t in SC["tests"]]
MISS = SC["miss"]

client = anthropic.Anthropic()


def price(model: str) -> tuple[float, float]:
    return (0.0, 0.0) if model.startswith("ollama:") else PRICES[model]

TOOLS = [
    {"name": "run_check", "description": "Run one of the available checks and read its raw output.",
     "strict": True,
     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "enum": CHECKS}},
                      "required": ["name"], "additionalProperties": False}},
    {"name": "conclude", "description": "State the cause you act on. This ends the diagnosis.",
     "strict": True,
     "input_schema": {"type": "object", "properties": {"cause": {"type": "string", "enum": CAUSES}},
                      "required": ["cause"], "additionalProperties": False}},
]


def brief() -> str:
    causes = "\n".join(f"- {c}: {SC['meaning'][c]}" for c in CAUSES)
    checks = "\n".join(f"- {t} ({COST[t]:.0f} min): shows {QUESTIONS[t][0]}" for t in CHECKS)
    return (
        "You are the on-call engineer for a Kafka platform. An alert reports that the consumer group "
        "'lab' is lagging behind on topic 'orders'. In normal operation the order service sends about "
        "350 messages per second.\n\n"
        f"Possible causes, exactly one is true:\n{causes}\n\n"
        f"Available checks, each can be run once:\n{checks}\n\n"
        f"Acting on a wrong cause costs {MISS:.0f} minutes. Your goal is to minimise the expected total "
        "cost: the minutes spent on checks plus the cost of a wrong conclusion. Run a check only when it "
        "is worth its cost, and conclude as soon as further checks are not worth it. "
        "Use exactly one tool per turn."
    )


def run_episode(ep: dict) -> dict:
    system = brief()
    messages = [{"role": "user", "content": "The alert has just fired. Start the diagnosis."}]
    tin = tout = 0
    minutes, path, answer, nudged, stop, said = 0.0, [], None, False, None, []
    for _ in range(2 * len(CHECKS) + 6):
        r = client.messages.create(
            model=MODEL, max_tokens=16000, system=system, tools=TOOLS, messages=messages,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        )
        tin += r.usage.input_tokens
        tout += r.usage.output_tokens
        stop = r.stop_reason
        if r.stop_reason == "refusal":
            break
        messages.append({"role": "assistant", "content": r.content})
        text = " ".join(b.text for b in r.content if b.type == "text").strip()
        if text:
            said.append(text)
        calls = [b for b in r.content if b.type == "tool_use"]
        if not calls:
            if nudged:
                break
            nudged = True
            messages.append({"role": "user", "content": "Please use run_check or conclude."})
            continue
        call = calls[0]
        if call.name == "conclude":
            answer = call.input["cause"]
            break
        name = call.input["name"]
        if name in path:
            result = "Already run earlier; its output is above."
        else:
            path.append(name)
            minutes += COST[name]
            result = ep["probes"][name] or "(empty output)"
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": call.id, "content": result}]})
    right = answer == ep["cause"]
    return {"episode": ep["episode"], "cause": ep["cause"], "answer": answer, "right": right,
            "path": path, "minutes": minutes, "loss": minutes + (0.0 if right else MISS),
            "stop_reason": stop, "tin": tin, "tout": tout,
            "dollars": round(tin * PRICE_IN + tout * PRICE_OUT, 5), "said": said}


def run_episode_ollama(ep: dict) -> dict:
    """The same episode through Ollama: same brief, same tools, the model's default settings."""
    tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": {k: v for k, v in t["input_schema"].items()
                                                              if k != "additionalProperties"}}} for t in TOOLS]
    messages = [{"role": "system", "content": brief()},
                {"role": "user", "content": "The alert has just fired. Start the diagnosis."}]
    tin = tout = invalid = 0
    minutes, path, answer, nudged, stop, said = 0.0, [], None, False, None, []
    for _ in range(2 * len(CHECKS) + 6):
        r = httpx.post(f"{OLLAMA}/api/chat", json={"model": MODEL.removeprefix("ollama:"), "messages": messages,
                                                   "tools": tools, "stream": False}, timeout=600).json()
        tin += r.get("prompt_eval_count", 0)
        tout += r.get("eval_count", 0)
        stop = r.get("done_reason")
        msg = r["message"]
        messages.append({k: msg[k] for k in ("role", "content", "tool_calls") if k in msg})
        if msg.get("content", "").strip():
            said.append(msg["content"].strip())
        calls = msg.get("tool_calls") or []
        if not calls:
            if nudged:
                break
            nudged = True
            messages.append({"role": "user", "content": "Please use run_check or conclude."})
            continue
        fn = calls[0]["function"]
        name, args = fn.get("name"), fn.get("arguments") or {}
        if name == "conclude" and args.get("cause") in CAUSES:
            answer = args["cause"]
            break
        if name != "run_check" or args.get("name") not in CHECKS:
            invalid += 1
            messages.append({"role": "tool", "tool_name": name or "", "content":
                             f"Invalid call. Use run_check with one of {CHECKS}, or conclude with one of {CAUSES}."})
            continue
        check = args["name"]
        if check in path:
            result = "Already run earlier; its output is above."
        else:
            path.append(check)
            minutes += COST[check]
            result = ep["probes"][check] or "(empty output)"
        messages.append({"role": "tool", "tool_name": "run_check", "content": result})
    right = answer == ep["cause"]
    return {"episode": ep["episode"], "cause": ep["cause"], "answer": answer, "right": right,
            "path": path, "minutes": minutes, "loss": minutes + (0.0 if right else MISS),
            "stop_reason": stop, "tin": tin, "tout": tout, "invalid": invalid, "dollars": 0.0, "said": said}


def main() -> None:
    global MODEL, PRICE_IN, PRICE_OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default=MODEL, help=f"one of {sorted(PRICES)}, or ollama:<model>")
    ap.add_argument("--limit", type=int, default=None, help="first N episodes only (a pilot)")
    ap.add_argument("--run", type=int, default=None,
                    help="repeat N on the same episodes, written beside the preregistered file (exploratory)")
    a = ap.parse_args()
    MODEL = a.model
    PRICE_IN, PRICE_OUT = price(MODEL)
    episode = run_episode_ollama if MODEL.startswith("ollama:") else run_episode
    episodes = [json.loads(l) for l in (OUT / f"episodes-{a.tag}.jsonl").open()][:a.limit]
    # Sonnet keeps its preregistered file name; another model writes beside it
    name = MODEL.replace(":", "-").replace("/", "-")
    path = OUT / (f"sonnet-{a.tag}.jsonl" if MODEL == "claude-sonnet-5-5" else f"{name}-{a.tag}.jsonl")
    if a.run is not None:
        path = path.with_name(f"{path.stem}-run{a.run}.jsonl")
    done = {json.loads(l)["episode"] for l in path.open()} if path.exists() else set()
    total = 0.0
    with path.open("a") as f:
        for ep in episodes:
            if ep["episode"] in done:
                continue
            for attempt in range(3):
                try:
                    rec = episode(ep)
                    break
                except anthropic.APIStatusError as e:
                    if e.status_code < 500 and e.status_code != 429:
                        raise
                    time.sleep(10 * (attempt + 1))
                except (anthropic.APIConnectionError, httpx.TransportError):
                    time.sleep(10 * (attempt + 1))
            else:
                rec = {"episode": ep["episode"], "cause": ep["cause"], "failed": True}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            total += rec.get("dollars", 0)
            print(f"{ep['episode']:>3} {ep['cause']:18} -> {rec.get('answer')!s:18} "
                  f"{'ok ' if rec.get('right') else 'BAD'} {rec.get('path')} {rec.get('dollars')} $", flush=True)
    print(f"total {total:.3f} $")


if __name__ == "__main__":
    main()
