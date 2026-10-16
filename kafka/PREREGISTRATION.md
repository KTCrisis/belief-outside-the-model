# Preregistration: the investigation agent on real Kafka faults

Written on 5 October 2026, before the first episode of the main series. Everything below is frozen by the commit that adds this file. Any later change is a **deviation**, logged at the end of this file with its date and reason; it never replaces the original text.

## 1. Purpose

Measure, on real faults of a Kafka platform and real check outputs, what the simulations showed:

- the language model's likelihood table, corrected by closed incidents, beats the model's table alone;
- stopping at the price of an error beats stopping at a confidence threshold;
- a narrow judgment model (Jev) reads raw outputs correctly.

## 2. Setup (frozen)

- **Lab**: `flux7-stack/tools/kafka-lab`, commit `0bcffcd`. One Kafka 4.0 broker, one producer, three Java consumers; six injected faults; seven checks whose raw output is recorded; full reset between episodes (topic and group recreated).
- **Runner**: `bayes-action/kafka/episodes.py`, commit `1d575b5`. 130 s healthy, injection, 60 s wait, then all seven checks, in a fixed order.
- **Pilot**: `results/episodes-pilot.jsonl`, six episodes, a technical check. **Excluded from every analysis.** It served to check the pipeline and showed: the slow consumer has no signature of its own (saturated CPU, normal incoming rate); restarting a consumer never clears the lag, since the fault stays injected.
- **Scenario**: `scenarios.py`, "kafka": 6 causes, 7 checks with their cost in minutes (1 to 8), M = 60 minutes per wrong diagnosis.

## 3. Main series

- **120 episodes**, 20 per cause, in shuffled balanced blocks (seed 2026), tag `main`: `python episodes.py --per-cause 20 --tag main`.
- **Two-fold cross-fitting**: fold A = blocks 1 to 10, fold B = blocks 11 to 20 (60 episodes each, 10 per cause). Tables counted on fold A are evaluated on fold B, and tables counted on fold B on fold A. Every episode is evaluated exactly once, by a table that never saw it, so all endpoints use the 120 episodes.
- **Exclusion** of an episode, and only if: the reset failed, or the lag at the alert (last measurement before the checks) is below twice the healthy lag (the fault did not take). The number of exclusions is reported, per cause.

## 4. Reading the outputs

Each raw output is read through one yes/no question that describes a present fact. The questions are frozen here; the code that asks them will be committed, and its hash appended as an addendum, before any reading.

| Check | Question: is the signal present? |
| --- | --- |
| rebalance_logs | the consumers joined, left or rebalanced the group within the window |
| partition_lag_spread | the lag is concentrated on one or two partitions rather than spread |
| producer_rate_graph | the incoming rate is clearly above its usual level (about 300 messages/s) |
| broker_quota_metrics | a client quota is set, or the consumers suffer a notable throttle time |
| jvm_gc_logs | repeated full GC pauses caused by a lack of memory appear |
| consumer_cpu_profile | the consumers saturate the CPU in their message processing |
| restart_consumer | the total lag decreases after the restart |

Three readers:

1. **Deterministic reference** (written before the data, on the pilot outputs only): at least one "joined", "Revoke" or "rejoining" line; maximum partition lag > 3 × the median of the others; total rate ≥ 600 messages/s; a quota present or a mean throttle > 100 ms; at least 3 "Pause Full (Allocation Failure)" lines; at least 2 consumers at ≥ 80 % CPU with the main thread in `enrichOrder`; lag at +30 s below the lag before the restart.
2. **Jev** (Cloudflare Workers AI, `typesafe/jev`): the probability it returns.
3. **Haiku** (`claude-haiku-4-5`): median of 5 answers, as in experiment 2.

## 5. Likelihood tables compared

| Table | Source |
| --- | --- |
| designer | the hand-written table in `scenarios.py`, written before the lab existed |
| LLM | the 10 medians elicited on 04/10 (`exp2/results/tables-kafka.json`) |
| counted | the other fold's 60 episodes, read by the deterministic reference, Beta(1, 1) |
| mixed, k = 10 | the LLM median as prior, weight 10, corrected by the other fold |
| mixed, empirical k | the same, k estimated by empirical Bayes on the other fold (`tables.ebayes_strength`) |

The prior over causes is uniform (1/6), as in the series (20 per cause).

## 6. Rules compared

All replayed offline on the 120 episodes: the rule picks a check, sees its output as read, updates its belief, and continues or concludes. Checks it did not pick do not exist for it.

- value of information, λ = 12 min/bit (= M/5, fixed on 04/10);
- information per cost ranking, priced stop (λ = 12);
- **HESP controller, as published** (Liu and Wang 2026, arXiv:2609.33446, sections 4.1 to 4.4): uniform prior over the six causes plus an explicit `other`; likelihood tables counted on the other fold without any language model, smoothed with ε = 0.01; checks ranked by expected information gain per cost; the episode ends when the leading cause reaches a posterior of 0.8 **and** a current observation supports it (raised its score), or when no check is left. Its tables are its own (counted only, no language-model prior), as in the paper;
- myopic Bayes decision (game A);
- confidence-threshold heuristics at 0.9 (cheapest first, most informative).

The default reader is Jev; the deterministic reference and Haiku are replayed as variants.

**Language model alone.** `claude-sonnet-5-5`, replayed offline on the 120 episodes: it receives the same brief as the hybrid (the six causes and what each means, the seven checks with their cost and what they show, the 60-minute price of an error), asks for one check at a time and receives that check's recorded raw output, then concludes. At most seven checks; one run per episode; no table, no memory of other episodes. Its loss is computed exactly as the hybrid's.

**Loss** of an episode = minutes of the checks run + 60 if the concluded cause is wrong.

## 7. Hypotheses and criteria

Three primary endpoints, each with a 98.33 % interval (Bonferroni correction for three), 10,000 bootstrap resamples over the 120 episodes, paired:

- **H1 (table)**: with the value-of-information rule and the Jev reader, the **mixed, empirical k** table gives a lower mean loss than the **LLM table** (mean over the 10 medians). Confirmed if the interval of the difference lies entirely below zero.
- **H2 (stop)**: with the same tables (counted on the other fold, ε = 0.01) and the same ranking (information per cost), the **priced stop** (λ = 12) gives a lower mean loss than the **HESP controller's stop**. Same criterion.
- **H3 (hybrid vs language model alone)**: the hybrid (mixed, empirical k table; Jev reader; value-of-information rule with priced stop) gives a lower mean loss than **Sonnet alone**. Same criterion.

Secondary endpoints, exploratory, uncorrected:

- agreement of Jev and Haiku with the deterministic reference, per check (Brier score and share of readings on the wrong side);
- **prediction on Jev**, stated now: Jev agrees better with the reference on the textual checks (rebalance_logs, jvm_gc_logs, consumer_cpu_profile) than on the numeric ones (partition_lag_spread, producer_rate_graph, broker_quota_metrics, restart_consumer), since TypeSafe documents weak counting and arithmetic. Measured as the mean Brier score of each group;
- Sonnet alone: checks run, accuracy, tokens and dollars, compared with the hybrid;
- losses of every table and every rule; loss of the counted table alone;
- comparison of the counted table with the designer table (which hand-written values reality contradicts);
- token and dollar cost of reading;
- HESP's own measure next to ours: the share of episodes ending on the correct cause, for every rule, so that neither metric favours one method by construction.

A null or contrary result is reported as such.

## 8. Outside this series

Drift in the history, asymmetric error costs, simultaneous causes: planned separately, each with its own preregistration.

## 9. Expected costs

Lab: about 9 hours of local compute. Jev reading: under one dollar. Haiku reading: 120 × 7 × 5 reads, in the order of 2 to 4 dollars. Sonnet alone: in the order of 12 to 18 dollars by an earlier estimate; current pricing is checked, and the spend approved, before the run.

## Addendum (planned in section 4)

5 October 2026, before any reading of the main series: the reader is `kafka/read.py`, commit `4e45c50`, SHA-256 prefix `b683f554c8a151d9`. The main series was launched at 20:21 the same evening, after the commit of this preregistration (`9968002`).

## Addendum 2: informative censoring (exploratory)

5 October 2026, 20:55 CEST, before any reading of the main series, written into the OSF registration at the same time. Tables are also counted from a **self-chosen history**: for each episode of the counting fold, in block order, only the checks that the value-of-information rule (λ = 12) would have run are kept, the rule using at each episode the table counted so far (mixed with the LLM median at k = 10, and separately with no prior, Beta(1, 1)); instead of all seven checks. These tables are compared with the ones counted from the complete history: losses on the evaluation fold (same rule, Jev reader), and per-cell differences. Exploratory, no correction. In simulation, a self-chosen history without a language-model prior never ran a check and never learned.

Survival-type likelihoods (time since a signal was last seen, system age as prior) cannot be measured on this series (one check at a fixed time, system reset at each episode) and are planned with real incidents.

## Addendum 3: settings of the language model alone (before any call)

5 October 2026, 21:20 CEST, before any Sonnet call and any reading of the main series. `claude-sonnet-5-5` runs with its default settings: adaptive thinking (the `thinking` parameter omitted), default effort, no fallback model (a refusal is a failed run, handled as in the missing-data rule), `max_tokens` 16,000. Two tools, `run_check` (one of the seven checks; the recorded raw output is returned) and `conclude` (one of the six causes); tool choice left to the model. A check already run returns a note instead of an output. A run that ends without `conclude` is prompted once to conclude; if it still does not, the episode counts as a wrong diagnosis. The harness is tried on the six pilot episodes first (excluded from every analysis).

## Registration

Registered on OSF on 5 October 2026 (OSF Preregistration template), under embargo until 1 October 2027. Associated project: https://osf.io/e8ydh. The registration attaches the PDF of this file at commit 9d9aced. DOI: to be added once issued.

## Deviations

1. 5 October 2026, 21:30 CEST, from the Sonnet pilot (6 pilot episodes, excluded from every analysis), before any main-series call: the brief given to Sonnet alone now states the usual incoming rate ("about 300 messages per second"). The reading questions already give this fact to Jev and Haiku (section 4); without it Sonnet could not tell a burst from normal traffic, and concluded "slow_consumer" on the pilot's burst episode after reading 3,481 messages/s. The pilot's first Sonnet run is kept as `results/sonnet-pilot-v0.jsonl`.
2. 5 October 2026, 21:45 CEST, from the pilot readings (excluded from every analysis), before any reading of the main series: the usual incoming rate stated in the producer_rate_graph question (section 4) and in Sonnet's brief is changed from "about 300" to "about 350 messages per second", the healthy rate actually measured in the lab (348 to 358 messages/s on the pilot; the producer's target of 300 overshoots). With "300", Jev and Haiku read 5 of the 6 pilot outputs as above normal (wrong side 83 %); with "350", 1 of 6 for Jev and 0 for Haiku. The deterministic reference (threshold 600 messages/s) is unchanged. Earlier pilot readings and Sonnet runs are kept as `results/readings-pilot-v0.jsonl` and `results/sonnet-pilot-v1.jsonl`.
