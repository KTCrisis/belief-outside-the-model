# Preregistration, study 2: a reader routed by the kind of signal

Draft of 6 October 2026, written after the analysis of study 1 and before any episode of study 2. The literature review behind sections 4 to 6 is `docs/references/2026-10-06-reader-context-review.md`. Frozen by the commit that adds its final version; any later change is a **deviation**, logged at the end with its date and reason, and never replaces the original text.

## 1. Why a second study

Study 1 (`PREREGISTRATION.md`, OSF https://osf.io/e8ydh, under embargo until 1 October 2027; analysis `results/analysis-main.txt`, commit `1b36de9`) tested three hypotheses on 120 real faults of a Kafka lab:

- **H1 confirmed**: the language model's likelihood table corrected by counted incidents (mixed, empirical k) gives a lower loss than the model's table alone (19.11 against 24.80 minutes; difference −5.69, 98.33 % interval [−8.89, −2.72]).
- **H2 confirmed**: the priced stop beats the HESP controller's stop (19.73 against 23.10; −3.37 [−4.24, −2.52]).
- **H3 contradicted**: Sonnet alone (8.56, 120 correct out of 120, 2.64 dollars) beats the hybrid with the Jev reader (19.11; +10.55 [+6.00, +15.75]).

The exploratory analysis of study 1, done after the primary endpoints, located the gap in the **reading**, not in the decision:

| Hybrid (value of information, mixed empirical k) | Correct | Checks per episode | Mean loss |
| --- | --- | --- | --- |
| deterministic reference reader | 100 % | 3.3 | 9.33 |
| Haiku reader | | | 11.70 |
| Jev reader | 88 % | 4.0 | 19.11 |
| Sonnet alone (for comparison) | 100 % | 3.0 | 8.56 |

Readings on the wrong side of the reference, per check:

| Check | Kind of signal | Jev | Haiku |
| --- | --- | --- | --- |
| rebalance_logs | text | 0.0 % | 0.0 % |
| jvm_gc_logs | text | 0.0 % | 0.0 % |
| broker_quota_metrics | text and configuration | 0.0 % | 0.0 % |
| restart_consumer | two numbers | 0.0 % | 0.8 % |
| partition_lag_spread | numbers to compare | 3.3 % | 0.8 % |
| consumer_cpu_profile | numbers and a stack trace | 16.7 % | 41.7 % |
| producer_rate_graph | a number against a usual level | 24.2 % | 5.8 % |

Almost all the hybrid's extra loss comes from one cause, the upstream burst (hybrid correct in 42 % of its episodes): on outputs where the total rate was ten times the usual level, Jev returned probabilities near 0.5 (0.47 for 3,480 messages per second against a usual 350), so the check taught nothing, and the rule went on to other checks and often to a wrong cause. With the same rule and tables, a reader that turns each output into the right fact matches Sonnet alone within one minute.

A second finding qualifies the first. In study 1 every model reader was **narrow** by design: it received one output, what the check shows and one yes/no question, without the causes, their meaning or the other outputs (`read.py`; the choice came from the bench, where a narrow reading beat a reading inside a long conversation). Sonnet alone reads with the whole picture: the six causes and their meaning, the outputs already seen, and it decides what in an output matters. Part of the readers' "errors" came from the question, not from them: on consumer_cpu_profile during GC pauses, the three consumers run at 88 to 91 % CPU, two of them in `refreshCustomerCache` and one in `enrichOrder`; the question asks whether "the consumers saturate the CPU in their message processing"; Jev (20 of 20) and Haiku (19 of 20) answer yes, which is defensible, while the reference counts only `enrichOrder` and says no. Study 1 cannot tell how much of Sonnet's advantage comes from its size and how much from its context.

Study 1 thus suggests a hypothesis it could not test without bias, since the idea came from its data: **a reader routed by the kind of signal** (a judgment model on text, extraction then a stated comparison on numbers) brings the hybrid to the level of the large model alone. Study 2 tests it on new faults.

## 2. Setup (frozen)

- **Lab**: unchanged, `flux7-stack/tools/kafka-lab`, commit `0bcffcd`.
- **Runner**: unchanged, `bayes-action/kafka/episodes.py`; only the seed and the tag change.
- **Scenario**: unchanged, "kafka": 6 causes, 7 checks costing 1 to 8 minutes, M = 60 minutes per wrong diagnosis, λ = 12 minutes per bit.
- **Questions**: the seven questions of study 1 as amended by its deviation 2 (usual rate "about 350 messages per second"), with one change: consumer_cpu_profile becomes "at least two consumers are at 80 % CPU or more with their main thread in the message processing method `enrichOrder` (a cache refresh does not count)", the meaning the deterministic reference always had. The reference and the tables are unchanged.

## 3. New series

- **120 episodes**, 20 per cause, shuffled balanced blocks, **seed 2027**, tag `study2`: `python episodes.py --per-cause 20 --seed 2027 --tag study2`. No episode of study 1 or of its pilot is used for evaluation.
- **No cross-fitting.** The likelihood tables are counted on the 120 episodes of study 1 (read by the deterministic reference), and every episode of study 2 is evaluated with them. The empirical k is estimated on study 1.
- **Exclusion**: as in study 1 (failed reset, or lag at the alert below twice the healthy lag), reported per cause.

## 4. The routed reader (frozen before the series)

Each check is routed by what its question asks: a verdict that can be **read off** the text goes to the judgment model; a verdict that must be **derived** (a comparison of numbers) goes to extraction, then a stated comparison. The same boundary was reported independently for a decision-only judge (Li et al. 2026, JEV-as-a-Judge, arXiv:2609.26550: close to a frontier judge where the verdict can be read off, behind where it must be derived), and delegating computation to an executor is known to help on numeric questions (Chen et al., Program of Thoughts, arXiv:2211.12588). The classification, fixed now:

| Check | Verdict | Route |
| --- | --- | --- |
| rebalance_logs | read off | Jev, as in study 1 |
| jvm_gc_logs | read off | Jev, as in study 1 |
| broker_quota_metrics | read off | Jev, as in study 1 |
| restart_consumer | derived | extraction, then comparison |
| partition_lag_spread | derived | extraction, then comparison |
| producer_rate_graph | derived | extraction, then comparison |
| consumer_cpu_profile | derived | extraction, then comparison |

**Extraction.** `claude-haiku-4-5` with structured output, asked only to copy values from the output as written, never to judge; three extractions, each turned into a verdict by the comparison below, the median verdict kept. A value absent from the output is copied as −1. The values copied:

- restart_consumer: the total lag before the restart and 30 s after;
- partition_lag_spread: the lag of each partition;
- producer_rate_graph: the total incoming rate as written, and its unit (per second, per minute, or other); a rate per minute is divided by 60, any other unit is a failed extraction;
- consumer_cpu_profile: for each consumer, its CPU use as written, whether it is written as a percentage or a fraction (a fraction is multiplied by 100), and the first method of its main thread's stack.

**Comparison**, stated now, from the questions' own words and not tuned on study 1's errors:

- restart_consumer: the signal is present if the lag 30 s after is below the lag before;
- partition_lag_spread: present if the largest partition lag exceeds 3 times the median of the others;
- producer_rate_graph: present if the total rate is at least 1.5 times the usual level stated in the question (350, so 525 messages per second);
- consumer_cpu_profile: present if at least two consumers are at 80 % CPU or more with `enrichOrder` as the first method of their main thread's stack.

A comparison returns 0.98 when the signal is present and 0.02 when absent; an extraction that fails (no number, malformed answer) returns 0.5, counted and reported.

Note: on a lab whose formats are fixed, extraction then comparison is close to the deterministic reference, by construction; what it adds is that no parser is written per output. Section 7 measures that difference on purpose.

The code is committed, and its hash appended as an addendum, before any reading of the series.

## 5. Arms

All replayed offline on the 120 episodes, with the value-of-information rule (λ = 12), priced stop, and the mixed table with empirical k:

1. **routed reader** (section 4);
2. **Jev reader** (as in study 1);
3. **Haiku reader** (as in study 1, median of 5);
4. **deterministic reference reader** (the ceiling of a perfect reader on these formats);
5. **Sonnet alone**, same harness and settings as study 1 (addendum 3 and deviations 1 and 2 of study 1), one run per episode;
6. **Sonnet as a narrow reader**: `claude-sonnet-5-5`, Haiku's prompt of study 1 unchanged, median of 3 answers; separates the effect of size from the effect of context;
7. **Haiku as a reader with context**: Haiku's prompt preceded by the situation part of Sonnet's brief: the alert (group 'lab' lagging on topic 'orders'), the usual rate (about 350 messages per second), the six causes and what each means, the seven checks and what they show; without the costs, the price of an error or the goal, which concern the decision and not the reading; median of 5 answers; one output at a time, without the other outputs, so that the reading stays independent of the path. If it closes the gap, a cheap reader suffices. Risk stated now: a reader that knows the causes may read an output in the light of a cause it suspects, so that its reading is no longer conditional on the output alone, as the likelihood table assumes; the agreement of arm 7 with the reference is reported per cause, and section 6 measures the pull directly. Arm 7 also names, in a separate field after its probability, the cause it finds most likely from this output alone;
8. **Haiku as a sequential reader** (after Linear Sequential Unmasking, Dror et al. 2015, doi:10.1111/1556-4029.12805; LSU-E, Dror and Kukucka 2021, doi:10.1016/j.fsisyn.2021.100161): in one conversation, Haiku first answers the narrow question of arm 3 and its answer is recorded; then it is shown the situation of arm 7 and asked a **leading** question: "Given this situation, do you revise your probability that the output shows the signal? Give your final probability", and which cause it finds most likely from this output. Median of 5 conversations. Both readings are kept: the first is blind, the second informed. The hybrid is replayed with the second. This separates what the context clarifies (an ambiguous question, as consumer_cpu_profile in study 1) from what it contaminates;
9. **Haiku as a sequential reader, neutral question**: arm 8 with the second question worded "Now that you know the situation, give your probability that the output shows the signal again. It may stay the same if the situation does not change what the output shows."

**Why two sequential arms (observed on the pilot, excluded from every analysis).** On the six pilot episodes, the blind readings of arm 8 were all on the right side of the reference (0 of 42 wrong). After the leading question, 7 of 42 were wrong, 5 of them on partition_lag_spread: on outputs whose lag was spread evenly over the six partitions (for instance 1,350 to 2,193 messages per partition), the reading went from 0.15 to 0.78 and the reader named partition_skew, once that cause was on the table. Arm 7, which receives the same situation in a single turn, was wrong on 1 of 42. The difference could come from the context or from the wording "do you revise", which invites a change (a demand effect, close to the user-driven sycophancy described by Marozzo and Liò 2026, arXiv:2606.13220). With the neutral wording, still on the pilot, 5 of 42 were wrong after revision, 3 of them on partition_lag_spread. Arm 8 measures what happens when the reader is pushed to reconsider, arm 9 what the context does when it is not; the gap between arms 8 and 9, and between arm 9 and arm 7, are reported. Six episodes decide nothing: these figures only explain why both arms exist.

## 6. Hypotheses and criteria

Two primary endpoints, each with a 97.5 % interval (Bonferroni for two), 10,000 paired bootstrap resamples over the kept episodes:

- **H4 (routed against Jev)**: the routed reader gives a lower mean loss than the Jev reader. Confirmed if the interval of the difference (routed − Jev) lies entirely below zero.
- **H5 (hybrid not worse than the model alone)**: the hybrid with the routed reader is not worse than Sonnet alone by more than **2 minutes** per episode. Confirmed if the upper bound of the interval of the difference (hybrid − Sonnet) is below +2. If the whole interval also lies below zero, the hybrid is reported as better.

The margin of 2 minutes is a third of the cost of the dearest check but one, and under the gap between the reference reader and Sonnet in study 1 plus its noise.

Stated now: a null or a reversed effect of context is plausible. In the closest test found, telling a model the examiner's hypothesis lowered its wrong readings, from 36.1 % to 19.6 % of trials (Trivedi 2026, J Cybersecurity, Digital Forensics and Jurisprudence 2:85-99); and exact Bayesian updating over misspecified likelihoods can lose to a model's holistic judgment (Chen et al. 2026, arXiv:2605.06915), which may also explain study 1. These outcomes would be reported as they come.

Secondary endpoints, exploratory, uncorrected:

- **pull toward the suspected cause** (arms 7, 8 and 9): for each output, the shift between the informed and the blind reading, signed by what the likelihood table expects under the cause the reader named: shift × (2 × P(signal | named cause) − 1), the blind reading being arm 3's for arm 7, and the first reading of the same conversation for arms 8 and 9. Computed separately on the outputs where the reference says the signal is present and on those where it says absent, so that a correct reading is not mistaken for a pull; a positive mean within these groups is the signature of double counting (the "corroboration inflation" of Kassin, Dror and Kukucka 2013, doi:10.1016/j.jarmac.2013.01.001); reported with a 95 % bootstrap interval, per check and per named cause;
- **calibration of the final belief**, not only accuracy: Brier score and log score of the hybrid's posterior on the true cause, for every reader arm;
- **reading quality apart from final accuracy** (as Cloud-OpsBench separates evidence closure from answer accuracy, Wang et al. 2026, arXiv:2603.00468): for every reader arm, the Brier score against the reference on the checks the hybrid actually ran; for Sonnet alone, whether its conclusion is **supported** by what it saw: its own path replayed with the reference reader and the study 1 tables gives its concluded cause a posterior of at least 0.8;

- size against context: losses of arms 3, 6 and 7 against each other and against arm 5 (narrow Haiku, narrow Sonnet, Haiku with context, Sonnet alone);
- dollars per diagnosis of each arm, counting for the hybrid only the readings of the checks it actually ran, and for Sonnet its tokens;
- per check, readings on the wrong side and Brier score of the routed, Jev and Haiku readers against the reference;
- share of failed extractions;
- share of correct diagnoses (HESP's measure) for every arm;
- whether study 1's findings repeat on the new series: H1 and H2 as stated in study 1, with tables counted on study 1.

A null or contrary result is reported as such.

## 7. Format shift (exploratory)

The claim behind the routed reader is that it survives a change of format where a hand-written parser does not. A **shifted copy** of the 120 outputs is made by a script committed before any reading, with three changes:

- rates given in messages per minute instead of per second, with the usual level in the question converted the same way;
- CPU given as a fraction (0.93) instead of a percentage;
- the columns of the lag table reordered, and headers renamed.

The deterministic reference (unchanged) and the routed reader (unchanged, questions converted) read the shifted copy; the hybrid's loss with each is reported. Exploratory, no correction.

## 8. Expected costs

Trial: all readers and the shifted copy were tried on the six pilot episodes (excluded from every analysis), for 1.54 dollars.

Lab: about 9 hours of local compute, at night. Extraction: 120 × 4 checks × 3 answers, under one dollar. Jev: under one dollar. Haiku reader: 2 to 4 dollars. Sonnet alone: about 3 dollars (2.64 in study 1). Sonnet as a narrow reader: 120 × 7 × 3 reads, about 5 dollars. Haiku with context: about 3 dollars. Haiku sequential, two arms: about 6 dollars. Shifted copy: about one more dollar.

## Addendum (planned in section 4)

6 October 2026, before any episode of study 2: the readers are `kafka/read2.py`, the shifted copy `kafka/shift.py`, the analysis `kafka/analyze2.py`, all at commit `dc56363` (SHA-256 prefixes `3fa07c4dbf87740e`, `eda230813193ed52`, `c6e251d4a415d4a8`). Sonnet alone runs with `kafka/sonnet.py` unchanged from study 1. If any of these files changes before the first episode, this addendum is rewritten with the new commit; after it, any change is a deviation.

## Registration

To be registered on OSF as a new registration of project https://osf.io/e8ydh, citing study 1, before the first episode of study 2.

Registered on OSF on 6 October 2026, about 22:50 CEST (OSF Preregistration template), under embargo until 1 October 2027, with the PDF of this file at commit 70c9465 attached. Link and DOI: to be added.

**Errata in the registration form**, noted on 6 October 2026, the same evening, before any episode of study 2. The form only summarizes this protocol; where they differ, this protocol governs. Two fields were submitted wrong and could not be edited afterwards:

- *Missing data* holds a placeholder, "[rule from study 1's registration]", instead of a rule. The rule is the one the frozen analysis code applies: a failed run of Sonnet alone (a refusal included, study 1 addendum 3) has no loss, and the paired comparison of H5 uses the kept episodes where Sonnet's run did not fail (`analyze2.py`, `s_loss` and `ok`). A failed extraction in the routed reader returns 0.5, counted and reported (section 4). Nothing is imputed.
- *Inference criteria* says that H4's interval is "one-sided in direction". The interval is two-sided: a 97.5 % percentile interval from 10,000 paired bootstrap resamples (`analyze2.py`, `boot`); only the criterion is directional (the whole interval below zero), as section 6 states.

## Deviations

None yet.
