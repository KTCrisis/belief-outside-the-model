# Belief Outside the Model

Code, recorded outputs and analysis outputs for the paper *Belief Outside the Model: Two Preregistered
Studies of a Hybrid Diagnosis Agent on Real Kafka Faults* (Marc Verchiani, 2026).

Preprint: https://doi.org/10.5281/zenodo.23414699 (all versions).

Preregistrations: study 1 https://osf.io/hdmq9, study 2 https://osf.io/8dq6m, OSF project https://osf.io/e8ydh.

## Layout

| Folder | Content |
| --- | --- |
| `kafka-lab/` | the laboratory: one Kafka 4.0 broker, one producer, three consumers, six injected faults, seven checks (Docker Compose) |
| `kafka/` | the two studies: episode collection, readers, Sonnet alone, analyses, preregistrations |
| `kafka/results/` | recorded episodes, readings, LLM runs and every analysis output quoted in the paper |
| `exp2/` | elicitation of the likelihood tables by an LLM (`tables.py`) and the Jev client; the other scripts are earlier experiments not used in the paper (see `exp2/README.md`) |
| root | the simulated bench of the companion note (`sim.py`, `exact.py`, `value_fit.py`, `m_sweep.py`, `lambda_check.py`) and the figures (`figures.py`) |

`kafka-lab/` is a copy of the laboratory at the commit named in the preregistrations (`0bcffcd`); it has not changed
since. The only edit to the code of the studies is the default path to the laboratory in `kafka/episodes.py`;
`figures.py` lost a figure unrelated to the paper.

## Replaying the analyses

Every analysis runs offline from the recorded files, without calling a model. With Python 3.12, numpy and scipy:

```sh
cd kafka
python analyze.py --tag main          # study 1, preregistered analysis   -> results/analysis-main.txt
python analyze2.py --tag study2       # study 2, preregistered analysis   -> results/analysis-study2.txt
python secondary.py                   # secondary results (Appendix D)    -> results/secondary.txt
python robustness.py                  # reader prior, order of the checks -> results/robustness.txt
python sonnet_repeats.py              # repeated runs of Sonnet alone     -> results/sonnet-repeats.txt
python lambda_fit.py                  # setting lambda (Appendix A)       -> results/lambda-fit.txt
python optimal_replay.py              # rule against the exact optimum    -> results/optimal-replay.txt
python sensitivity.py                 # prices M and lambda (Appendix B)  -> results/sensitivity.txt
python cost_sensitivity.py            # measured check durations          -> results/cost-sensitivity.txt
python paper_numbers.py               # remaining figures of the paper    -> results/paper-numbers.txt
python viz_export.py                  # data of figure 7                  -> results/viz-export.json
cd .. && python figures.py --paper    # figures of the paper (also needs matplotlib) -> figures/
```

Each script prints to standard output; redirect it to the file shown to compare with the recorded version.
The bootstraps use a fixed seed, so the outputs are identical.

## Collecting new episodes

This part needs Docker and API access, and costs money.

```sh
docker compose -p stack-kafka-lab -f kafka-lab/compose.yaml up -d
cd kafka
python episodes.py --per-cause 1 --tag pilot       # inject faults, record every check
python read.py --tag pilot --readers reference     # study 1 readers
python read2.py --tag pilot --readers all          # study 2 readers
python sonnet.py --tag pilot                       # Sonnet alone
```

`sonnet.py` and the Haiku readers need `ANTHROPIC_API_KEY`. The Jev reader calls the `typesafe/jev` model on
Cloudflare Workers AI and needs `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_WORKERS_AI_TOKEN`; its readings of both
studies are recorded, so the analyses do not need it.

`kafka/results/run-notes.md` logs what else ran on the workstation during data collection.

## Citation

```bibtex
@misc{verchiani2026belief,
  title     = {Belief Outside the Model: Two Preregistered Studies of a Hybrid Diagnosis Agent on Real Kafka Faults},
  author    = {Verchiani, Marc},
  year      = {2026},
  month     = oct,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23414699},
  url       = {https://doi.org/10.5281/zenodo.23414699},
  note      = {Preprint}
}
```

The DOI above resolves to the latest version of the preprint; version 3.7 is
https://doi.org/10.5281/zenodo.23414700. `CITATION.cff` carries the same reference.

## License

MIT, see `LICENSE`.
