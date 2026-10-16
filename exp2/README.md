# exp2

Earlier simulated experiments of the same research, run before the two studies. The paper uses only two files of
this folder:

- `tables.py`: elicitation of the likelihood tables by an LLM (`results/tables-kafka.json`, `results/tables-summary.txt`);
- `jev_reader.py`: the client of the Jev reader.

`tables.py` imports `exp2.py` and `memory_lab.py`, so they stay here. The other scripts (`rumor.py`, `drift.py`,
`censoring.py`, `tree.py`) and their results belong to other studies and are not discussed in the paper.
