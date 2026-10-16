# Run notes, main series

Factual log of what else ran on the workstation during data collection (16 cores, 43 GB, one GPU).

- 2026-10-05 ~21:50 to ~22:20 CEST: other agent runs (Ollama gpt-oss:20b, 12 GB on the GPU, ~1 core for the runner) and a model download, during episodes ~18 to ~30. Episode durations (264 to 277 s) and lag profiles per cause unchanged compared with episodes 1 to 17.
- 2026-10-05 ~22:20 CEST: a grid of other agent runs (gpt-oss:20b on the GPU, 4 one-day runs in parallel, light CPU) during episodes ~27 to ~32.
- 2026-10-05 ~22:38 CEST: other agent runs with Mistral Small (aligned 3.2, then base 3.1), one run at a time on the GPU.
- 2026-10-05 ~23:20 CEST onward: a series of other agent runs (Claude Haiku over the API; gpt-oss:20b then Mistral Small 3.2 on the GPU, one run at a time, Mistral spills ~2 GB to the CPU).
- 2026-10-06 ~01:40 to ~05:40 CEST (episodes ~69 to 119, all in fold B): the healthy lag measured before each injection rose from 30-300 to 500-1,500, episode durations from ~270 s to ~290-300 s. No scheduled job and no other agent run on the machine after 01:40 (the Mistral runs ended then); cause unknown. At 06:58 the healthy lag was back to 298 with the consumers keeping up. Every fault still took (no episode meets the exclusion rule). To report, with an exploratory fold-by-fold check, since the shift falls inside fold B.
- 2026-10-06 ~07:00 CEST: the main series ended at 05:42 (120/120), before a scheduled GPU job at 06:00.
