# Contributing

## Roles

| Role | Owns |
| --- | --- |
| Edge lead | Pi setup, export (FP32 / INT8), runtime, decode + NMS, benchmark, power and temperature |
| Accuracy lead | Practice clips, hand counts, evaluation, threshold sweep, tiling, charts |
| Product / pitch lead | Occupancy layer, annotated outputs, README, deck, pitch, rehearsal |

## Workflow

1. Branch from `main`: `git switch -c <area>/<short-name>`.
2. Run `bash scripts/test.sh` before every push.
3. If a change moves a number (latency, count error), rerun the affected tool and say in the
   pull request which result file changed and by how much.
4. Merge only when tests pass on a Pi as well as a laptop for anything in `kaavalu/` or `run.py`.

## Phases

A phase is done only when its check passes on the Pi.

| Phase | Done when |
| --- | --- |
| P0 Toolchain | a `.pte` runs on the Pi and prints its latency |
| P1 Core | per-frame counts, CSV, annotated frames and timing on a practice clip |
| P2 Practice data | a hand-count CSV for each practice clip |
| P3 Accuracy | before/after tiling error table and the chosen threshold |
| P4 Efficiency | `benchmark_report.json` with temperature and energy |
| P5 Decision layer | occupancy timeline for each practice clip |
| P6 Package | fresh Pi: clone, `bash scripts/setup_pi.sh`, one command, all outputs |

**Freeze rule:** once P5 works, no new features. Fixes, rehearsal and results only.

## Data

- Practice clips go in `data/practice/` and are not committed (see `.gitignore`); list their
  source and licence in `data/practice/README.md`.
- Hand-count CSVs are committed: they are small and they are the evidence.
