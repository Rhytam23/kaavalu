# Working on this repo

Read this before changing code, whether you are a teammate or a coding tool.

## Commands

```bash
bash scripts/test.sh                 # all tests (torch parity tests skip without torch)
python run.py --input <clip> --out results_dev
python export/export_detector.py --calib data/practice --out models --verify data/practice
```

## Rules

- **Approved model only.** The core detector stays the organisers' approved pretrained model
  with unchanged weights. Tiling reuses it; no new models in the core path.
- **Keep the Pi path torch-free at the edges.** Decoding, NMS, tiling, counting and occupancy
  are NumPy only. `torch` is imported only inside `model_core.py`, the `torch` backend and the
  ExecuTorch input conversion.
- **Match torchvision.** Any change to `kaavalu/decode.py` must keep
  `tests/test_torch_parity.py` passing.
- **Every number traces to a file.** Results go to `results*/` as JSON or CSV; charts are drawn
  from those files by `tools/charts.py`, never typed by hand.
- **Don't tune on the organiser clip.** Thresholds and tiling settings are chosen on
  `data/practice/` and then applied once.
- **No scope creep.** No tracking, dashboards, cloud, apps or face analysis (see the brief).
- **Config over code.** New behaviour gets a `config.yaml` key with the current behaviour as the
  default, and a command-line flag only if it is changed often.

## Style

- Python 3.10+, standard library + NumPy + OpenCV in `kaavalu/`.
- Short modules with one job each; docstrings say why, not what.
- Add a test in `tests/test_core.py` for any new NumPy logic.
