# Security and privacy

## What Kaavalu does and does not do

- Counts objects per frame. It does **not** identify, recognise or track anyone; there is no face
  analysis and no re-identification between frames.
- Runs fully offline on the device. Nothing is uploaded; there is no network code.
- Stores counts (`counts.csv`), box coordinates (`detections.json`) and annotated frames.
  Annotated frames contain the original imagery, so treat `results/` like the source footage.

## Handling footage

- Use footage you are allowed to use; record its source and licence in `data/practice/README.md`.
- Keep raw clips out of git (`.gitignore` excludes `data/practice/*.mp4` and `*.avi`).
- For a deployment, set `outputs.save_all_frames: false` so only counts, the timeline and the
  chosen sample images are kept, and delete samples after review.

## Supply chain

- Model weights come from torchvision's official `DEFAULT` COCO weights; the export metadata
  records the model name.
- Pin `executorch` (see `requirements-export.txt`) so every teammate exports with the same
  toolchain.
- Only load `.pte` files you exported yourselves: a `.pte` is a program for the runtime.

## Reporting a problem

Open a private issue or message the team directly with what you found, how to reproduce it and
which file or commit is affected. Please don't post footage or personal data in the report.
