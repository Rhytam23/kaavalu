"""Export the detector core to ExecuTorch (.pte), in FP32 and INT8. Run on the laptop / Colab.

    python export/export_detector.py --calib data/practice --out models
    python export/export_detector.py --calib data/practice --out models --verify data/practice

Writes, for each precision:
    models/det_fp32.pte, models/det_fp32.json
    models/det_int8.pte, models/det_int8.json
    models/anchors_320.npy
The .json metadata records model name, input size, normalisation and the anchors
file, so the Pi never needs torchvision.

--verify compares .pte + decode.py against the full torchvision model on 5 frames.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.export import export

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kaavalu.frames import iter_frames  # noqa: E402
from kaavalu.model_core import Core, compute_anchors, load_torchvision, normalisation  # noqa: E402


def _quant_api():
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config)
    try:  # ExecuTorch >= 0.7 takes the PT2E flow from torchao
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    except ImportError:  # older releases
        from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
    return XNNPACKQuantizer, get_symmetric_quantization_config, prepare_pt2e, convert_pt2e


def lower(module, example, path):
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    prog = to_edge_transform_and_lower(export(module, example),
                                       partitioner=[XnnpackPartitioner()]).to_executorch()
    with open(path, "wb") as f:
        f.write(prog.buffer)
    return os.path.getsize(path)


def preprocess(image_bgr, size, mean, std):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = (rgb.astype(np.float32) / 255.0 - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))


def calibration_tensors(paths, n, size, mean, std):
    """Collect up to n frames spread across the given videos / image folders."""
    frames = []
    for p in paths:
        for _, _, img in iter_frames(p):
            frames.append(img)
    if not frames:
        raise SystemExit("no calibration frames found; pass --calib with real street / crowd footage")
    step = max(1, len(frames) // n)
    picked = frames[::step][:n]
    print(f"calibration: {len(picked)} frames (from {len(frames)} available)")
    return [preprocess(f, size, mean, std) for f in picked]


def write_meta(out_dir, stem, model_name, size, mean, std, anchors_file, precision, size_bytes, extra=None):
    meta = {
        "model": model_name, "weights": "torchvision DEFAULT (COCO)", "size": size,
        "mean": mean, "std": std, "anchors": anchors_file, "precision": precision,
        "outputs": ["cls_logits", "bbox_regression"], "num_classes": 91,
        "file_bytes": size_bytes, "box_weights": [10.0, 10.0, 5.0, 5.0],
    }
    if extra:
        meta.update(extra)
    with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def verify(det, model_paths, frames_src, n=5, threshold=0.3):
    """Compare our .pte + NumPy post-processing with the full torchvision model.

    Uses n frames spread across the source. Frames where the reference model finds
    nothing prove nothing, so the check fails if fewer than 5 reference boxes in total.
    """
    from kaavalu.decode import box_iou
    from kaavalu.detector import Detector

    every = [img for _, _, img in iter_frames(frames_src)]
    step = max(1, len(every) // n)
    frames = every[::step][:n]
    ok_all = True
    ref_total = 0
    for path in model_paths:
        ours = Detector(path, score_threshold=threshold)
        stats = {"ref": 0, "got": 0, "matched": 0}
        print(f"\nverify {os.path.basename(path)}")
        for i, img in enumerate(frames):
            # The full model does its own resize + normalisation and returns
            # boxes in original-image pixels.
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            with torch.no_grad():
                ref = det([t])[0]
            keep = ref["scores"] > threshold
            ref_boxes = ref["boxes"][keep].numpy()
            got = ours.detect(img)
            n_ref, n_got = len(ref_boxes), len(got["boxes"])
            ref_total += n_ref
            matched = 0
            if n_ref and n_got:
                iou = box_iou(ref_boxes.astype(np.float32), got["boxes"])
                matched = int((iou.max(axis=1) >= 0.7).sum())
            frac = matched / n_ref if n_ref else 1.0
            print(f"  frame {i}: torchvision {n_ref} boxes, ours {n_got}, matched {matched} ({frac:.0%})")
            stats["ref"] += n_ref
            stats["got"] += n_got
            stats["matched"] += matched
            # FP32 must reproduce the reference. INT8 drift is expected and is
            # measured as count error against the hand count (tools/evaluate.py).
            if ours.meta.get("precision") == "fp32":
                ok_all &= frac >= 0.95 and abs(n_ref - n_got) <= max(1, int(0.1 * n_ref))
        print(f"  total: torchvision {stats['ref']}, ours {stats['got']}, "
              f"matched {stats['matched']} ({stats['matched'] / max(1, stats['ref']):.0%})"
              + ("" if ours.meta.get("precision") == "fp32" else "  [INT8: reported, not pass/fail]"))
    if ref_total < 5 * len(model_paths):
        print(f"\nVERIFY INCONCLUSIVE: the reference model found only {ref_total} boxes above "
              f"{threshold}; use frames with clearly visible objects")
        return False
    print("\nVERIFY (FP32 vs torchvision)", "PASSED" if ok_all else "FAILED (see per-frame numbers above)")
    return ok_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ssdlite320_mobilenet_v3_large")
    ap.add_argument("--size", type=int, default=320, help="square input size")
    ap.add_argument("--out", default="models")
    ap.add_argument("--calib", nargs="*", default=[], help="videos / image folders for INT8 calibration")
    ap.add_argument("--calib-frames", type=int, default=100)
    ap.add_argument("--skip-int8", action="store_true")
    ap.add_argument("--per-tensor", dest="per_channel", action="store_false",
                    help="per-tensor INT8 weights (default: per-channel, more accurate)")
    ap.add_argument("--verify", default=None, help="video / image folder: check boxes on 5 frames")
    ap.add_argument("--verify-threshold", type=float, default=0.3)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    det = load_torchvision(args.model)
    core = Core(det).eval()
    mean, std = normalisation(det)
    size = args.size
    example = (torch.randn(1, 3, size, size),)
    suffix = "" if size == 320 else f"_{size}"

    anchors = compute_anchors(det, size)
    anchors_file = f"anchors_{size}.npy"
    np.save(os.path.join(args.out, anchors_file), anchors)
    print(f"anchors: {anchors.shape[0]} -> {anchors_file}")

    written = []
    fp32 = os.path.join(args.out, f"det_fp32{suffix}.pte")
    nbytes = lower(core, example, fp32)
    write_meta(args.out, f"det_fp32{suffix}", args.model, size, mean, std, anchors_file, "fp32", nbytes)
    print(f"wrote {fp32} ({nbytes / 1e6:.1f} MB)")
    written.append(fp32)

    if not args.skip_int8:
        if not args.calib:
            raise SystemExit("INT8 needs --calib (about 100 real frames); or pass --skip-int8")
        XNNPACKQuantizer, sym_config, prepare_pt2e, convert_pt2e = _quant_api()
        # Per-channel weights: per-tensor scales hurt MobileNet's depthwise convs badly.
        quantizer = XNNPACKQuantizer().set_global(sym_config(is_per_channel=args.per_channel))
        gm = prepare_pt2e(export(core, example).module(), quantizer)
        with torch.no_grad():
            for x in calibration_tensors(args.calib, args.calib_frames, size, mean, std):
                gm(x)
        qm = convert_pt2e(gm)
        int8 = os.path.join(args.out, f"det_int8{suffix}.pte")
        nbytes = lower(qm, example, int8)
        write_meta(args.out, f"det_int8{suffix}", args.model, size, mean, std, anchors_file, "int8", nbytes,
                   {"calibration_frames": args.calib_frames,
                    "calibration_sources": [os.path.basename(os.path.normpath(p)) for p in args.calib],
                    "per_channel": args.per_channel})
        print(f"wrote {int8} ({nbytes / 1e6:.1f} MB)")
        written.append(int8)

    if args.verify:
        sys.exit(0 if verify(det, written, args.verify, threshold=args.verify_threshold) else 1)


if __name__ == "__main__":
    main()
