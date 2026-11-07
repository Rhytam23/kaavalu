"""Detector wrapper with two interchangeable backends.

* ``executorch``: loads a .pte written by export/export_detector.py (the Pi path).
* ``torch``: the same network in plain PyTorch eager mode (baseline for benchmarks
  and for laptop testing without a .pte).

Both return raw head outputs; decoding, NMS and rescaling happen here in NumPy,
so the two backends are compared on exactly the same post-processing.
"""

import json
import os
import time

import cv2
import numpy as np

from .decode import postprocess


def load_meta(model_path):
    meta_path = os.path.splitext(model_path)[0] + ".json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"missing {meta_path}: export/export_detector.py writes it next to the .pte")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    anchors_path = os.path.join(os.path.dirname(model_path), meta["anchors"])
    meta["anchors_array"] = np.load(anchors_path)
    return meta


def set_threads(backend, n):
    """Best-effort CPU thread control. Returns True if the setting was applied."""
    if not n:
        return False
    if backend == "torch":
        import torch
        torch.set_num_threads(int(n))
        return True
    try:  # ExecuTorch's XNNPACK threadpool (present in recent pybindings builds)
        from executorch.extension.pybindings import portable_lib
        portable_lib._unsafe_reset_threadpool(int(n))
        return True
    except Exception:
        return False


class _ExecuTorchModel:
    def __init__(self, path):
        from executorch.runtime import Runtime
        self._method = Runtime.get().load_program(path).load_method("forward")

    def __call__(self, x):
        import torch
        out = self._method.execute([torch.from_numpy(x)])
        return np.asarray(out[0]), np.asarray(out[1])


class _TorchModel:
    def __init__(self, name):
        import torch
        from .model_core import Core, load_torchvision
        self._torch = torch
        self.det = load_torchvision(name)
        self._core = Core(self.det).eval()

    def __call__(self, x):
        with self._torch.no_grad():
            cls, reg = self._core(self._torch.from_numpy(x))
        return cls.numpy(), reg.numpy()


class Detector:
    """detect(BGR image) -> {'boxes' (image pixels), 'scores', 'labels'}."""

    def __init__(self, model="models/det_int8.pte", backend=None, size=320,
                 score_threshold=0.4, class_filter=None, nms_threshold=0.55, threads=None,
                 model_name="ssdlite320_mobilenet_v3_large"):
        if backend is None:
            backend = "executorch" if str(model).endswith(".pte") else "torch"
        self.backend = backend
        self.score_threshold = score_threshold
        self.class_filter = None if class_filter is None else sorted(class_filter)
        self.nms_threshold = nms_threshold
        self.threads_applied = set_threads(backend, threads)

        if backend == "executorch":
            meta = load_meta(model)
            self.meta = meta
            self.size = int(meta["size"])
            self.mean = np.asarray(meta["mean"], dtype=np.float32)
            self.std = np.asarray(meta["std"], dtype=np.float32)
            self.anchors = meta["anchors_array"]
            self._model = _ExecuTorchModel(model)
            self.description = f"{meta['model']} ({meta['precision']}, ExecuTorch/XNNPACK, {self.size}px)"
        elif backend == "torch":
            from .model_core import compute_anchors, normalisation
            self._model = _TorchModel(model_name)
            mean, std = normalisation(self._model.det)
            self.size = int(size)
            self.mean = np.asarray(mean, dtype=np.float32)
            self.std = np.asarray(std, dtype=np.float32)
            self.anchors = compute_anchors(self._model.det, self.size)
            self.meta = {"model": model_name, "precision": "fp32", "size": self.size}
            self.description = f"{model_name} (fp32, PyTorch eager, {self.size}px)"
        else:
            raise ValueError(f"unknown backend {backend!r}")

        self.model_ms = 0.0   # time inside the network for the last detect() call
        self.total_model_ms = 0.0
        self.runs = 0         # total network runs so far

    def preprocess(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        x = rgb.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    def raw(self, x):
        t0 = time.perf_counter()
        cls, reg = self._model(x)
        self.model_ms = (time.perf_counter() - t0) * 1000.0
        self.total_model_ms += self.model_ms
        self.runs += 1
        return cls, reg

    def detect(self, image_bgr):
        h, w = image_bgr.shape[:2]
        cls, reg = self.raw(self.preprocess(image_bgr))
        dets = postprocess(cls, reg, self.anchors, (self.size, self.size),
                           score_threshold=self.score_threshold,
                           class_filter=self.class_filter,
                           nms_threshold=self.nms_threshold)
        if len(dets["boxes"]):
            scale = np.array([w / self.size, h / self.size, w / self.size, h / self.size],
                             dtype=np.float32)
            dets["boxes"] = dets["boxes"] * scale
        return dets
