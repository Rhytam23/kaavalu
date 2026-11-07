"""Our NumPy post-processing must match torchvision's on the same head outputs.

Skipped when torch / torchvision are not installed (e.g. a Pi with only the runtime).
Uses randomly initialised weights, so no download is needed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from kaavalu.decode import postprocess  # noqa: E402
from kaavalu.model_core import Core, compute_anchors  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    det = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone=None).eval()
    return det


def test_anchors_match_model(model):
    a = compute_anchors(model, 320)
    assert a.ndim == 2 and a.shape[1] == 4
    x = torch.randn(1, 3, 320, 320)
    cls, reg = Core(model)(x)
    assert cls.shape[1] == reg.shape[1] == a.shape[0]


def test_postprocess_matches_torchvision(model):
    torch.manual_seed(1)
    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        feats = list(model.backbone(x).values())
        out = model.head(feats)
        from torchvision.models.detection.image_list import ImageList
        anchors = model.anchor_generator(ImageList(x, [(320, 320)]), feats)
        # A freshly initialised head outputs ~0 (flat 1/91 scores), so feed both
        # post-processors the same synthetic logits and offsets of the real shape.
        head = {"cls_logits": torch.randn_like(out["cls_logits"]) * 3,
                "bbox_regression": torch.randn_like(out["bbox_regression"])}
        thr = 0.3
        model.score_thresh = thr
        ref = model.postprocess_detections(head, anchors, [(320, 320)])[0]

    ours = postprocess(head["cls_logits"].numpy(), head["bbox_regression"].numpy(),
                       anchors[0].numpy(), (320, 320), score_threshold=thr,
                       nms_threshold=model.nms_thresh, topk_candidates=model.topk_candidates,
                       max_detections=model.detections_per_img)
    assert len(ref["boxes"]) > 0
    assert len(ours["boxes"]) == len(ref["boxes"])
    order_ref = np.lexsort((ref["labels"].numpy(), -ref["scores"].numpy()))
    order_our = np.lexsort((ours["labels"], -ours["scores"]))
    np.testing.assert_allclose(ours["scores"][order_our], ref["scores"].numpy()[order_ref], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(ours["boxes"][order_our], ref["boxes"].numpy()[order_ref], rtol=1e-3, atol=1e-2)
