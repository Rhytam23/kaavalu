"""COCO category names in torchvision's 91-index layout (index 0 = background).

torchvision detection models output label indices into this list; the "N/A"
entries are COCO ids that were never used.
"""

COCO_CATEGORIES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table",
    "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

assert len(COCO_CATEGORIES) == 91


def class_ids(names):
    """Map class names (e.g. ["person", "car"]) to torchvision label indices."""
    ids = {}
    for name in names:
        key = name.strip().lower()
        if key not in COCO_CATEGORIES or key == "n/a" or key == "__background__":
            raise ValueError(f"unknown COCO class {name!r}")
        ids[COCO_CATEGORIES.index(key)] = key
    return ids
