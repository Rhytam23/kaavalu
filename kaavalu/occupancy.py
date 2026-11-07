"""Count-to-decision layer: occupancy % -> smoothed -> GREEN / AMBER / RED.

Occupancy % = count in zone / zone capacity x 100. In deployment the venue sets
capacity = zone area (m^2) x a safe density. Thresholds are configurable and
must be stated alongside any result.
"""

from collections import deque

GREEN, AMBER, RED = "GREEN", "AMBER", "RED"


def level_for(pct, amber_at=60.0, red_at=85.0):
    if pct > red_at:
        return RED
    if pct >= amber_at:
        return AMBER
    return GREEN


class OccupancyTracker:
    """Feed one count per frame; get the smoothed occupancy and level back.

    For a video, occupancy is averaged over ``smooth_s`` seconds of frames, and
    a fast rise (more than ``rise_points`` within ``rise_window_s``) raises an
    early AMBER even while the level is still GREEN. For separate images
    (``is_video=False``) every image is judged on its own, with no trend.
    """

    def __init__(self, capacity, fps=None, is_video=True, smooth_s=1.0,
                 amber_at=60.0, red_at=85.0, rise_points=20.0, rise_window_s=3.0):
        if capacity is None or capacity <= 0:
            raise ValueError("capacity must be a positive number")
        self.capacity = float(capacity)
        self.is_video = bool(is_video and fps)
        self.fps = float(fps) if fps else None
        self.amber_at, self.red_at = amber_at, red_at
        self.rise_points = rise_points
        n_smooth = max(1, int(round(smooth_s * self.fps))) if self.is_video else 1
        n_rise = max(1, int(round(rise_window_s * self.fps))) if self.is_video else 1
        self._window = deque(maxlen=n_smooth)
        self._history = deque(maxlen=n_rise + 1)

    def update(self, count):
        raw = 100.0 * count / self.capacity
        self._window.append(raw)
        smoothed = sum(self._window) / len(self._window)
        self._history.append(smoothed)
        level = level_for(smoothed, self.amber_at, self.red_at)
        rising = False
        if self.is_video and len(self._history) > 1:
            rising = (smoothed - min(self._history)) > self.rise_points
            if rising and level == GREEN:
                level = AMBER
        return {"occupancy_pct": raw, "smoothed_pct": smoothed, "level": level, "rising": rising}
