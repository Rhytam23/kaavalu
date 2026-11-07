"""Hand-count tool: click every target object in each frame.

    python tools/handcount.py --input organiser_clip.mp4 --out data/organiser_handcount.csv --every 1

Use the same --input and --every as run.py so frame numbers line up.
Split the work on the day with --start / --stop (e.g. three people, one third each),
then join the CSVs with tools/evaluate.py (it accepts several --truth files).

Keys:  left click = add point   right click = remove nearest point
       u = undo   n or space = next frame   b = previous frame
       s = save   q / Esc = save and quit
The CSV is saved after every frame, so a crash loses at most one frame.
"""

import argparse
import csv
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kaavalu.frames import iter_frames  # noqa: E402

MAX_W, MAX_H = 1280, 800


def load(path):
    data = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pts = []
                if row.get("points"):
                    for p in row["points"].split(";"):
                        x, y = p.split(":")
                        pts.append((float(x), float(y)))
                data[int(row["frame"])] = {"name": row.get("name", ""), "points": pts}
    return data


def save(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "name", "count", "points"])
        for idx in sorted(data):
            pts = data[idx]["points"]
            w.writerow([idx, data[idx]["name"], len(pts), ";".join(f"{x:.1f}:{y:.1f}" for x, y in pts)])
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True, help="CSV to write (resumes if it exists)")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--start", type=int, default=0, help="first position in the frame list (0-based)")
    ap.add_argument("--stop", type=int, default=None, help="stop before this position")
    ap.add_argument("--label", default="person", help="shown in the window title only")
    args = ap.parse_args()

    frames = list(iter_frames(args.input, every=args.every))[args.start:args.stop]
    if not frames:
        raise SystemExit("no frames in that range")
    data = load(args.out)
    pos = 0
    win = "Kaavalu hand count"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    state = {"scale": 1.0, "points": []}

    def on_mouse(event, x, y, flags, param):
        s = state["scale"]
        if event == cv2.EVENT_LBUTTONDOWN:
            state["points"].append((x / s, y / s))
        elif event == cv2.EVENT_RBUTTONDOWN and state["points"]:
            ox, oy = x / s, y / s
            j = min(range(len(state["points"])),
                    key=lambda k: (state["points"][k][0] - ox) ** 2 + (state["points"][k][1] - oy) ** 2)
            state["points"].pop(j)

    cv2.setMouseCallback(win, on_mouse)

    def commit():
        idx, name, _ = frames[pos]
        data[idx] = {"name": name, "points": list(state["points"])}
        save(args.out, data)

    def load_pos():
        idx = frames[pos][0]
        state["points"] = list(data.get(idx, {}).get("points", []))

    load_pos()
    while True:
        idx, name, img = frames[pos]
        h, w = img.shape[:2]
        s = min(1.0, MAX_W / w, MAX_H / h)
        state["scale"] = s
        view = cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img.copy()
        for x, y in state["points"]:
            cv2.circle(view, (int(x * s), int(y * s)), 5, (0, 0, 255), -1)
            cv2.circle(view, (int(x * s), int(y * s)), 6, (255, 255, 255), 1)
        done = sum(1 for f in frames if f[0] in data)
        text = f"{name}  [{pos + 1}/{len(frames)}]  {args.label}: {len(state['points'])}  (done {done})"
        cv2.rectangle(view, (0, 0), (view.shape[1], 30), (30, 30, 30), -1)
        cv2.putText(view, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("n"), ord(" ")):
            commit()
            if pos < len(frames) - 1:
                pos += 1
                load_pos()
            else:
                print("last frame reached; press q to finish")
        elif key == ord("b"):
            commit()
            pos = max(0, pos - 1)
            load_pos()
        elif key == ord("u") and state["points"]:
            state["points"].pop()
        elif key == ord("s"):
            commit()
        elif key in (ord("q"), 27):
            commit()
            break
    cv2.destroyAllWindows()
    print(f"saved {args.out}: {sum(1 for f in frames if f[0] in data)} of {len(frames)} frames counted")


if __name__ == "__main__":
    main()
