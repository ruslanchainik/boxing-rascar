"""Tiny labeling tool for boxing punches.

Hotkeys (focus on opencv window):
  Space          play / pause
  → / ←          step +1 / -1 frame
  Shift→/Shift← step +10 / -10 frames
  . / ,          step +30 / -30 frames (1 sec at 30fps)
  ] / [          step +120 / -120 frames (~4 sec)
  Home / End     jump to start / end of video

  ENTER          add punch at current frame (using last attrs, edit in console)
  Q              quick-add punch at current frame with last attrs (no edit)
  R              red, B = blue (for the next add)
  J/C/H/U        jab/cross/hook/uppercut for next add
  L/W            left/right hand for next add (W = right since R is red)
  G/Y            head (G=go to head)/body (Y=body)
  V/X/M          landed/blocked/miss
  K              toggle clear=true/false

  N              delete last row
  S              save CSV
  Esc            save and quit

CSV columns: video_key,frame,fighter,punch_type,hand,target,effectiveness,clear
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import cv2
import numpy as np


VAL_FIGHTER = ["red", "blue"]
VAL_TYPE = ["jab", "cross", "hook", "uppercut"]
VAL_HAND = ["left", "right"]
VAL_TARGET = ["head", "body"]
VAL_EFF = ["landed", "blocked", "miss"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--video_key", default=None)
    p.add_argument("--display_w", type=int, default=1280)
    return p.parse_args()


def load_existing(out_path: Path) -> list[dict]:
    if not out_path.exists():
        return []
    rows = []
    with out_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            r["frame"] = int(r["frame"])
            rows.append(r)
    return rows


def save_csv(out_path: Path, rows: list[dict]):
    rows = sorted(rows, key=lambda r: r["frame"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_key","frame","fighter","punch_type","hand","target","effectiveness","clear"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"saved {len(rows)} rows to {out_path}")


def main():
    args = parse_args()
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"cannot open {args.video}"); sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video: {args.video.name} | {n_frames} frames | fps={fps:.2f} | {width}x{height}")

    vk = args.video_key or args.video.stem
    rows = load_existing(args.out)
    print(f"loaded {len(rows)} existing rows")

    # state for next punch
    last = {"fighter":"red","punch_type":"jab","hand":"left","target":"head","effectiveness":"landed","clear":"true"}

    cur_frame = 0
    playing = False
    display_w = args.display_w
    scale = display_w / width
    display_h = int(height * scale)
    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("label", display_w, display_h + 200)

    def goto(f):
        nonlocal cur_frame
        cur_frame = max(0, min(n_frames - 1, f))
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)

    def add(edit_console=True):
        if edit_console:
            print(f"\nadd at frame {cur_frame}. press Enter to keep [shown], or type new value:")
            r = {"video_key": vk, "frame": cur_frame}
            for key, opts in [("fighter",VAL_FIGHTER),("punch_type",VAL_TYPE),("hand",VAL_HAND),("target",VAL_TARGET),("effectiveness",VAL_EFF)]:
                v = input(f"  {key} [{last[key]}] (options: {','.join(opts)}): ").strip()
                if not v: v = last[key]
                if v not in opts:
                    print(f"  invalid '{v}', keeping {last[key]}"); v = last[key]
                r[key] = v; last[key] = v
            v = input(f"  clear [{last['clear']}] (true/false): ").strip()
            if not v: v = last["clear"]
            r["clear"] = v if v in ("true","false") else last["clear"]
            last["clear"] = r["clear"]
        else:
            r = {"video_key": vk, "frame": cur_frame, **last}
        rows.append(r)
        print(f"+ row #{len(rows)}: frame={r['frame']} {r['fighter']} {r['punch_type']} {r['hand']} {r['target']} {r['effectiveness']} clear={r['clear']}")

    # Pre-read first frame
    goto(0)
    frame_cache = {}
    while True:
        if cur_frame not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
            ok, fr = cap.read()
            if not ok: fr = np.zeros((height, width, 3), np.uint8)
            frame_cache[cur_frame] = fr
            if len(frame_cache) > 60: frame_cache.pop(next(iter(frame_cache)))
        fr = frame_cache[cur_frame].copy()
        if scale != 1.0:
            fr = cv2.resize(fr, (display_w, display_h))

        # overlay
        canvas = np.zeros((display_h + 200, display_w, 3), np.uint8)
        canvas[:display_h] = fr
        info = [
            f"frame {cur_frame}/{n_frames-1}  fps={fps:.1f}  t={cur_frame/max(fps,1):.2f}s",
            f"rows: {len(rows)}  | next: {last['fighter']} {last['punch_type']} {last['hand']} {last['target']} {last['effectiveness']} clear={last['clear']}",
            "ENTER=add(edit)  Q=quick-add  arrows=step  Shift=10  ,/.=30  [/]=120",
            "R/B=red/blue  J/C/H/U=type  L/W=l/r hand  G/Y=head/body  V/X/M=land/blk/miss",
            "K=toggle clear  N=delete last  S=save  Esc=save+exit",
        ]
        for i, line in enumerate(info):
            cv2.putText(canvas, line, (10, display_h + 25 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,255,200), 1, cv2.LINE_AA)
        # mark if there is a labeled punch within ±15 frames
        near = [r for r in rows if abs(r["frame"] - cur_frame) <= 15]
        if near:
            cv2.rectangle(canvas, (0,0), (display_w-1, display_h-1), (0,200,255), 4)
            for j, r in enumerate(near[:3]):
                cv2.putText(canvas, f"f{r['frame']}: {r['fighter']} {r['punch_type']}", (10, 30 + j*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2, cv2.LINE_AA)
        cv2.imshow("label", canvas)
        delay = max(1, int(1000 / max(fps, 1))) if playing else 30
        key = cv2.waitKeyEx(delay)
        if key == -1:
            if playing:
                goto(cur_frame + 1)
            continue
        kc = key & 0xFF
        # special keys (arrows on win use scan codes via waitKeyEx)
        if key == 27:  # Esc
            save_csv(args.out, rows); break
        elif kc == ord(' '):
            playing = not playing
        elif key == 2555904 or kc == ord('d'):  # right arrow
            goto(cur_frame + 1)
        elif key == 2424832 or kc == ord('a'):  # left arrow
            goto(cur_frame - 1)
        elif key == 2228224 or kc == ord('s') and False:  # down (unused)
            pass
        elif key == 39:  # apostrophe ('): shift-right shortcut
            goto(cur_frame + 10)
        elif key == 59:  # ';' shift-left shortcut
            goto(cur_frame - 10)
        elif kc == ord('.'):
            goto(cur_frame + 30)
        elif kc == ord(','):
            goto(cur_frame - 30)
        elif kc == ord(']'):
            goto(cur_frame + 120)
        elif kc == ord('['):
            goto(cur_frame - 120)
        elif kc == ord('\r') or kc == 13:
            add(edit_console=True)
        elif kc == ord('q'):
            add(edit_console=False)
        # quick attribute change for "next add"
        elif kc == ord('r'): last["fighter"] = "red"
        elif kc == ord('b'): last["fighter"] = "blue"
        elif kc == ord('j'): last["punch_type"] = "jab"
        elif kc == ord('c'): last["punch_type"] = "cross"
        elif kc == ord('h'): last["punch_type"] = "hook"
        elif kc == ord('u'): last["punch_type"] = "uppercut"
        elif kc == ord('l'): last["hand"] = "left"
        elif kc == ord('w'): last["hand"] = "right"
        elif kc == ord('g'): last["target"] = "head"
        elif kc == ord('y'): last["target"] = "body"
        elif kc == ord('v'): last["effectiveness"] = "landed"
        elif kc == ord('x'): last["effectiveness"] = "blocked"
        elif kc == ord('m'): last["effectiveness"] = "miss"
        elif kc == ord('k'): last["clear"] = "false" if last["clear"]=="true" else "true"
        elif kc == ord('n'):
            if rows:
                rm = rows.pop()
                print(f"- removed frame={rm['frame']}")
        elif kc == ord('S') or kc == ord('s'):
            save_csv(args.out, rows)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
