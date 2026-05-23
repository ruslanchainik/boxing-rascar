"""Extract ring-region 224x398 crops via ffmpeg (much faster than cv2).

Uses imageio-ffmpeg bundled binary, supports HEVC decode + h264 encode.
"""
from __future__ import annotations
import argparse, os, sys, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
import imageio_ffmpeg

sys.path.append(str(Path(__file__).resolve().parents[2]))
from boxing_lstm_pipeline.paths import POSE_DIR, TRAIN_VIDEOS, TEST_VIDEOS


FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=Path("C:/hack/artifacts/rgb_crops"))
    p.add_argument("--crop_w", type=int, default=398)
    p.add_argument("--crop_h", type=int, default=224)
    p.add_argument("--target_fps", type=int, default=15)
    p.add_argument("--margin", type=float, default=0.15)
    p.add_argument("--preset", default="ultrafast")
    p.add_argument("--crf", type=int, default=28)
    p.add_argument("--threads", type=int, default=0, help="ffmpeg -threads (0=auto)")
    return p.parse_args()


def video_path_for(row):
    return Path("C:/hack") / row["video_path"]


def estimate_ring_bbox(base: np.ndarray, width: int, height: int, margin: float):
    valid = base[:, 58] + base[:, 117]
    use = valid > 0
    if not use.any():
        return 0, 0, width, height
    red_cx = base[use, 51] * width; red_cy = base[use, 52] * height
    blue_cx = base[use, 110] * width; blue_cy = base[use, 111] * height
    all_x = np.concatenate([red_cx, blue_cx])
    all_y = np.concatenate([red_cy, blue_cy])
    cx_min, cx_max = float(all_x.min()), float(all_x.max())
    cy_min, cy_max = float(all_y.min()), float(all_y.max())
    bw = cx_max - cx_min; bh = cy_max - cy_min
    cx_min -= bw * margin; cx_max += bw * margin
    cy_min -= bh * margin; cy_max += bh * margin
    x1 = max(0, int(cx_min)); y1 = max(0, int(cy_min))
    x2 = min(width, int(cx_max)); y2 = min(height, int(cy_max))
    if x2 - x1 < 100 or y2 - y1 < 100:
        return 0, 0, width, height
    return x1, y1, x2, y2


def get_video_size(path: Path) -> tuple[int, int]:
    """Use ffprobe via ffmpeg to get width x height."""
    cmd = [FFMPEG, "-i", str(path)]
    # ffmpeg prints stream info to stderr; we parse it
    r = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, encoding="utf-8", errors="replace")
    for line in r.stderr.splitlines():
        if "Stream" in line and "Video" in line:
            # find "WxH" pattern
            for tok in line.replace(",", " ").split():
                if "x" in tok and tok.split("x")[0].isdigit():
                    w, h = tok.split("x")[:2]
                    if w.isdigit() and h.split()[0].isdigit():
                        try:
                            return int(w), int(h.split()[0])
                        except Exception:
                            pass
    return 1920, 1080  # default fallback


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.concat([pd.read_csv(TRAIN_VIDEOS), pd.read_csv(TEST_VIDEOS)], ignore_index=True)

    for _, row in manifest.iterrows():
        vk = row["video_key"]
        out_path = args.out_dir / f"{vk}.mp4"
        if out_path.exists():
            print(f"skip {vk}"); continue
        vp = video_path_for(row)
        if not vp.exists():
            print(f"missing {vp}"); continue
        npz_pose = POSE_DIR / f"{vk}.npz"
        if not npz_pose.exists():
            print(f"missing pose {vk}"); continue

        # Get video dims from pose npz where available (faster than ffprobe)
        pdata = np.load(npz_pose)
        width = int(pdata["width"]) if "width" in pdata.files else 1920
        height = int(pdata["height"]) if "height" in pdata.files else 1080
        base = pdata["base_features"].astype(np.float32)

        x1, y1, x2, y2 = estimate_ring_bbox(base, width, height, args.margin)
        cw = x2 - x1; ch = y2 - y1
        print(f"{vk}: source {width}x{height} crop=({x1},{y1}) {cw}x{ch}")

        # Build vf: crop=W:H:X:Y, scale=398:224, fps=15
        vf = f"crop={cw}:{ch}:{x1}:{y1},scale={args.crop_w}:{args.crop_h},fps={args.target_fps}"
        cmd = [
            FFMPEG, "-y",
            "-i", str(vp),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", args.preset,
            "-crf", str(args.crf),
            "-pix_fmt", "yuv420p",
            "-an",  # no audio
            "-threads", str(args.threads),
            str(out_path),
        ]
        t0 = time.time()
        r = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, encoding="utf-8", errors="replace")
        elapsed = time.time() - t0
        if r.returncode != 0:
            print(f"  FAILED {vk}: {r.stderr.splitlines()[-1] if r.stderr else 'unknown'}")
            continue
        sz_mb = out_path.stat().st_size / 1e6
        print(f"  saved {vk} ({elapsed:.1f}s, {sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
