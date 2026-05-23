"""Combine engineered pose features with glove tracks → final feature_dim."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

ENG_DIR = Path("C:/hack/artifacts/pose_features_eng")  # has 464-dim features
GLOVE_DIR = Path("C:/hack/artifacts/glove_tracks")    # has 20-dim per frame
OUT_DIR = Path("C:/hack/artifacts/pose_features_glove")


def derive(arr, lag=1):
    out = np.zeros_like(arr)
    if lag < len(arr):
        out[lag:] = arr[lag:] - arr[:-lag]
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(ENG_DIR.glob("*.npz"))
    print(f"processing {len(files)} files")
    for p in files:
        vk = p.stem
        glove_path = GLOVE_DIR / f"{vk}.npz"
        if not glove_path.exists():
            print(f"missing glove {vk}, skip")
            continue
        out_path = OUT_DIR / p.name
        if out_path.exists():
            print(f"skip {vk}")
            continue
        eng = np.load(p)
        features_eng = eng["features"].astype(np.float32)   # (T, 464)
        glove = np.load(glove_path)["glove"].astype(np.float32)  # (T, 20)

        T = min(len(features_eng), len(glove))
        features_eng = features_eng[:T]
        glove = glove[:T]

        # Compute additional features from glove tracks:
        # For each of 4 wrists (5 features each): cx, cy, conf, size, has
        # Add: velocity magnitude, acceleration magnitude → 2 per wrist = 8 extra
        # Plus "is glove inside opponent bbox" — would need opp bbox, skip for now
        extra = []
        for w in range(4):
            cx = glove[:, w * 5 + 0]
            cy = glove[:, w * 5 + 1]
            xy = np.stack([cx, cy], axis=1)
            vel = derive(xy)
            acc = derive(vel)
            vmag = np.linalg.norm(vel, axis=1)
            amag = np.linalg.norm(acc, axis=1)
            extra.append(vmag)
            extra.append(amag)
        extra = np.stack(extra, axis=1).astype(np.float32)  # (T, 8)

        # Concat: orig eng (464) + glove (20) + derived (8) = 492
        new_features = np.concatenate([features_eng, glove, extra], axis=1).astype(np.float32)

        save_kwargs = dict(eng)
        save_kwargs["features"] = new_features
        save_kwargs["glove_raw"] = glove
        save_kwargs["glove_derived"] = extra
        np.savez_compressed(out_path, **save_kwargs)
        print(f"saved {out_path.name}: {features_eng.shape[1]} + {glove.shape[1]} + {extra.shape[1]} = {new_features.shape[1]}")


if __name__ == "__main__":
    main()
