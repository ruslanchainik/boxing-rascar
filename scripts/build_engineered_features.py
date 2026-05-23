"""Read existing pose_features npz files, append engineered features, save to new dir."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from boxing_lstm_pipeline.engineered_features import compute_engineered, ENG_DIM
from boxing_lstm_pipeline.paths import POSE_DIR


SRC = POSE_DIR
DST = Path("C:/hack/artifacts/pose_features_eng")


def main():
    DST.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*.npz"))
    print(f"[eng] {len(files)} files to process")
    for p in files:
        out_path = DST / p.name
        if out_path.exists():
            print(f"  skip {p.name}")
            continue
        data = np.load(p)
        base = data["base_features"].astype(np.float32)
        feats = data["features"].astype(np.float32)
        eng = compute_engineered(base)
        # Extended features = original 426 + 38 engineered = 464
        feats_ext = np.concatenate([feats, eng], axis=1).astype(np.float32)
        save_kwargs = {
            "frames": data["frames"],
            "features": feats_ext,
            "base_features": base,
            "engineered_features": eng,
            "width": data["width"],
            "height": data["height"],
            "fps": data["fps"],
        }
        np.savez_compressed(out_path, **save_kwargs)
        print(f"  saved {out_path.name} feat_dim={feats_ext.shape[1]} (was {feats.shape[1]}, +{eng.shape[1]})")


if __name__ == "__main__":
    main()
