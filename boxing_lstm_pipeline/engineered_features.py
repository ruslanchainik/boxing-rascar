"""Compute engineered features from existing base pose features.

Base layout (142 dim per frame):
  [red_role(59)][blue_role(59)][pair(24)]
Each role: kpts(51) + bbox(4) + colors(3) + presence(1)
Each kpt: x, y, conf (17 kpts COCO format).
Coords are in [0, 1] (normalized by frame width/height).
"""
from __future__ import annotations
import numpy as np

ROLE_DIM = 59

# Per-fighter offsets to keypoint coords (x, y) in base features
def kpt_xy(kpt_idx: int, offset: int) -> tuple[int, int]:
    return offset + kpt_idx * 3, offset + kpt_idx * 3 + 1

NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12


def _safe_deriv(x: np.ndarray, lag: int = 1) -> np.ndarray:
    out = np.zeros_like(x)
    if lag < len(x):
        out[lag:] = x[lag:] - x[:-lag]
    return out


def _elbow_cos(sh, el, wr):
    v1 = el - sh
    v2 = wr - el
    n1 = np.linalg.norm(v1, axis=1) + 1e-6
    n2 = np.linalg.norm(v2, axis=1) + 1e-6
    return (v1 * v2).sum(1) / (n1 * n2)


def _fighter_features(base: np.ndarray, offset: int) -> list[np.ndarray]:
    """14 engineered features per fighter."""
    T = base.shape[0]
    # gather kpt xy slices
    def xy(idx):
        sx, sy = kpt_xy(idx, offset)
        return base[:, sx:sy + 1]  # shape (T,2)
    lw = xy(L_WRIST); rw = xy(R_WRIST)
    ls = xy(L_SHOULDER); rs = xy(R_SHOULDER)
    le = xy(L_ELBOW); re = xy(R_ELBOW)
    lh = xy(L_HIP); rh = xy(R_HIP)
    nose = xy(NOSE)
    chest = (ls + rs) / 2
    # Velocities
    lw_v = _safe_deriv(lw); rw_v = _safe_deriv(rw)
    lw_vmag = np.linalg.norm(lw_v, axis=1)
    rw_vmag = np.linalg.norm(rw_v, axis=1)
    # Accelerations
    lw_a = _safe_deriv(lw_v); rw_a = _safe_deriv(rw_v)
    lw_amag = np.linalg.norm(lw_a, axis=1)
    rw_amag = np.linalg.norm(rw_a, axis=1)
    # Extension from chest
    lw_ext = np.linalg.norm(lw - chest, axis=1)
    rw_ext = np.linalg.norm(rw - chest, axis=1)
    # Hand crosses centerline (mid-shoulder x)
    cx = chest[:, 0]
    lw_cross = (lw[:, 0] > cx).astype(np.float32)
    rw_cross = (rw[:, 0] < cx).astype(np.float32)
    # Elbow extension (cos angle: 1=straight, -1=folded)
    l_elbow_ang = _elbow_cos(ls, le, lw)
    r_elbow_ang = _elbow_cos(rs, re, rw)
    # Guard up: wrist near own nose (within 0.05 of frame in xy)
    lw_guard = ((lw[:, 1] < nose[:, 1] + 0.05) & (np.abs(lw[:, 0] - nose[:, 0]) < 0.05)).astype(np.float32)
    rw_guard = ((rw[:, 1] < nose[:, 1] + 0.05) & (np.abs(rw[:, 0] - nose[:, 0]) < 0.05)).astype(np.float32)
    # Hip rotation: shoulder axis angle - hip axis angle, derivative
    sh_axis = rs - ls
    hp_axis = rh - lh
    sh_ang = np.arctan2(sh_axis[:, 1], sh_axis[:, 0])
    hp_ang = np.arctan2(hp_axis[:, 1], hp_axis[:, 0])
    twist = sh_ang - hp_ang
    twist_v = _safe_deriv(twist[:, None]).reshape(-1)
    # Wrap to [-pi, pi]
    twist_v = (twist_v + np.pi) % (2 * np.pi) - np.pi
    # Body COM velocity (mid-shoulder)
    com_v = _safe_deriv(chest)
    com_vmag = np.linalg.norm(com_v, axis=1)

    return [
        lw_vmag, rw_vmag,
        lw_amag, rw_amag,
        lw_ext, rw_ext,
        lw_cross, rw_cross,
        l_elbow_ang.astype(np.float32), r_elbow_ang.astype(np.float32),
        lw_guard, rw_guard,
        twist_v.astype(np.float32),
        com_vmag.astype(np.float32),
    ]


def _pair_features(base: np.ndarray) -> list[np.ndarray]:
    T = base.shape[0]
    def xy(kpt, offset):
        sx, sy = kpt_xy(kpt, offset)
        return base[:, sx:sy + 1]
    # Red and blue chest
    red_chest = (xy(L_SHOULDER, 0) + xy(R_SHOULDER, 0)) / 2
    blue_chest = (xy(L_SHOULDER, ROLE_DIM) + xy(R_SHOULDER, ROLE_DIM)) / 2
    # Inter-fighter distance and velocity
    inter_dist = np.linalg.norm(red_chest - blue_chest, axis=1).astype(np.float32)
    inter_vel = _safe_deriv(inter_dist[:, None]).reshape(-1).astype(np.float32)
    # Wrist→opponent nose distances and derivatives
    red_lw = xy(L_WRIST, 0); red_rw = xy(R_WRIST, 0)
    blue_lw = xy(L_WRIST, ROLE_DIM); blue_rw = xy(R_WRIST, ROLE_DIM)
    blue_nose = xy(NOSE, ROLE_DIM); red_nose = xy(NOSE, 0)
    d_rlw_bn = np.linalg.norm(red_lw - blue_nose, axis=1).astype(np.float32)
    d_rrw_bn = np.linalg.norm(red_rw - blue_nose, axis=1).astype(np.float32)
    d_blw_rn = np.linalg.norm(blue_lw - red_nose, axis=1).astype(np.float32)
    d_brw_rn = np.linalg.norm(blue_rw - red_nose, axis=1).astype(np.float32)
    def deriv1(x):
        return _safe_deriv(x[:, None]).reshape(-1).astype(np.float32)
    return [
        inter_dist, inter_vel,
        d_rlw_bn, d_rrw_bn, d_blw_rn, d_brw_rn,
        deriv1(d_rlw_bn), deriv1(d_rrw_bn),
        deriv1(d_blw_rn), deriv1(d_brw_rn),
    ]


def compute_engineered(base: np.ndarray) -> np.ndarray:
    """Compute engineered features. base shape: (T, 142). Returns (T, 38)."""
    feats = []
    feats.extend(_fighter_features(base, 0))       # 14 (red)
    feats.extend(_fighter_features(base, ROLE_DIM))  # 14 (blue)
    feats.extend(_pair_features(base))             # 10
    out = np.stack(feats, axis=1).astype(np.float32)
    # Replace nan/inf with 0
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out  # (T, 38)


ENG_DIM = 38
