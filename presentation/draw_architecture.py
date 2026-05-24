"""Render block diagram of BoxingTCN architecture as PNG."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent / "architecture.png"

# Style ---------------------------------------------------------------
BG = "#0b0d12"
INK = "#11141b"
INK_MUTE = "#181c25"
ACCENT = "#ffb547"
ACCENT_DIM = "#f5d76e"
TXT = "#e4e4e7"
TXT_MUTE = "#a1a1aa"
EDGE = "#252b38"

RED = "#ef3b3b"
BLUE = "#3b82f6"
GREEN = "#10b981"
PURPLE = "#a78bfa"
ORANGE = "#fb923c"
AMBER = "#fbbf24"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "text.color": TXT,
    "font.family": "DejaVu Sans",
    "font.size": 9,
})

fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect("equal")
ax.axis("off")


def box(x, y, w, h, label, sub=None, fc=INK, ec=EDGE, fontsize=11,
        subsize=8, bold=False, label_color=TXT, sub_color=TXT_MUTE,
        linestyle="solid"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.4",
                       linewidth=1.2, edgecolor=ec, facecolor=fc,
                       linestyle=linestyle)
    ax.add_patch(p)
    weight = "bold" if bold else "normal"
    if sub:
        ax.text(x + w/2, y + h*0.62, label, ha="center", va="center",
                fontsize=fontsize, color=label_color, fontweight=weight)
        ax.text(x + w/2, y + h*0.30, sub, ha="center", va="center",
                fontsize=subsize, color=sub_color, family="monospace")
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, color=label_color, fontweight=weight)


def arrow(x1, y1, x2, y2, color=ACCENT, width=1.2, style="->"):
    ar = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle=style, mutation_scale=14,
                         linewidth=width, color=color,
                         shrinkA=0, shrinkB=0)
    ax.add_patch(ar)


# ====================================================================
# TITLE
# ====================================================================
ax.text(50, 96.5, "BoxingTCN  ·  per-frame multi-task spotting",
        ha="center", fontsize=20, fontweight="bold", color=ACCENT)
ax.text(50, 94.2,
        "19-model ensemble  ·  3 feature groups (pose 426 / eng 464 / glove 492)  ·  ~2.3M params per model",
        ha="center", fontsize=10, color=TXT_MUTE, style="italic")


# ====================================================================
# LEFT COLUMN: feature sources
# ====================================================================
ax.text(2, 90.5, "1.  Feature sources",
        fontsize=13, fontweight="bold", color=ACCENT_DIM)

# source boxes
sx, sy = 2, 56
sw, sh = 22, 9
gap = 1.8

def src_box(yy, label, dim, desc, fc, ec, color):
    p = FancyBboxPatch((sx, yy), sw, sh,
                       boxstyle="round,pad=0.02,rounding_size=0.4",
                       linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(sx + sw/2, yy + sh - 1.4, label,
            ha="center", va="center", fontsize=11,
            color=color, fontweight="bold")
    ax.text(sx + sw/2, yy + sh - 3.0, dim,
            ha="center", va="center", fontsize=8.5,
            color=TXT_MUTE, family="monospace")
    ax.text(sx + sw/2, yy + 1.7, desc,
            ha="center", va="center", fontsize=7.5,
            color=TXT_MUTE, family="monospace")

src_box(sy + 2*(sh+gap),
        "Pose features", "426 dim  ·  yolo11n-pose",
        "17 kpts × 2 fighter (51×2)\n+ bbox(4) + HSV(3) + Δ1/Δ3 deltas",
        "#10243a", BLUE, BLUE)
src_box(sy + 1*(sh+gap),
        "+ Engineered features", "464 dim  (+38)",
        "wrist accel/jerk · hand extension\nelbow cos · hip rotation · COM vel",
        "#0e2a1f", GREEN, GREEN)
src_box(sy,
        "+ Glove tracker features", "492 dim  (+28)",
        "yolov8n on auto-labels (mAP50 0.61)\n→ glove pos/size/vel/accel",
        "#1d1530", PURPLE, PURPLE)


# ====================================================================
# MIDDLE COLUMN: TCN core (vertical flow)
# ====================================================================
ax.text(40, 90.5, "2.  TCN core  (per group)",
        fontsize=13, fontweight="bold", color=ACCENT_DIM)

cx = 40
cw = 24

# input
iy = 82
box(cx, iy, cw, 5, "Input  T × D", "D ∈ {426, 464, 492}",
    fc=INK, ec=ACCENT, label_color=ACCENT, fontsize=11, subsize=8.5)

# arrows from sources to input
for (yy, col) in [(sy + 2*(sh+gap) + sh/2, BLUE),
                  (sy + 1*(sh+gap) + sh/2, GREEN),
                  (sy + sh/2, PURPLE)]:
    arrow(sx + sw + 0.4, yy, cx - 0.4, iy + 2.5, color=col, width=1.2)

# LayerNorm + Proj
ly = iy - 7
box(cx, ly, cw, 5, "LayerNorm + Conv1d 1×1", "→ T × 192",
    fc=INK_MUTE, fontsize=10, subsize=8.5)
arrow(cx + cw/2, iy, cx + cw/2, ly + 5)

# TCN blocks row — vertical stack (6 boxes inside a frame)
by = ly - 32
# container
container = FancyBboxPatch((cx - 0.5, by - 0.5), cw + 1, 30,
                           boxstyle="round,pad=0.02,rounding_size=0.5",
                           linewidth=1.5, edgecolor=ACCENT_DIM,
                           facecolor="#0e1219", linestyle="--")
ax.add_patch(container)
ax.text(cx + cw/2, by + 30 + 0.7, "6 dilated TCN blocks",
        ha="center", fontsize=10, color=ACCENT_DIM, fontweight="bold")

dilations = [1, 2, 4, 8, 16, 1]
bw, bh = cw - 4, 3.6
bgap = 0.9
for i, d in enumerate(dilations):
    bx_ = cx + 2
    by_ = by + 24.5 - i * (bh + bgap)
    box(bx_, by_, bw, bh,
        f"TCN block",
        f"dil = {d}",
        fc="#1f2737", ec="#3a4258", fontsize=10, subsize=8)
    if i > 0:
        arrow(bx_ + bw/2, by_ + bh + bgap, bx_ + bw/2, by_ + bh,
              color=ACCENT, width=1.0)

# arrow LN → first block
arrow(cx + cw/2, ly, cx + cw/2, by + 24.5 + bh + 0.3, color=ACCENT)

# recep field note
ax.text(cx + cw + 1, by + 14,
        "Receptive\nfield\n≈ 512 frames\n≈ 17 s @ 30 fps",
        fontsize=8.5, color=TXT_MUTE, style="italic", va="center")

# arrow last block → output features
out_y = by - 6
box(cx, out_y, cw, 5, "Per-frame features", "T × 192",
    fc=INK, ec=ACCENT, label_color=ACCENT, fontsize=11, subsize=8.5)
arrow(cx + cw/2, by, cx + cw/2, out_y + 5)


# ====================================================================
# RIGHT COLUMN: heads
# ====================================================================
ax.text(72, 90.5, "3.  Multi-task heads",
        fontsize=13, fontweight="bold", color=ACCENT_DIM)

heads = [
    ("event",         "1  ·  focal BCE",                 AMBER),
    ("fighter",       "2  ·  red / blue",                RED),
    ("punch_type",    "4  ·  jab / cross / hook / upper", BLUE),
    ("hand",          "2  ·  left / right",              PURPLE),
    ("target",        "2  ·  head / body",               GREEN),
    ("effectiveness", "3  ·  landed / blocked / miss",   ORANGE),
]

hx = 72
hw = 26
htop = 81
hh = 5.5
hgap = 1.0

for i, (name, info, color) in enumerate(heads):
    hy = htop - i * (hh + hgap)
    box(hx, hy, hw, hh, f"Head: {name}", info,
        fc=INK_MUTE, label_color=color, fontsize=11, subsize=8.5)
    # arrow from output features
    arrow(cx + cw + 0.5, out_y + 2.5,
          hx - 0.5, hy + hh/2, color=ACCENT, width=0.9)


# ====================================================================
# TCN BLOCK DETAIL (lower middle, full width)
# ====================================================================
ax.text(2, 38, "4.  TCN block (residual + dilated convolution)",
        fontsize=13, fontweight="bold", color=ACCENT_DIM)

dx, dy = 2, 22
dw, dh = 96, 13

# outer container
container = FancyBboxPatch((dx, dy), dw, dh,
                           boxstyle="round,pad=0.05,rounding_size=0.6",
                           linewidth=1.4, edgecolor=ACCENT_DIM,
                           facecolor="#0e1219", linestyle="--")
ax.add_patch(container)

# nodes
nodes = [
    ("Input", "T × 192", "#1f2737", ACCENT),
    ("Conv1d", "k=5, dil", "#1f2737", TXT),
    ("GroupNorm", "8 groups", INK_MUTE, TXT),
    ("GELU", "", INK_MUTE, TXT),
    ("Dropout", "p=0.15", INK_MUTE, TXT),
    ("Conv1d", "k=5, dil", "#1f2737", TXT),
    ("GroupNorm", "8 groups", INK_MUTE, TXT),
    ("⊕", "add", "#10b981", BG),
    ("GELU", "", INK_MUTE, TXT),
    ("Out", "T × 192", "#1f2737", ACCENT),
]

n_count = len(nodes)
node_w = 8
node_h = 4.5
node_gap = (dw - 4 - n_count * node_w) / (n_count - 1)
nx_start = dx + 2
ny = dy + 4

for i, (lbl, sub, fc, color) in enumerate(nodes):
    nx = nx_start + i * (node_w + node_gap)
    box(nx, ny, node_w, node_h, lbl, sub if sub else None, fc=fc,
        label_color=color, fontsize=10, subsize=7.5)
    if i > 0:
        arrow(nx - node_gap, ny + node_h/2, nx, ny + node_h/2,
              color=ACCENT_DIM, width=0.9)

# residual arc — from after Input to ⊕
res_in_x = nx_start + node_w/2     # midpoint of "Input" → use right edge
res_in_x = nx_start + node_w + node_gap / 2  # right after Input box
add_node_idx = 7
res_out_x = nx_start + add_node_idx * (node_w + node_gap) + node_w/2

res_top_y = dy + dh - 1.2
ax.plot([res_in_x, res_in_x, res_out_x, res_out_x],
        [ny + node_h, res_top_y, res_top_y, ny + node_h],
        color=GREEN, linewidth=1.6)
ax.text((res_in_x + res_out_x) / 2, res_top_y + 0.4,
        "residual skip",
        ha="center", fontsize=9.5, color=GREEN, fontweight="bold")


# ====================================================================
# BOTTOM: training + ensemble + postprocess
# ====================================================================
ax.text(2, 18, "5.  Training  ·  ensemble  ·  postprocess",
        fontsize=13, fontweight="bold", color=ACCENT_DIM)

steps = [
    ("Multi-task loss",
     "focal BCE(event, γ=2) + Σ λᵢ · CE(attrᵢ, weight=inv_freq)\n"
     "Gaussian soft target σ=3 · label smoothing 0.05 · color-swap aug · EMA"),
    ("19-model ensemble",
     "9 orig (D=426) + 5 eng (D=464) + 5 glove (D=492)\n"
     "per-frame mean of event scores and attr logits"),
    ("Test-time aug",
     "color-swap (red ↔ blue swap in features + swap fighter logits)\n"
     "averaged with original prediction  ·  contributed +0.028 LB"),
    ("Peak picking + postprocess",
     "threshold 0.65 · min_distance 6 frames · 5-frame attr smoothing\n"
     "30-fps frame mapping for Kaggle metric"),
]

sx2, sy2 = 2, 14
sw2, sh2 = 70, 2.4
for i, (name, body) in enumerate(steps):
    by_ = sy2 - i * (sh2 + 0.4)
    box(sx2, by_, 18, sh2, name, fc=INK_MUTE,
        label_color=ACCENT_DIM, fontsize=10)
    ax.text(sx2 + 19, by_ + sh2/2, body, fontsize=8, color=TXT_MUTE,
            va="center", family="monospace")


# ====================================================================
# Result tag (bottom right)
# ====================================================================
result_x = 76
result_y = 4
result_w = 22
result_h = 9
box(result_x, result_y, result_w, result_h,
    "Public LB", "0.16718",
    fc="#1a2233", ec=ACCENT, label_color=ACCENT,
    fontsize=14, subsize=28, bold=True, sub_color=ACCENT)
ax.text(result_x + result_w/2, result_y - 1.2,
        "9 test videos  ·  macro average  ·  0.026 → 0.167",
        ha="center", color=TXT_MUTE, fontsize=8.5, style="italic")


plt.tight_layout()
plt.savefig(OUT, dpi=180, facecolor=BG, bbox_inches="tight")
print(f"saved {OUT}")
