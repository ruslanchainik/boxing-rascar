from __future__ import annotations

import torch
from torch import nn


HEAD_DIMS = {
    "fighter": 2,
    "punch_type": 4,
    "hand": 2,
    "target": 2,
    "effectiveness": 3,
}

CLASS_NAMES = {
    "fighter": ["red", "blue"],
    "punch_type": ["jab", "cross", "hook", "uppercut"],
    "hand": ["left", "right"],
    "target": ["head", "body"],
    "effectiveness": ["landed", "blocked", "miss"],
}


class TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float = 0.1, kernel: int = 5) -> None:
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.drop(h)
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class BoxingTCN(nn.Module):
    """Per-frame multi-task TCN.

    Inputs: (B, T, F) pose features.
    Outputs dict with logits per frame:
      - event: (B, T) sigmoid heatmap of punch presence
      - fighter/punch_type/hand/target/effectiveness: (B, T, C)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 192,
        n_blocks: int = 6,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [TCNBlock(hidden_dim, dilation=2 ** (i % 5), dropout=dropout) for i in range(n_blocks)]
        )
        self.event_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.attr_heads = nn.ModuleDict(
            {name: nn.Conv1d(hidden_dim, dim, kernel_size=1) for name, dim in HEAD_DIMS.items()}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (B, T, F)
        x = self.input_norm(x)
        h = self.proj(x.transpose(1, 2))  # (B, C, T)
        for blk in self.blocks:
            h = blk(h)
        out = {"event": self.event_head(h).squeeze(1)}  # (B, T)
        for name, head in self.attr_heads.items():
            out[name] = head(h).transpose(1, 2)  # (B, T, C)
        return out
