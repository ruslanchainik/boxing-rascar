from __future__ import annotations

import torch
from torch import nn


HEAD_DIMS = {
    "event": 1,
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


class BoxingLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.25,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        enc_dim = hidden_dim * (2 if bidirectional else 1)
        self.body = nn.Sequential(
            nn.LayerNorm(enc_dim),
            nn.Linear(enc_dim, enc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(enc_dim, dim)
                for name, dim in HEAD_DIMS.items()
            }
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.norm(x)
        enc, _ = self.encoder(x)
        center = enc[:, enc.shape[1] // 2]
        z = self.body(center)
        return {name: head(z) for name, head in self.heads.items()}


def decode_classes(logits: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    decoded: dict[str, list[str]] = {}
    for name, names in CLASS_NAMES.items():
        idx = logits[name].argmax(dim=1).detach().cpu().numpy().tolist()
        decoded[name] = [names[i] for i in idx]
    return decoded

