import torch.nn as nn


class TAPRegressor(nn.Module):
    """MLP head: pooled embedding -> 4 TAP property predictions."""

    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.mlp(x)
