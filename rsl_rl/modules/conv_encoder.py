# conv_encoder.py
import torch
import torch.nn as nn
from tensordict import TensorDict


class ConvHistoryEncoder(nn.Module):
    def __init__(self, history_keys: list[str], obs: TensorDict, T: int, out_dim: int = 128):
        super().__init__()
        self.T = T
        self.history_keys = history_keys

        # compute in_channels directly from obs shapes — no manual n_joints/n_actions
        in_channels = sum(obs["policy"][k].shape[-1] for k in history_keys)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64,  kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv1d(64,          128, kernel_size=3, padding=2),
            nn.ReLU(),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, history: dict) -> torch.Tensor:
        # history values: [N, T, feat_dim]
        x = torch.cat(
            [history[k] for k in self.history_keys], dim=-1
        )                        # [N, T, D]
        x = x.permute(0, 2, 1)  # [N, D, T]

        x = self.conv(x)         # [N, 128, T']
        x = x[:, :, -1]          # [N, 128]  last (most recent) timestep
        return self.proj(x)      # [N, out_dim]