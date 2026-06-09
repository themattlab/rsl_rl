import torch
import torch.nn as nn
from tensordict import TensorDict


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution: output at position t depends only on inputs at t' <= t.
    Achieved by left-padding (kernel_size - 1) zeros, then trimming the right
    tail so the output length matches the input length exactly.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self._trim = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self._trim,   # pad left AND right by (k-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, T]
        out = self.conv(x)                  # [N, C_out, T + trim]
        return out[:, :, : -self._trim]     # [N, C_out, T]  — drop right padding


class ConvHistoryEncoder(nn.Module):
    def __init__(
        self,
        history_keys: list[str],
        obs: TensorDict,
        T: int,
        out_dim: int = 128,
    ):
        super().__init__()
        self.T = T
        self.history_keys = history_keys

        in_channels = sum(obs["policy"][k].shape[-1] for k in history_keys)

        self.conv = nn.Sequential(
            CausalConv1d(in_channels, 64,  kernel_size=5),
            nn.ReLU(),
            CausalConv1d(64,          128, kernel_size=5),
            nn.ReLU(),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, history: dict) -> torch.Tensor:
        # history values: [N, T, feat_dim]
        x = torch.cat(
            [history[k] for k in self.history_keys], dim=-1
        )                        # [N, T, D]
        x = x.permute(0, 2, 1)  # [N, D, T]

        x = self.conv(x)         # [N, 128, T]  — same length, causal
        x = x[:, :, -1]          # [N, 128]  — truly the most-recent timestep
        return self.proj(x)      # [N, out_dim]


# class ConvHistoryEncoder(nn.Module):
#     def __init__(
#         self,
#         history_keys: list[str],
#         obs: TensorDict,
#         T: int,
#         out_dim: int = 128,
#     ):
#         super().__init__()
#         self.T = T
#         self.history_keys = history_keys
#         print("\n\n\n\n\nUsing Linear History Encoder\n\n\n\n")

#         in_dim = T * sum(obs["policy"][k].shape[-1] for k in history_keys)
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, 256), nn.ELU(),
#             nn.Linear(256, 256),    nn.ELU(),
#             nn.Linear(256, out_dim),
#         )

#     def forward(self, history: dict) -> torch.Tensor:
#         x = torch.cat([history[k] for k in self.history_keys], dim=-1)  # [N, T, D]
#         return self.net(x.flatten(1))  # [N, out_dim]
