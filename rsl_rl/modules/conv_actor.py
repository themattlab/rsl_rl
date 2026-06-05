# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn
from .conv_encoder import ConvHistoryEncoder

from rsl_rl.networks import MLP, EmpiricalNormalization

ACTIVATIONS = {
    "elu":   nn.ELU,
    "relu":  nn.ReLU,
    "tanh":  nn.Tanh,
    "selu":  nn.SELU,
    "lrelu": nn.LeakyReLU,
}


class ConvActorCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            print(
                "ActorCritic extra params: " + str([key for key in kwargs])
            )
        super().__init__()

        history_len = kwargs["history_len"]
        enc_out_dim = kwargs.get("enc_out_dim", 128)

        self.flat_keys: list[str] = []
        self.history_keys: list[str] = []

        for key,val in obs["policy"].items():
            if val.ndim == 2:
                self.flat_keys.append(key)
            elif val.ndim == 3:
                self.history_keys.append(key)
            else:
                raise ValueError(f"Observation {key} has invalid dimension: {val.ndim}")

        self.encoder = ConvHistoryEncoder( history_keys=self.history_keys, obs=obs, T=history_len, out_dim=enc_out_dim)

        flat_dim = sum(obs["policy"][key].shape[-1] for key in self.flat_keys)
        mlp_input_dim = enc_out_dim + flat_dim

        # Get the observation dimensions
        self.obs_groups = obs_groups

        self.state_dependent_std = state_dependent_std

        # Actor
        self.actor  = self._build_mlp(mlp_input_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(flat_dim) if actor_obs_normalization and flat_dim > 0
            else nn.Identity()
        )

        # Critic
        self.critic = self._build_mlp(mlp_input_dim, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(flat_dim) if critic_obs_normalization and flat_dim > 0
            else nn.Identity()
        )

        print("-------------- Using ConvActorCritic --------------  ")
        print(f"History Length: {history_len}, Encoder Output Dim: {enc_out_dim}")
        print(f"Flat Keys: {self.flat_keys}, History Keys: {self.history_keys}")
        print("-------------- Using ConvActorCritic --------------")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def _build_mlp(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int] | list[int],
        activation: str,
    ) -> nn.Module:
        act_cls = ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act_cls()]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def _encode(self, obs: TensorDict, normalizer: nn.Module) -> torch.Tensor:
        """Encode history through conv + concat normalised flat terms."""
        history = {k: obs["policy"][k] for k in self.history_keys}
        conv_out = self.encoder(history)                     # [N, enc_out_dim]

        if self.flat_keys:
            flat = torch.cat(
                [obs["policy"][k] for k in self.flat_keys], dim=-1
            )                                                # [N, flat_dim]
            flat = normalizer(flat)
            return torch.cat([conv_out, flat], dim=-1)       # [N, mlp_input_dim]

        return conv_out

    def _get_std(self, mean: torch.Tensor) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            return self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            return torch.exp(self.log_std).expand_as(mean)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    # def _update_distribution(self, obs: TensorDict) -> None:
    #     if self.state_dependent_std:
    #         # Compute mean and standard deviation
    #         mean_and_std = self.actor(obs)
    #         if self.noise_std_type == "scalar":
    #             mean, std = torch.unbind(mean_and_std, dim=-2)
    #         elif self.noise_std_type == "log":
    #             mean, log_std = torch.unbind(mean_and_std, dim=-2)
    #             std = torch.exp(log_std)
    #         else:
    #             raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
    #     else:
    #         # Compute mean
    #         mean = self.actor(obs)
    #         # Compute standard deviation
    #         if self.noise_std_type == "scalar":
    #             std = self.std.expand_as(mean)
    #         elif self.noise_std_type == "log":
    #             std = torch.exp(self.log_std).expand_as(mean)
    #         else:
    #             raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
    #     # Create distribution
    #     self.distribution = Normal(mean, std)

    def _maybe_encode(self, obs, normalizer: nn.Module) -> torch.Tensor:
        """Accept either raw nested TensorDict (from env) or already-encoded flat tensor (from storage)."""
        if isinstance(obs, torch.Tensor):
            return obs          # already encoded — came from storage mini-batch
        return self._encode(obs, normalizer)   # raw dict — came from env

    def _update_distribution(self, obs) -> None:
        features = self._maybe_encode(obs, self.actor_obs_normalizer)
        mean = self.actor(features)
        self.distribution = Normal(mean, self._get_std(mean))

    def act(self, obs, **kwargs) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs) -> torch.Tensor:
        features = self._maybe_encode(obs, self.actor_obs_normalizer)
        return self.actor(features)

    def evaluate(self, obs, **kwargs) -> torch.Tensor:
        features = self._maybe_encode(obs, self.critic_obs_normalizer)
        return self.critic(features)

    def _flat_terms(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate the 2D (non-history) observation terms into a single flat tensor.

        This mirrors :meth:`ActorCritic.get_actor_obs`, but only over the flat terms since
        the history terms are consumed by the conv encoder rather than the normalizer/MLP
        directly.
        """
        return torch.cat([obs["policy"][k] for k in self.flat_keys], dim=-1)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._flat_terms(obs)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._flat_terms(obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        # Only the flat terms are normalized; the conv encoder output is not.
        if not self.flat_keys:
            return
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self._flat_terms(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self._flat_terms(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
