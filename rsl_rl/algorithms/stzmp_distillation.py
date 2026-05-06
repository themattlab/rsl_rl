# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ST-ZMP distillation algorithm.

Extends the standard :class:`~rsl_rl.algorithms.Distillation` algorithm with a
combined loss::

    L = w_bc  · MSE(student_actions, teacher_actions)
      + w_nll · GaussianNLL(mu_zmp, logvar_zmp, delta_zmp_star)

where the Gaussian NLL is the standard negative log-likelihood::

    NLL = 0.5 · mean( logvar + (mu - target)² / exp(logvar) )

The ``delta_zmp_star`` ground-truth ZMP shift must be provided as a privileged
observation group (key ``delta_zmp_obs_key``, default ``"zmp_star"``) in the
environment's TensorDict.  Because it is part of the observation TensorDict it
is stored automatically by :class:`~rsl_rl.storage.RolloutStorage` and replayed
during the update without any extra buffer.

References
----------
STZMP_Implementation_Guide.docx, sections 4-6.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.distillation import Distillation
from rsl_rl.modules.stzmp_student_teacher import STZMPStudentTeacher
from rsl_rl.utils import resolve_optimizer


class STZMPDistillation(Distillation):
    """Distillation with NLL + behaviour-cloning combined loss for ST-ZMP.

    Inherits storage initialisation, multi-GPU support, rollout collection
    (``act`` / ``process_env_step``), and checkpointing from
    :class:`~rsl_rl.algorithms.Distillation`.  Only ``update()`` is overridden
    to add the NLL term.

    Args:
        policy: The :class:`~rsl_rl.modules.STZMPStudentTeacher` module.
        w_bc: Weight for the behaviour-cloning (MSE) loss (default 1.0).
        w_nll: Weight for the Gaussian NLL loss on the ZMP latent (default 1.0).
        delta_zmp_obs_key: Key in the observation TensorDict that holds the
            ground-truth 2-D ZMP shift ``(Δx, Δy)`` (default ``"zmp_star"``).
        num_learning_epochs: Number of epochs per update (default 1).
        gradient_length: Number of time steps to accumulate before a gradient
            step — enables BPTT-style training (default 15).
        learning_rate: Optimiser learning rate (default 1e-3).
        max_grad_norm: Gradient clipping norm (``None`` = no clipping).
        loss_type: Base loss for BC — ``"mse"`` or ``"huber"`` (default ``"mse"``).
        optimizer: Optimiser name (default ``"adam"``).
        device: Compute device (default ``"cpu"``).
        multi_gpu_cfg: Optional distributed training config dict.
    """

    policy: STZMPStudentTeacher

    def __init__(
        self,
        policy: STZMPStudentTeacher,
        w_bc: float = 1.0,
        w_nll: float = 1.0,
        delta_zmp_obs_key: str = "zmp_star",
        # Inherited Distillation params
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float | None = None,
        loss_type: str = "mse",
        optimizer: str = "adam",
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            policy=policy,
            num_learning_epochs=num_learning_epochs,
            gradient_length=gradient_length,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            loss_type=loss_type,
            optimizer=optimizer,
            device=device,
            multi_gpu_cfg=multi_gpu_cfg,
        )
        self.w_bc = w_bc
        self.w_nll = w_nll
        self.delta_zmp_obs_key = delta_zmp_obs_key

        print(
            f"[STZMPDistillation] w_bc={w_bc}  w_nll={w_nll}  "
            f"delta_zmp_obs_key='{delta_zmp_obs_key}'"
        )

    # ── Loss helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _gaussian_nll(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian negative log-likelihood (per-element mean).

        ``NLL = 0.5 · mean(logvar + (mu - target)² / exp(logvar))``

        Args:
            mu: Predicted mean, shape ``[B, 2]``.
            logvar: Predicted log-variance (clamped), shape ``[B, 2]``.
            target: Ground-truth ZMP shift ``(Δx, Δy)``, shape ``[B, 2]``.

        Returns:
            Scalar loss.
        """
        return 0.5 * (logvar + (mu - target).pow(2) / logvar.exp()).mean()

    # ── Update ───────────────────────────────────────────────────────────────

    def update(self) -> dict[str, float]:
        """Run one distillation update over the collected rollout.

        For each time step in the rollout storage:

        1. Run ``policy.act_with_latent(obs)`` to obtain student actions,
           ``mu_zmp``, and ``logvar_zmp`` (with gradients).
        2. Compute BC loss against teacher-generated ``privileged_actions``.
        3. Compute NLL loss against stored ``delta_zmp_star`` targets.
        4. Accumulate total loss; perform an optimiser step every
           ``gradient_length`` steps.

        Returns:
            Dict with keys ``"behavior"``, ``"nll"``, ``"total"`` — each the
            mean per-step value over this update.
        """
        self.num_updates += 1

        mean_bc_loss    = 0.0
        mean_nll_loss   = 0.0
        mean_total_loss = 0.0
        loss            = torch.tensor(0.0, device=self.device)
        cnt             = 0

        for _ in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()

            for obs, _, privileged_actions, dones in self.storage.generator():
                # ── Student forward ──────────────────────────────────────────
                # act_with_latent runs the encoder (stochastic) + actor in one pass
                actions, mu_zmp, logvar_zmp = self.policy.act_with_latent(obs)

                # ── Behaviour-cloning loss ───────────────────────────────────
                bc_loss = self.loss_fn(actions, privileged_actions)

                # ── NLL loss on ZMP latent ───────────────────────────────────
                if self.delta_zmp_obs_key not in obs.keys():
                    raise KeyError(
                        f"[STZMPDistillation] delta_zmp_obs_key='{self.delta_zmp_obs_key}' "
                        f"not found in stored observations. "
                        f"Available keys: {list(obs.keys())}.\n"
                        f"  Add a '{self.delta_zmp_obs_key}' obs group to your Isaac Lab task "
                        f"and include it in obs_groups in the training config."
                    )
                delta_zmp_star = obs[self.delta_zmp_obs_key]                 # [B, 2]
                nll_loss = self._gaussian_nll(mu_zmp, logvar_zmp, delta_zmp_star)

                # ── Weighted total ───────────────────────────────────────────
                step_loss = self.w_bc * bc_loss + self.w_nll * nll_loss
                loss = loss + step_loss

                mean_bc_loss    += bc_loss.item()
                mean_nll_loss   += nll_loss.item()
                mean_total_loss += step_loss.item()
                cnt             += 1

                # ── Gradient step every gradient_length steps ────────────────
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(
                            self.policy.parameters(), self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = torch.tensor(0.0, device=self.device)

                # Reset done environments
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        # Normalise by number of steps
        mean_bc_loss    /= cnt
        mean_nll_loss   /= cnt
        mean_total_loss /= cnt

        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        return {
            "behavior": mean_bc_loss,
            "nll":      mean_nll_loss,
            "total":    mean_total_loss,
        }
