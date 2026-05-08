# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spatio-Temporal ZMP Attention Student-Teacher module.

Architecture overview
---------------------
The teacher is a standard MLP trained with privileged force information (F_ext).
The student replaces F_ext with a 2-D Gaussian ZMP latent (mu_zmp, sigma²_zmp)
inferred from proprioceptive history via a cross-attention encoder.

Policy obs layout (``obs["policy"]``)
--------------------------------------
Isaac Lab concatenates observation terms **in this exact order** into a flat vector::

    [base_lin_vel     (base_lin_vel_dim)
     base_ang_vel     (base_ang_vel_dim)
     projected_gravity(projected_gravity_dim)
     command          (command_dim)             ← foot_position_commands
     joint_pos_hist   (history_len × num_joints)  ← oldest step first
     joint_vel_hist   (history_len × num_joints)
     action_hist      (history_len × num_joints)
     height_scan      (height_scan_dim, inferred)]

The constructor asserts the total dimension, prints a detailed layout table, and
raises a descriptive error if sizes are inconsistent.

Encoder
-------
1. Shared temporal MLP compresses each leg's [q | dq | action] history to d_model.
2. Learned positional embeddings distinguish the four legs.
3. Base projection MLP embeds task-conditioned base token:
   [projected_gravity | base_ang_vel | base_lin_vel | command].
   Including lin_vel and command makes the query task-conditioned — the encoder
   learns to attend to the correct manipulator leg given the active command.
4. Cross-attention: base token (query) attends to four leg tokens (key/value).
5. mu_head + logvar_head produce the 2-D Gaussian ZMP latent.

Actor
-----
Receives: joint_pos_cur, joint_vel_cur, actions_recent (last actor_action_history_steps
steps), all base terms, command, height_scan, z_sample, logvar_zmp.
The encoder always sees the full history_len action steps inside each leg token;
the actor sees only a short recent window (default 3) for smoothness.

References
----------
STZMP_Implementation_Guide.docx, sections 3-5.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.utils import resolve_nn_activation


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class STZMPEncoder(nn.Module):
    """Spatio-temporal cross-attention encoder for ZMP latent estimation.

    Processes per-leg proprioceptive history and current base state to produce
    a 2-D Gaussian over the horizontal ZMP shift (Δx, Δy).

    Args:
        num_legs: Number of legs (default 4).
        leg_token_dim: Flattened input size for each leg token.
            Equals ``history_len × joints_per_leg × 3`` (q + dq + action).
        base_dim: Dimension of the task-conditioned base token input (default 14:
            projected_gravity(3) + base_ang_vel(3) + base_lin_vel(3) + command(5)).
            Including lin_vel and command makes the cross-attention query
            task-conditioned so the encoder attends to the correct leg.
        d_model: Attention embedding dimension (default 32).
        num_heads: Number of attention heads; must divide ``d_model`` (default 4).
        temporal_mlp_width: Hidden width of the shared temporal MLP (default 64).
        activation: Activation function name (default ``"elu"``).
    """

    def __init__(
        self,
        num_legs: int = 4,
        leg_token_dim: int = 90,
        base_dim: int = 14,
        d_model: int = 32,
        num_heads: int = 4,
        temporal_mlp_width: int = 64,
        activation: str = "elu",
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.num_legs = num_legs
        self.d_model = d_model

        act = resolve_nn_activation(activation)

        # Shared temporal MLP — identical weights applied to every leg token
        self.leg_temporal_mlp = nn.Sequential(
            nn.Linear(leg_token_dim, temporal_mlp_width),
            act,
            nn.Linear(temporal_mlp_width, d_model),
        )

        # Learned positional embedding — one d_model vector per leg
        self.leg_pos_embedding = nn.Parameter(torch.randn(num_legs, d_model) * 0.02)

        # Base state projection MLP (H=1, no history)
        self.base_proj_mlp = nn.Sequential(
            nn.Linear(base_dim, d_model),
            act,
            nn.Linear(d_model, d_model),
        )

        # Single-query cross-attention: base queries the four leg tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )

        # Gaussian ZMP output heads
        self.mu_head = nn.Linear(d_model, 2)

        self.logvar_head = nn.Linear(d_model, 2)
        # Initialise so σ² ≈ 0.14 initially (doc section 3.2 / 8.1).
        # This prevents the trivially-large-σ early phase described in section 5.1.
        nn.init.constant_(self.logvar_head.bias, -2.0)
        nn.init.normal_(self.logvar_head.weight, std=0.01)

    def forward(
        self,
        leg_tokens: torch.Tensor,
        base_current: torch.Tensor,
        deterministic: bool = False,
        return_attn_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run the encoder forward pass.

        Args:
            leg_tokens: Shape ``[B, num_legs, leg_token_dim]``.
            base_current: Shape ``[B, base_dim]``.
            deterministic: If ``True``, return ``mu_zmp`` as the sample (no noise).
            return_attn_weights: If ``True``, compute and return the cross-attention
                weight matrix (shape ``[B, num_heads, 1, num_legs]`` averaged to
                ``[B, 1, num_legs]``). Adds a small overhead; leave ``False`` during
                training and set to ``True`` only for logging / paper figures.

        Returns:
            Tuple ``(mu_zmp, logvar_zmp, z_sample, attn_weights)``, each ``[B, 2]``
            for the first three.  ``attn_weights`` is ``[B, 1, num_legs]`` when
            ``return_attn_weights=True``, else ``None``.
        """
        B = leg_tokens.shape[0]

        # Step 1: compress each leg history with the shared temporal MLP
        # Process all legs simultaneously by merging batch and leg dimensions
        leg_flat = leg_tokens.reshape(B * self.num_legs, -1)          # [B*L, leg_token_dim]
        e_legs_flat = self.leg_temporal_mlp(leg_flat)                  # [B*L, d_model]
        e_legs = e_legs_flat.reshape(B, self.num_legs, self.d_model)   # [B, L, d_model]

        # Step 2: add learned positional embeddings (broadcast over batch)
        e_legs = e_legs + self.leg_pos_embedding.unsqueeze(0)          # [B, L, d_model]

        # Step 3: project base state to d_model
        e_base = self.base_proj_mlp(base_current).unsqueeze(1)         # [B, 1, d_model]

        # Step 4: cross-attention — base (query) attends to legs (key, value).
        # need_weights=True is required to get the attention matrix; it is disabled
        # during normal training for speed (average_attn_weights collapses heads → [B,1,L]).
        z_latent, attn_weights = self.cross_attn(
            query=e_base,                                # [B, 1, d_model]
            key=e_legs,                                  # [B, L, d_model]
            value=e_legs,                                # [B, L, d_model]
            need_weights=return_attn_weights,            # False → faster; True → returns [B, 1, L]
            average_attn_weights=True,                   # average over heads → [B, 1, L]
        )
        z_latent = z_latent.squeeze(1)                                  # [B, d_model]

        # Step 5: Gaussian heads
        mu_zmp = self.mu_head(z_latent)                                 # [B, 2]
        # Clamp logvar to [-10, 2] for numerical stability (doc section 3.2)
        logvar_zmp = self.logvar_head(z_latent).clamp(-10.0, 2.0)       # [B, 2]

        # Reparameterisation trick (training) / deterministic mean (deployment)
        if deterministic:
            z_sample = mu_zmp
        else:
            std = torch.exp(0.5 * logvar_zmp)
            z_sample = mu_zmp + std * torch.randn_like(std)             # [B, 2]

        return mu_zmp, logvar_zmp, z_sample, attn_weights


# ---------------------------------------------------------------------------
# Student-Teacher module
# ---------------------------------------------------------------------------

class STZMPStudentTeacher(nn.Module):
    """Student-Teacher distillation module with spatio-temporal ZMP attention.

    The **teacher** is a standard MLP operating on privileged observations
    (including F_ext).  The **student** replaces F_ext with a Gaussian ZMP
    latent produced by :class:`STZMPEncoder` from proprioceptive history alone.

    Policy obs layout
    -----------------
    Isaac Lab must concatenate the following observation terms in this exact order
    into the ``"policy"`` observation group::

        base_lin_vel          (base_lin_vel_dim D,       default 3)
        base_ang_vel          (base_ang_vel_dim D,       default 3)
        projected_gravity     (projected_gravity_dim D,  default 3)
        foot_position_commands(command_dim D,            default 5)
        joint_pos             (history_len × num_joints, oldest first)
        joint_vel             (history_len × num_joints)
        actions               (history_len × num_joints)
        height_scan           (height_scan_dim D,        inferred)

    A startup assertion checks that the configured sizes sum to the actual obs
    dimension and prints a detailed layout table.

    Args:
        obs: Initial TensorDict from ``env.get_observations()``.
        obs_groups: Observation group mapping (must contain ``"policy"`` and ``"teacher"``).
        num_actions: Number of action dimensions.
        history_len: History steps per joint (default 10).
        num_joints: Total robot joints (default 12).
        num_legs: Number of legs (default 4).
        leg_joint_indices: Joint indices per leg, e.g. ``[[0,1,2],[3,4,5],[6,7,8],[9,10,11]]``.
        history_order: ``"oldest_first"`` (Isaac Lab default) or ``"newest_first"``.
        base_lin_vel_dim: Dimension of base linear velocity (default 3).
        base_ang_vel_dim: Dimension of base angular velocity (default 3).
        projected_gravity_dim: Dimension of projected gravity (default 3).
        command_dim: Dimension of foot position commands (default 5).
        d_model: Encoder embedding dimension (default 32).
        num_heads: Attention heads (default 4).
        temporal_mlp_width: Temporal MLP hidden width (default 64).
        activation: Activation name (default ``"elu"``).
        actor_hidden_dims: Student actor MLP hidden widths.
        teacher_hidden_dims: Teacher MLP hidden widths.
        teacher_obs_normalization: Apply empirical normalization to teacher obs.
        init_noise_std: Initial action noise std (default 0.1).
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # History / joint layout
        history_len: int = 10,
        num_joints: int = 12,
        num_legs: int = 4,
        leg_joint_indices: list[list[int]] | None = None,
        history_order: str = "oldest_first",
        # Per-term sizes within policy obs (must match Isaac Lab term order)
        base_lin_vel_dim: int = 3,
        base_ang_vel_dim: int = 3,
        projected_gravity_dim: int = 3,
        command_dim: int = 5,
        # Encoder hyper-parameters
        d_model: int = 32,
        num_heads: int = 4,
        temporal_mlp_width: int = 64,
        activation: str = "elu",
        # MLP hidden layer widths
        actor_hidden_dims: list[int] | None = None,    # default: [512, 256, 128]
        teacher_hidden_dims: list[int] | None = None,  # default: [512, 256, 128]
        # Action history depth for the actor (encoder always sees full history_len)
        actor_action_history_steps: int = 3,
        # Human-readable leg names in the same order as leg_joint_indices
        leg_names: list[str] | None = None,
        # Misc
        teacher_obs_normalization: bool = False,
        init_noise_std: float = 0.1,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            print(
                "[STZMPStudentTeacher] Ignoring unexpected constructor arguments: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.loaded_teacher = False
        self.obs_groups = obs_groups

        # ── Leg joint mapping ────────────────────────────────────────────────
        if leg_joint_indices is None:
            joints_per_leg = num_joints // num_legs
            leg_joint_indices = [
                list(range(i * joints_per_leg, (i + 1) * joints_per_leg))
                for i in range(num_legs)
            ]
        if len(leg_joint_indices) != num_legs:
            raise ValueError(
                f"[STZMPStudentTeacher] leg_joint_indices has {len(leg_joint_indices)} entries "
                f"but num_legs={num_legs}."
            )
        # Register as buffers so they move to the right device automatically
        for i, idx in enumerate(leg_joint_indices):
            self.register_buffer(
                f"_leg_idx_{i}", torch.tensor(idx, dtype=torch.long), persistent=False
            )
        self._num_leg_index_sets = num_legs
        joints_per_leg = len(leg_joint_indices[0])

        # ── Leg names (for debug outputs and metadata) ───────────────────────
        if leg_names is None:
            leg_names = [f"leg_{i}" for i in range(num_legs)]
        if len(leg_names) != num_legs:
            raise ValueError(
                f"[STZMPStudentTeacher] leg_names has {len(leg_names)} entries "
                f"but num_legs={num_legs}."
            )
        self.leg_names: list[str] = list(leg_names)

        # ── History ordering ─────────────────────────────────────────────────
        if history_order not in ("oldest_first", "newest_first"):
            raise ValueError(
                f"[STZMPStudentTeacher] history_order must be 'oldest_first' or "
                f"'newest_first', got '{history_order}'."
            )
        self.history_len = history_len
        self.num_joints = num_joints
        self.num_legs = num_legs
        # Index of the most-recent timestep after reshape to [B, H, num_joints]
        self.newest_step_idx: int = history_len - 1 if history_order == "oldest_first" else 0

        # ── Actor action history window ───────────────────────────────────────
        # The encoder leg tokens carry the full history_len action steps.
        # The actor receives only the last actor_action_history_steps actions
        # (oldest-to-newest in the flattened output) for short-range smoothness.
        if actor_action_history_steps < 1 or actor_action_history_steps > history_len:
            raise ValueError(
                f"[STZMPStudentTeacher] actor_action_history_steps must be in "
                f"[1, history_len={history_len}], got {actor_action_history_steps}."
            )
        self.actor_action_history_steps: int = actor_action_history_steps
        # Slice indices into the [B, H, N] action_hist tensor (always oldest-first slice)
        if history_order == "oldest_first":
            self._actor_act_start: int = history_len - actor_action_history_steps
            self._actor_act_end: int   = history_len
        else:  # newest_first: index 0 is most recent, so take first k steps
            self._actor_act_start = 0
            self._actor_act_end   = actor_action_history_steps

        # ── Policy obs dimension accounting ──────────────────────────────────
        actual_policy_dim: int = sum(
            obs[grp].shape[-1] for grp in obs_groups["policy"]
        )
        prefix_dim = base_lin_vel_dim + base_ang_vel_dim + projected_gravity_dim + command_dim
        history_dim = history_len * num_joints          # one feature type (q, dq, or action)
        total_history_dim = 3 * history_dim             # joint_pos + joint_vel + actions

        height_scan_dim = actual_policy_dim - prefix_dim - total_history_dim
        if height_scan_dim < 0:
            raise ValueError(
                f"[STZMPStudentTeacher] Configured dims exceed actual policy obs size.\n"
                f"  actual_policy_dim = {actual_policy_dim}\n"
                f"  prefix_dim        = {prefix_dim}  "
                f"(base_lin_vel={base_lin_vel_dim} + base_ang_vel={base_ang_vel_dim} + "
                f"projected_gravity={projected_gravity_dim} + command={command_dim})\n"
                f"  total_history_dim = {total_history_dim}  "
                f"(3 × history_len={history_len} × num_joints={num_joints})\n"
                f"  → height_scan_dim would be {height_scan_dim} (must be ≥ 0)\n"
                f"Check that obs term order in Isaac Lab matches the documented layout."
            )
        self.height_scan_dim = height_scan_dim

        # Compute and store flat slice boundaries for each obs term
        s = 0
        self._sl_base_lin_vel    = (s, s + base_lin_vel_dim);       s += base_lin_vel_dim
        self._sl_base_ang_vel    = (s, s + base_ang_vel_dim);       s += base_ang_vel_dim
        self._sl_proj_gravity    = (s, s + projected_gravity_dim);  s += projected_gravity_dim
        self._sl_command         = (s, s + command_dim);            s += command_dim
        self._sl_joint_pos_hist  = (s, s + history_dim);            s += history_dim
        self._sl_joint_vel_hist  = (s, s + history_dim);            s += history_dim
        self._sl_action_hist     = (s, s + history_dim);            s += history_dim
        self._sl_height_scan     = (s, s + height_scan_dim);        s += height_scan_dim

        assert s == actual_policy_dim, (
            f"[STZMPStudentTeacher] Internal slice accounting error: "
            f"accounted for {s}D but obs['policy'] is {actual_policy_dim}D."
        )

        # Apply default hidden dims here to avoid mutable-default-argument issues
        if actor_hidden_dims is None:
            actor_hidden_dims = [512, 256, 128]
        if teacher_hidden_dims is None:
            teacher_hidden_dims = [512, 256, 128]

        # ── Encoder ──────────────────────────────────────────────────────────
        # Each leg token: [q_hist | dq_hist | action_hist] for joints_per_leg joints
        leg_token_dim = history_len * joints_per_leg * 3
        # Task-conditioned base token: gravity(3) + ang_vel(3) + lin_vel(3) + command(5) = 14D
        base_enc_dim = projected_gravity_dim + base_ang_vel_dim + base_lin_vel_dim + command_dim

        # Print layout for the user to verify
        _print_obs_layout(
            actual_policy_dim,
            base_lin_vel_dim, base_ang_vel_dim, projected_gravity_dim, command_dim,
            history_len, num_joints, height_scan_dim, history_order,
            actor_action_history_steps=actor_action_history_steps,
            base_enc_dim=base_enc_dim,
            leg_names=leg_names,
            leg_joint_indices=leg_joint_indices,
        )

        self.encoder = STZMPEncoder(
            num_legs=num_legs,
            leg_token_dim=leg_token_dim,
            base_dim=base_enc_dim,
            d_model=d_model,
            num_heads=num_heads,
            temporal_mlp_width=temporal_mlp_width,
            activation=activation,
        )

        # ── Student actor MLP ────────────────────────────────────────────────
        # Receives: joint_pos_cur(N) + joint_vel_cur(N) + actions_recent(K*N)
        #           + all base terms + command + height_scan + z_sample(2) + logvar_zmp(2)
        # K = actor_action_history_steps (encoder sees full H steps; actor sees short K window)
        actor_input_dim = (
            num_joints                                      # joint_pos_cur
            + num_joints                                    # joint_vel_cur
            + num_joints * actor_action_history_steps       # actions_recent (K steps)
            + base_lin_vel_dim                              # base_lin_vel
            + base_ang_vel_dim                              # base_ang_vel
            + projected_gravity_dim                         # projected_gravity
            + command_dim                                   # foot_position_commands
            + height_scan_dim                               # height_scan
            + 2                                             # z_sample (ZMP latent)
            + 2                                             # logvar_zmp (uncertainty gate)
        )
        self.student_actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        print(
            f"[STZMPStudentTeacher] Student actor: input={actor_input_dim}D  "
            f"hidden={list(actor_hidden_dims)}  output={num_actions}D  "
            f"(actor_action_history_steps={actor_action_history_steps})"
        )

        # Scalar action noise std (same convention as StudentTeacher)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

        # ── Teacher MLP ──────────────────────────────────────────────────────
        num_teacher_obs: int = sum(obs[grp].shape[-1] for grp in obs_groups["teacher"])
        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
        self.teacher.eval()
        print(
            f"[STZMPStudentTeacher] Teacher:       input={num_teacher_obs}D  "
            f"hidden={list(teacher_hidden_dims)}  output={num_actions}D"
        )

        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer: nn.Module = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = nn.Identity()

        print("-" * 60)

    # ── Obs helpers ──────────────────────────────────────────────────────────

    @property
    def _leg_indices(self) -> list[torch.Tensor]:
        """Return the leg joint index tensors (stored as registered buffers)."""
        return [getattr(self, f"_leg_idx_{i}") for i in range(self._num_leg_index_sets)]

    def _get_policy_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[grp] for grp in self.obs_groups["policy"]], dim=-1)

    def _get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[grp] for grp in self.obs_groups["teacher"]], dim=-1)

    def _parse_policy_obs(self, policy_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Slice a flat policy obs tensor into named components.

        Returns a dict with keys:
          ``base_lin_vel``, ``base_ang_vel``, ``projected_gravity``, ``command``,
          ``joint_pos_hist`` ``[B, H, num_joints]``,
          ``joint_vel_hist`` ``[B, H, num_joints]``,
          ``action_hist``    ``[B, H, num_joints]``,
          ``height_scan``,
          ``joint_pos_cur``, ``joint_vel_cur`` — most-recent step only,
          ``actions_recent`` ``[B, actor_action_history_steps * num_joints]`` —
              last ``actor_action_history_steps`` actions flattened oldest-first.
              The encoder leg tokens always carry the full H-step action history.
        """
        B = policy_obs.shape[0]
        H, N = self.history_len, self.num_joints

        def _s(bounds: tuple[int, int]) -> torch.Tensor:
            return policy_obs[:, bounds[0]:bounds[1]]

        joint_pos_hist = _s(self._sl_joint_pos_hist).reshape(B, H, N)
        joint_vel_hist = _s(self._sl_joint_vel_hist).reshape(B, H, N)
        action_hist    = _s(self._sl_action_hist).reshape(B, H, N)
        idx = self.newest_step_idx

        # Last actor_action_history_steps actions, oldest-to-newest, flattened
        a_s, a_e = self._actor_act_start, self._actor_act_end
        actions_recent = action_hist[:, a_s:a_e, :].reshape(B, -1)

        return {
            "base_lin_vel":      _s(self._sl_base_lin_vel),
            "base_ang_vel":      _s(self._sl_base_ang_vel),
            "projected_gravity": _s(self._sl_proj_gravity),
            "command":           _s(self._sl_command),
            "joint_pos_hist":    joint_pos_hist,
            "joint_vel_hist":    joint_vel_hist,
            "action_hist":       action_hist,
            "height_scan":       _s(self._sl_height_scan),
            # Most-recent step of joint histories
            "joint_pos_cur":     joint_pos_hist[:, idx, :],
            "joint_vel_cur":     joint_vel_hist[:, idx, :],
            # Short action window for actor (full window goes to encoder)
            "actions_recent":    actions_recent,
        }

    def _build_leg_tokens(self, parsed: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build the per-leg token tensor for the encoder.

        For each leg, concatenates its joint position, velocity, and action
        histories (all H timesteps) into a single flat vector.

        Returns:
            Shape ``[B, num_legs, H × joints_per_leg × 3]``.
        """
        B = parsed["joint_pos_hist"].shape[0]
        leg_tokens = []
        for idx in self._leg_indices:
            # Each: [B, H, joints_per_leg] → [B, H * joints_per_leg]
            q  = parsed["joint_pos_hist"][:, :, idx].reshape(B, -1)
            dq = parsed["joint_vel_hist"][:, :, idx].reshape(B, -1)
            a  = parsed["action_hist"][:, :, idx].reshape(B, -1)
            leg_tokens.append(torch.cat([q, dq, a], dim=-1))   # [B, H*joints*3]
        return torch.stack(leg_tokens, dim=1)                   # [B, num_legs, token_dim]

    def _build_base_token(self, parsed: dict[str, torch.Tensor]) -> torch.Tensor:
        """Task-conditioned encoder base token.

        Returns ``[projected_gravity | base_ang_vel | base_lin_vel | command]``,
        shape ``[B, 14]`` (with default dim config).  Including the command
        (foot-position target + leg-active flag) makes the cross-attention query
        task-conditioned so the encoder naturally attends to the manipulator leg.
        """
        return torch.cat(
            [
                parsed["projected_gravity"],  # orientation context    (3D)
                parsed["base_ang_vel"],        # rotational dynamics    (3D)
                parsed["base_lin_vel"],        # translational dynamics (3D)
                parsed["command"],             # task goal + leg flag   (5D)
            ],
            dim=-1,
        )

    def _build_actor_input(
        self,
        parsed: dict[str, torch.Tensor],
        z_sample: torch.Tensor,
        logvar_zmp: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate all inputs for the student actor MLP."""
        return torch.cat(
            [
                parsed["joint_pos_cur"],      # current joint positions          (N D)
                parsed["joint_vel_cur"],      # current joint velocities         (N D)
                parsed["actions_recent"],     # recent actions (K steps × N)     (K*N D)
                parsed["base_lin_vel"],       # base linear velocity             (3D)
                parsed["base_ang_vel"],       # base angular velocity            (3D)
                parsed["projected_gravity"],  # projected gravity                (3D)
                parsed["command"],            # foot position command            (5D)
                parsed["height_scan"],        # height scan                      (KD)
                z_sample,                     # ZMP latent sample                (2D)
                logvar_zmp,                   # ZMP log-variance (uncertainty gate)(2D)
            ],
            dim=-1,
        )

    # ── Internal encode helper ───────────────────────────────────────────────

    def _encode(
        self,
        policy_obs: torch.Tensor,
        deterministic: bool = False,
        return_attn_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        parsed       = self._parse_policy_obs(policy_obs)
        leg_tokens   = self._build_leg_tokens(parsed)
        base_current = self._build_base_token(parsed)
        return self.encoder(
            leg_tokens, base_current,
            deterministic=deterministic,
            return_attn_weights=return_attn_weights,
        )

    # ── Distribution properties (required by OnPolicyRunner.log) ─────────────

    @property
    def action_mean(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        """Return current action std. Falls back to the std parameter before the first act() call."""
        if self.distribution is not None:
            return self.distribution.stddev
        return self.std

    @property
    def entropy(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.entropy().sum(dim=-1)

    # ── Public forward methods ───────────────────────────────────────────────

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Stochastic action sample for rollout collection.

        Uses the reparameterisation trick: ``z_sample = mu + σ·ε``.
        Returns a sampled action from the student's action distribution.
        """
        policy_obs   = self._get_policy_obs(obs)
        parsed       = self._parse_policy_obs(policy_obs)
        leg_tokens   = self._build_leg_tokens(parsed)
        base_current = self._build_base_token(parsed)
        mu_zmp, logvar_zmp, z_sample, _ = self.encoder(
            leg_tokens, base_current, deterministic=False
        )
        actor_input = self._build_actor_input(parsed, z_sample, logvar_zmp)
        mean = self.student_actor(actor_input)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)
        assert self.distribution is not None
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Deterministic action for deployment.

        Uses ``mu_zmp`` (no noise injection); sigma² jitter is eliminated.
        Switch to this for hardware deployment (doc section 5.3).
        """
        policy_obs   = self._get_policy_obs(obs)
        parsed       = self._parse_policy_obs(policy_obs)
        leg_tokens   = self._build_leg_tokens(parsed)
        base_current = self._build_base_token(parsed)
        mu_zmp, logvar_zmp, _, _attn = self.encoder(
            leg_tokens, base_current, deterministic=True
        )
        actor_input = self._build_actor_input(parsed, mu_zmp, logvar_zmp)
        return self.student_actor(actor_input)

    def act_with_latent(
        self, obs: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning both the action and the Gaussian ZMP latent.

        Used by :class:`~rsl_rl.algorithms.STZMPDistillation` during the update
        step to compute both the BC loss (on ``actions``) and the NLL loss (on
        ``mu_zmp`` / ``logvar_zmp``).

        Returns:
            Tuple of ``(actions [B, A], mu_zmp [B, 2], logvar_zmp [B, 2])``.
        """
        policy_obs   = self._get_policy_obs(obs)
        parsed       = self._parse_policy_obs(policy_obs)
        leg_tokens   = self._build_leg_tokens(parsed)
        base_current = self._build_base_token(parsed)
        mu_zmp, logvar_zmp, z_sample, _ = self.encoder(
            leg_tokens, base_current, deterministic=False
        )
        actor_input = self._build_actor_input(parsed, z_sample, logvar_zmp)
        actions = self.student_actor(actor_input)
        return actions, mu_zmp, logvar_zmp

    def act_deployment_gated(self, obs: TensorDict) -> torch.Tensor:
        """Confidence-gated deployment action (doc section 5.3).

        Applies ``z_deploy = sigmoid(exp(-logvar)) · mu_zmp`` so the ZMP signal
        shrinks toward zero when the encoder is uncertain (large σ²), preventing
        phantom bracing in free space.
        """
        policy_obs   = self._get_policy_obs(obs)
        parsed       = self._parse_policy_obs(policy_obs)
        leg_tokens   = self._build_leg_tokens(parsed)
        base_current = self._build_base_token(parsed)
        mu_zmp, logvar_zmp, _, _attn = self.encoder(
            leg_tokens, base_current, deterministic=True
        )
        confidence = torch.exp(-logvar_zmp)
        confidence = confidence / (confidence + 1.0)   # sigmoid → [0, 1]
        z_deploy = confidence * mu_zmp
        actor_input = self._build_actor_input(parsed, z_deploy, logvar_zmp)
        return self.student_actor(actor_input)

    def encode_with_attn(
        self, obs: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the encoder and return the ZMP latent **plus** attention weights.

        Intended for logging, paper figures, and on-robot debugging — not for
        training (``need_weights=True`` has a small overhead).

        Returns:
            ``(mu_zmp, logvar_zmp, attn_weights)``

            * ``mu_zmp``       — ``[B, 2]`` mean ZMP shift prediction.
            * ``logvar_zmp``   — ``[B, 2]`` log-variance.
            * ``attn_weights`` — ``[B, 1, num_legs]`` cross-attention scores
              averaged over heads. Entry ``[b, 0, l]`` is the attention weight
              the base token places on leg ``l`` for sample ``b``.
              Useful for: identifying which legs drive bracing, making paper
              figures, or streaming to a debug display on hardware.
        """
        policy_obs   = self._get_policy_obs(obs)
        parsed       = self._parse_policy_obs(policy_obs)
        leg_tokens   = self._build_leg_tokens(parsed)
        base_current = self._build_base_token(parsed)
        mu_zmp, logvar_zmp, _, attn_weights = self.encoder(
            leg_tokens, base_current,
            deterministic=True,
            return_attn_weights=True,
        )
        assert attn_weights is not None   # guaranteed by return_attn_weights=True
        return mu_zmp, logvar_zmp, attn_weights

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        """Run the frozen teacher MLP and return its deterministic action.

        Called during rollout to generate the BC target (privileged actions).
        """
        teacher_obs = self._get_teacher_obs(obs)
        teacher_obs = self.teacher_obs_normalizer(teacher_obs)
        with torch.no_grad():
            return self.teacher(teacher_obs)

    # ── Interface methods (required by Distillation / DistillationRunner) ────

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_states: tuple[HiddenState, HiddenState] = (None, None),
    ) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        pass

    def train(self, mode: bool = True) -> "STZMPStudentTeacher":
        super().train(mode)
        # Teacher must always stay in eval mode regardless of outer train/eval calls
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()
        return self

    def update_normalization(self, obs: TensorDict) -> None:
        """Update teacher obs normalizer statistics (called each env step)."""
        if self.teacher_obs_normalization:
            teacher_obs = self._get_teacher_obs(obs)
            self.teacher_obs_normalizer.update(teacher_obs)

    # ── Checkpoint loading ────────────────────────────────────────────────────

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load parameters, handling two checkpoint formats.

        **RL checkpoint** (keys contain ``"actor."``):
            Remaps ``actor.*`` → teacher MLP, ``actor_obs_normalizer.*`` →
            teacher normalizer.  Student encoder and actor are random-initialised.

        **Distillation checkpoint** (keys contain ``"student_actor."`` or ``"encoder."``):
            Full state loaded; training resumes from saved iteration.

        Returns:
            ``True`` if training resumes (distillation checkpoint),
            ``False`` for a fresh distillation run (RL checkpoint).
        """
        _is_rl_ckpt = any(
            key.startswith("actor.") or key.startswith("actor_obs_normalizer.")
            for key in state_dict
        )
        if _is_rl_ckpt:
            teacher_sd: dict = {}
            teacher_norm_sd: dict = {}
            for key, value in state_dict.items():
                if key.startswith("actor."):
                    teacher_sd[key[len("actor."):]] = value
                elif key.startswith("actor_obs_normalizer."):
                    teacher_norm_sd[key[len("actor_obs_normalizer."):]] = value
            self.teacher.load_state_dict(teacher_sd, strict=strict)
            if teacher_norm_sd:
                self.teacher_obs_normalizer.load_state_dict(teacher_norm_sd, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            print("[STZMPStudentTeacher] Loaded teacher weights from RL checkpoint.")
            return False

        if any("student_actor." in key or "encoder." in key for key in state_dict):
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            print("[STZMPStudentTeacher] Resuming from distillation checkpoint.")
            return True

        raise ValueError(
            "[STZMPStudentTeacher] Unrecognised checkpoint format.\n"
            "  Expected keys containing 'actor.' (RL checkpoint) or\n"
            "  'student_actor.'/'encoder.' (distillation checkpoint)."
        )


# ---------------------------------------------------------------------------
# Module-level debug helper
# ---------------------------------------------------------------------------

def _print_obs_layout(
    total_dim: int,
    base_lin_vel_dim: int,
    base_ang_vel_dim: int,
    projected_gravity_dim: int,
    command_dim: int,
    history_len: int,
    num_joints: int,
    height_scan_dim: int,
    history_order: str,
    actor_action_history_steps: int = 3,
    base_enc_dim: int = 14,
    leg_names: list[str] | None = None,
    leg_joint_indices: list[list[int]] | None = None,
) -> None:
    """Print the policy obs layout table and architecture summary for sanity-checking."""
    H, N = history_len, num_joints
    items: list[tuple[str, int]] = [
        ("base_lin_vel",                      base_lin_vel_dim),
        ("base_ang_vel",                      base_ang_vel_dim),
        ("projected_gravity",                 projected_gravity_dim),
        ("command (foot_position_commands)",  command_dim),
        (f"joint_pos history  [H={H} × N={N}]", H * N),
        (f"joint_vel history  [H={H} × N={N}]", H * N),
        (f"action history     [H={H} × N={N}]", H * N),
        (f"height_scan",                      height_scan_dim),
    ]
    print("=" * 65)
    print("[STZMPStudentTeacher] Policy obs layout validation")
    print(f"  history_order : {history_order}")
    print(f"  {'term':<42} {'dim':>5}  {'[start:end]'}")
    print("  " + "-" * 60)
    offset = 0
    for name, dim in items:
        print(f"  {name:<42} {dim:>5}   [{offset}:{offset + dim}]")
        offset += dim
    status = "✓ OK" if offset == total_dim else f"✗ MISMATCH (got {offset}, expected {total_dim})"
    print("  " + "-" * 60)
    print(f"  Total: {total_dim}D  {status}")
    print("=" * 65)
    print("[STZMPStudentTeacher] Encoder base token (query):")
    print(f"  [projected_gravity | base_ang_vel | base_lin_vel | command] = {base_enc_dim}D")
    print(f"  → cross-attention is task-conditioned (command contains leg flag)")
    print("[STZMPStudentTeacher] Actor action history window:")
    print(f"  Encoder leg tokens : full H={H} steps")
    print(f"  Actor actions input: last {actor_action_history_steps} steps "
          f"(actor_action_history_steps={actor_action_history_steps})")
    if leg_names and leg_joint_indices:
        print("[STZMPStudentTeacher] Leg → joint index mapping (attn_weights order):")
        for i, (name, idxs) in enumerate(zip(leg_names, leg_joint_indices)):
            print(f"  attn_weights[{i}] = {name:<6}  joints {idxs}")
    print("=" * 65)
