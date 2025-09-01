import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import List

from .BasePPOAgent import BasePPOAgent, PolicyNetwork, ValueNetwork
from fairnessmetrics.FairnessMetrics import CumulativeMetrics

"""
Counterfactual‑FEN (CF‑FEN)
===========================
Same architecture as FEN but the fairness penalty is a **counterfactual gap**
computed from a parallel world (world‑1) versus the main world (world‑0).

The recipe mirrors the final tuning we converged on for `FENAgent`:
* scale *all* fairness terms by the mean magnitude of **controller reward**
  `r_hat_m`.
* intrinsic shaping on ψ₁ only (β_fair = 12).
* λ‑penalty applied per‑macro‑step inside the controller update.
"""


class Controller(nn.Module):
    def __init__(self, obs_dim: int, k_sub: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, k_sub)
        )

    def forward(self, obs: torch.Tensor):
        return self.net(obs)


class FENAgentCF(BasePPOAgent):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        k_sub: int = 2,
        T_macro: int = 10,
        lr: float = 5e-4,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        reward_scale: float = 1.0,
        eps_denom: float = 1e-6,
        alpha_H: float = 0.05,
        device: str = "cuda:0",
        **_,
    ):
        super().__init__(
            input_dim,
            output_dim,
            lr,
            gamma=0.99,
            eps_clip=0.2,
            k_epochs=5,
            batch_size=256,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            device=device,
        )

        self.k_sub = k_sub
        self.T_macro = T_macro
        self.c = reward_scale
        self.eps = eps_denom
        self.alpha_H = alpha_H

        # controller + sub‑policies -------------------------------------------------
        self.controller = Controller(input_dim, k_sub).to(device)
        self.sub_pi: List[PolicyNetwork] = nn.ModuleList(
            [PolicyNetwork(input_dim, output_dim).to(device) for _ in range(k_sub)]
        )
        self.sub_v: List[ValueNetwork] = nn.ModuleList(
            [ValueNetwork(input_dim).to(device) for _ in range(k_sub)]
        )
        self.optimizer = torch.optim.Adam(
            list(self.controller.parameters())
            + [p for m in self.sub_pi for p in m.parameters()]
            + [p for m in self.sub_v for p in m.parameters()],
            lr=lr,
        )

        # misc state ----------------------------------------------------------------
        self.t_since_z = 0
        self.last_z: List[int] = []
        self.last_logp_z = torch.zeros(0, device=device)

        self.u_i = torch.zeros(0, device=device)
        self.u_bar_i = torch.zeros(0, device=device)

        # metric helper (reuse DP fn for generic gap)
        self.F = CumulativeMetrics()

    # ─────────────────────────────────────────────────────────────── select_action ──
    @torch.no_grad()
    def select_action(self, obs: torch.Tensor, memory):
        device = obs.device
        N = obs.size(0)

        resample = self.t_since_z == 0
        if resample:
            logits = self.controller(obs)
            dist_z = Categorical(logits=logits)
            z = dist_z.sample()
            logp_z = dist_z.log_prob(z)
            self.last_z, self.last_logp_z = z.cpu().tolist(), logp_z.detach().cpu()
        else:
            z = torch.tensor(self.last_z, device=device)
            logp_z = self.last_logp_z.to(device)

        if memory is not None:
            memory.z_indices.extend(z.cpu().tolist())
            memory.z_logprobs.extend(logp_z.cpu().tolist())
            memory.macro_flags.extend([resample] * N)

        self.t_since_z = (self.t_since_z + 1) % self.T_macro

        acts, logps, vals = [], [], []
        for o, zi in zip(obs, z):
            dist = Categorical(self.sub_pi[zi](o.unsqueeze(0)))
            a = dist.sample()
            acts.append(a.item())
            logps.append(dist.log_prob(a))
            vals.append(self.sub_v[zi](o.unsqueeze(0)))
        return acts, torch.cat(logps), torch.cat(vals).squeeze(-1)

    # ─────────────────────────────────────────────────────────────────── update ──
    def update(self, mem_main, mem_other, lambda_fair: float, beta_kl: float = 1e-3):
        """Single optimisation update using main‑world rollout `mem_main` and
        counterfactual world `mem_other` to compute a CF gap.
        """
        device = self.device

        # -------- rollout tensors --------------------------------------------------
        obs = torch.stack(mem_main.states).to(device)
        r_env = torch.tensor(mem_main.rewards, dtype=torch.float32).to(device)
        actions = torch.tensor(mem_main.actions, dtype=torch.long, device=device)
        logp_a_old = torch.stack(mem_main.logprobs).view(-1).to(device)
        z_ids = torch.tensor(mem_main.z_indices).to(device)
        logp_z_old = torch.tensor(mem_main.z_logprobs).to(device)
        macro_mask = torch.tensor(mem_main.macro_flags, dtype=torch.bool).to(device)

        # ───── sub‑policy updates (ψ₀ efficiency, ψ₁ fairness) ─────────────────––
        beta_fair = 12.0  # intrinsic weight tuned in smoke test
        gap_cf = 0.0
        if mem_other is not None:
            gap_cf = self._fairness_gap_estimate(sum(mem_main.rewards), sum(mem_other.rewards))
        step_scale_env = r_env.abs().mean().detach()

        loss_sub = 0.0
        for j in range(self.k_sub):
            idx = (z_ids == j).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            r = r_env[idx].clone()
            r += logp_z_old[idx].detach()  # diversity bonus to all
            with torch.no_grad():
                r += self.alpha_H * Categorical(self.sub_pi[j](obs[idx])).entropy()
            if j == 1:  # fairness‑oriented ψ₁
                r -= beta_fair * gap_cf * step_scale_env
            loss_sub += self._ppo_step(self.sub_pi[j], self.sub_v[j], obs[idx], actions[idx], logp_a_old[idx], r)

        # ───── controller update (gossip + λ penalty) ─────────────────────────────
        eps, c = self.eps, self.c
        if self.u_i.numel() == 0:
            self.u_i = torch.zeros(1, device=device)
            self.u_bar_i = torch.zeros_like(self.u_i)

        n_agents = self.u_i.numel()
        T_total = obs.size(0) // n_agents
        u_i_rep = self.u_i.repeat(T_total)
        u_bar_rep = self.u_bar_i.repeat(T_total)
        u_i_m, u_bar_m = u_i_rep[macro_mask], u_bar_rep[macro_mask]

        util_ratio = u_i_m / (u_bar_m + eps)
        r_hat_m = (u_bar_m / c) / (eps + (util_ratio - 1).abs())
        step_scale_ctrl = r_hat_m.mean().detach()
        r_hat_m -= lambda_fair * gap_cf * step_scale_ctrl

        obs_m, z_m, lpz_m = obs[macro_mask], z_ids[macro_mask], logp_z_old[macro_mask]
        ctrl_loss, kl_div = self._controller_ppo(obs_m, z_m, lpz_m, r_hat_m)

        total_loss = loss_sub + ctrl_loss + beta_kl * kl_div
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    # -----------------------------------------------------------------------------
    def _fairness_gap_estimate(self, r_a: float, r_b: float) -> float:
        _, gap = self.F.demographic_parity(r_a, r_b)
        return float(gap)

    # ───────────────────────────── helper: PPO step for sub‑policies ──
    def _ppo_step(self, policy, value, states, acts, logp_old, rewards):
        """Standard clipped PPO objective with value and entropy bonuses."""
        with torch.no_grad():
            v_pred = value(states).squeeze()
        adv = self.compute_gae(rewards.tolist(), v_pred.tolist())
        disc = (adv + v_pred).detach()

        loss_tot, n_batches = 0.0, 0
        for _ in range(self.k_epochs):
            for i in range(0, len(states), self.batch_size):
                sl = slice(i, i + self.batch_size)
                dist = Categorical(policy(states[sl]))
                lp   = dist.log_prob(acts[sl])
                ratio= torch.exp(lp - logp_old[sl])
                surr = torch.min(ratio * adv[sl],
                                 torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * adv[sl])
                policy_loss = -surr.mean()

                v_loss = (value(states[sl]).squeeze() - disc[sl]).pow(2).mean()
                ent    = dist.entropy().mean()
                loss   = policy_loss + self.value_loss_coef * v_loss - self.entropy_coef * ent
                loss_tot += loss
                n_batches += 1
        return loss_tot / max(n_batches, 1)

    # ───────────────────────────── helper: PPO step for controller ────
    def _controller_ppo(self, obs_m, z_m, lpz_old, r_hat_m):
        """PPO update for the high‑level controller."""
        logits = self.controller(obs_m)
        dist   = Categorical(logits=logits)
        lpz    = dist.log_prob(z_m)

        # crude value target = max‑logit (empirically ok)
        with torch.no_grad():
            v_pred = logits.max(dim=-1).values
        adv = self.compute_gae(r_hat_m.tolist(), v_pred.tolist())

        ratio = torch.exp(lpz - lpz_old)
        surr  = torch.min(ratio * adv,
                          torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * adv)
        ctrl_loss = -surr.mean()

        # KL(pθ ‖ Uniform)
        probs = dist.probs.mean(0)
        kl_div = (probs * (probs.log() - np.log(1.0 / self.k_sub))).sum()
        return ctrl_loss, kl_div

    # ───────────────────────────────────── saving / loading weights ───
    def save_weights(self, path: str):
        """Persist controller, sub‑policy nets, value nets, and optimiser."""
        torch.save(
            {
                "controller": self.controller.state_dict(),
                "sub_pi"    : [pi.state_dict() for pi in self.sub_pi],
                "sub_v"     : [v.state_dict()  for v in self.sub_v],
                "optimizer" : self.optimizer.state_dict(),
                "k_sub"     : self.k_sub,
                "T_macro"   : self.T_macro,
            },
            path,
        )

    def load_weights(self, path: str, map_location=None):
        """Restore weights & optimiser state from path."""
        ckpt = torch.load(path, map_location=map_location)
        self.controller.load_state_dict(ckpt["controller"])

        if len(ckpt["sub_pi"]) != len(self.sub_pi):
            raise ValueError("Mismatch in number of sub‑policies.")
        for pi, st in zip(self.sub_pi, ckpt["sub_pi"]):
            pi.load_state_dict(st)

        if len(ckpt["sub_v"]) != len(self.sub_v):
            raise ValueError("Mismatch in number of value networks.")
        for v, st in zip(self.sub_v, ckpt["sub_v"]):
            v.load_state_dict(st)

        self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Loaded CF‑FEN weights from {path}")
