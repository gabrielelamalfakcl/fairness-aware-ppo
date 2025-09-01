import numpy as np
from torch.distributions import Categorical
import torch
from typing import List
import torch.nn as nn

from .BasePPOAgent import BasePPOAgent, PolicyNetwork, ValueNetwork


class Controller(nn.Module):
    """
    The controller sample at T the mode z and decides which sub-policy will drive the agent (balance efficiency and fairness).
    Section 3.2 "Learning Fairness in Multi-Agent Systems", Jiang et al.
    """
    def __init__(self, obs_dim: int, k_sub: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, k_sub)
        )
    
    def forward(self, obs: torch.Tensor):
        return self.net(obs)


class FENAgent(BasePPOAgent):
    """
    Fairness-Emergent Network (FEN) agent. This agent uses a hierarchical structure to learn both efficiency and fairness
    in multi-agent systems

    """

    def __init__(self,
                 input_dim: int, # dimension of a single-agent local observation o_t
                 output_dim: int, # # discrete primitive actions
                 extra_obs_dim: int = 0, # not used
                 k_sub: int = 2, # num of sub policies
                 T_macro: int = 25, # macro-time-steps (before resampling)
                 lr: float = 1e-4,
                 entropy_coef: float = .01,
                 value_loss_coef:float = .5,
                 reward_scale: float = 1.0,
                 eps_denom: float = 1e-6,
                 alpha_H: float = 0.05,
                 device: str = "cuda:0",
                 **_):

        super().__init__(input_dim, output_dim, lr, gamma=.99,
                         eps_clip=.1, k_epochs=5, batch_size=256,
                         entropy_coef=entropy_coef, value_loss_coef=value_loss_coef,
                         device=device)

        self.k_sub = k_sub
        self.T_macro = T_macro
        
        # parameters of fair-efficient reward formula (Section 3.1)
        self.c: float = reward_scale
        self.eps: float = eps_denom
        self.alpha_H: float = alpha_H
        
        self.t_since_z = 0 # local clock
        self.z_t = torch.zeros(1, dtype=torch.long, device=device)
        self.z_logp = torch.zeros(1, device=device)
        
        self.last_z: List[int] = []
        self.last_logp_z = torch.zeros(0, device=device)

        # The controller selects a sub-policy z_t
        self.controller = Controller(input_dim, k_sub).to(device) 
        
        # Subpolicies
        self.sub_pi: List[PolicyNetwork] = nn.ModuleList([PolicyNetwork(input_dim, output_dim).to(device) for _ in range(k_sub)])
        self.sub_v : List[ValueNetwork] = nn.ModuleList([ValueNetwork(input_dim).to(device) for _ in range(k_sub)])

        # One optimiser for all policies
        self.optimizer = torch.optim.Adam(
            list(self.controller.parameters()) +
            [p for m in self.sub_pi for p in m.parameters()] +
            [p for m in self.sub_v  for p in m.parameters()],
            lr=lr
        )
                
        self.u_i = torch.zeros(0, device=device) # accumulated utility
        self.u_bar_i = torch.zeros(0, device=device) # consensus avg

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor, memory):
        """
        Selects an action based on the current observation and internal state.
        In training mode, it samples a sub-policy every T_macro steps.
        """
        
        device = obs.device
        N = obs.shape[0]

        # EVALUATION MODE (deterministic, fast)
        if getattr(self, "eval_mode", False):
            # Use controller deterministically
            logits = self.controller(obs)
            z = logits.argmax(dim=-1)
            logp_z = torch.zeros_like(z, dtype=torch.float32)

            # Vectorized action selection
            actions = torch.empty(N, dtype=torch.long, device=device)
            logps = torch.empty(N, dtype=torch.float32, device=device)
            vals = torch.empty(N, dtype=torch.float32, device=device)

            for z_i in range(self.k_sub):
                idx = (z == z_i).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                obs_z = obs[idx]
                pi = self.sub_pi[z_i]
                vnet = self.sub_v[z_i]

                pi_logits = pi(obs_z)
                a = pi_logits.argmax(dim=-1)  # greedy action
                actions[idx] = a
                logps[idx] = torch.zeros_like(a, dtype=torch.float32)
                vals[idx] = vnet(obs_z).squeeze(-1)

            return actions.tolist(), logps, vals

        # TRAINING MODE
        resample = (self.t_since_z == 0) # resampling discrete mode
        if resample:
            logits = self.controller(obs)
            dist_z = Categorical(logits=logits)
            z = dist_z.sample()
            logp_z = dist_z.log_prob(z)
            self.last_z = z.cpu().tolist()
            self.last_logp_z = logp_z.detach().cpu()
        else:
            # Use the same sub-policy for the next T_macro steps
            z = torch.tensor(self.last_z, device=device)
            logp_z = self.last_logp_z.to(device)

        # Store FEN data in memory 
        if memory is not None:
            memory.z_indices.extend(z.cpu().tolist())
            memory.z_logprobs.extend(logp_z.cpu().tolist())
            memory.macro_flags.extend([resample] * N)

        self.last_z = z.cpu().tolist()
        self.t_since_z = (self.t_since_z + 1) % self.T_macro

        # The chosen sub-policy outputs an action to interact with the environment
        acts, logps, vals = [], [], []
        for o, z_i in zip(obs, z):
            pi = self.sub_pi[z_i]
            vnet = self.sub_v[z_i]
            dist = Categorical(pi(o.unsqueeze(0)))
            a = dist.sample()
            acts.append(a.item())
            logps.append(dist.log_prob(a))
            vals.append(vnet(o.unsqueeze(0)))
        return acts, torch.cat(logps), torch.cat(vals).squeeze(-1)


    def update(self, memory, beta_kl: float = 1e-3):
        """
        Performs one PPO optimization step for both the sub-policies and the controller
        """
        
        device = self.device

        # Extract data from memory
        obs = torch.stack(memory.states).to(device)                     
        r_env = torch.tensor(memory.rewards, dtype=torch.float32).to(device)
        acts = (torch.cat(memory.actions).to(device)
                 if isinstance(memory.actions[0], torch.Tensor)
                 else torch.tensor(memory.actions, dtype=torch.long, device=device))
        logp_a_old = (torch.cat(memory.logprobs).to(device)
                      if isinstance(memory.logprobs[0], torch.Tensor) and memory.logprobs[0].dim()>0
                      else torch.stack(memory.logprobs).view(-1).to(device))
        z_ids = torch.tensor(memory.z_indices).to(device)
        logp_z_old = torch.tensor(memory.z_logprobs).to(device)
        macro_mask = torch.tensor(memory.macro_flags, dtype=torch.bool).to(device)

        # sub‑policy PPO updates
        loss_sub = 0.0

        for j in range(self.k_sub):
            idx = (z_ids == j).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            
            # Isolate observations, actions, and old log-probabilities for this sub-policy
            obs_j, acts_j, logp_a_old_j = obs[idx], acts[idx], logp_a_old[idx]

            # Separate objectives for the two subpolicies
            # Maximise only the environmental reward (greedy)
            if j == 0:
                r = r_env[idx].clone()

            else:
                # Diversity sub-policies: exploration of diverse behaviours
                with torch.no_grad():
                    ent = Categorical(self.sub_pi[j](obs_j)).entropy()
                
                # The reward is the log-probability of the controller choosing this sub-policy, plus the entropy bonus
                r = logp_z_old[idx].detach() + self.alpha_H * ent

            # Update the corresponding sub-policy and value network with the calculated reward
            loss_sub += self._ppo_step(self.sub_pi[j], self.sub_v[j], obs_j, acts_j, logp_a_old_j, r)

        # Controller PPO update
        obs_m = obs[macro_mask]
        z_m = z_ids[macro_mask]
        lpz_m = logp_z_old[macro_mask]

        # Build consensus utilities
        eps, c = self.eps, self.c
        if self.u_i.numel() == 0:
            self.u_i = torch.zeros_like(r_env[:self.k_sub])
            self.u_bar_i = torch.zeros_like(self.u_i)

        n_agents = self.u_i.numel()
        T_total = obs.shape[0] // n_agents
        u_i_m = self.u_i.repeat(T_total)[macro_mask]
        u_bar_m = self.u_bar_i.repeat(T_total)[macro_mask]

        util_ratio = u_i_m / (u_bar_m + eps)
        r_hat_m = (u_bar_m / c) / (eps + (util_ratio - 1.).abs())

        ctrl_loss, kl_div = self._controller_ppo(obs_m, z_m, lpz_m, r_hat_m)
        total_loss = loss_sub + ctrl_loss + beta_kl * kl_div

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()
       
    def save_weights(self, path: str):
        """
        Save controller, sub-policies, value nets and optimiser state
        """
        
        torch.save(
            {"controller": self.controller.state_dict(),
                "sub_pi" : [pi.state_dict() for pi in self.sub_pi],
                "sub_v" : [v.state_dict()  for v in self.sub_v ],
                "optimizer" : self.optimizer.state_dict(),
                "k_sub" : self.k_sub,
                "T_macro" : self.T_macro,
            },
            path,
        )
        
    def load_weights(self, path: str, map_location=None):
        """
        Load controller, sub-policies, value nets, and optimiser state
        """
        
        checkpoint = torch.load(path, map_location=map_location)
        self.controller.load_state_dict(checkpoint["controller"])
        if len(checkpoint["sub_pi"]) != len(self.sub_pi):
            raise ValueError("Mismatch in number of sub-policies.")
        for pi, state in zip(self.sub_pi, checkpoint["sub_pi"]):
            pi.load_state_dict(state)
        if len(checkpoint["sub_v"]) != len(self.sub_v):
            raise ValueError("Mismatch in number of value networks.")
        for v, state in zip(self.sub_v, checkpoint["sub_v"]):
            v.load_state_dict(state)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Loaded FEN weights from {path}")

    
    def _ppo_step(self, policy, value, states, acts, logp_old, rewards):
        with torch.no_grad():
            v_pred = value(states).squeeze()
        adv = self.compute_gae(rewards.tolist(), v_pred.tolist())
        disc = (adv + v_pred).detach()

        loss_tot, n_batches = 0, 0
        for _ in range(self.k_epochs):
            for i in range(0, len(states), self.batch_size):
                sl = slice(i, i + self.batch_size)
                dist = Categorical(policy(states[sl]))
                lp = dist.log_prob(acts[sl])
                ratio = torch.exp(lp - logp_old[sl])
                surr = torch.min(ratio * adv[sl], torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * adv[sl])
                policy_loss = -surr.mean()

                v_loss = (value(states[sl]).squeeze() - disc[sl]).pow(2).mean()
                ent = dist.entropy().mean()

                loss = policy_loss + self.value_loss_coef*v_loss - self.entropy_coef*ent
                loss_tot += loss
                n_batches += 1
        return loss_tot / max(n_batches, 1)

    def _controller_ppo(self, obs_m, z_m, lpz_old, r_hat_m):
        logits = self.controller(obs_m)
        dist = Categorical(logits=logits)
        lpz = dist.log_prob(z_m)

        with torch.no_grad():
            v_pred = logits.max(dim=-1).values
        adv = self.compute_gae(r_hat_m.tolist(), v_pred.tolist())

        ratio = torch.exp(lpz - lpz_old)
        surr = torch.min(ratio*adv, torch.clamp(ratio,1-self.eps_clip,1+self.eps_clip)*adv)
        ctrl_loss = -surr.mean()

        probs = dist.probs.mean(0)
        kl_div= (probs * (probs.log() - np.log(1./self.k_sub))).sum()
        return ctrl_loss, kl_div