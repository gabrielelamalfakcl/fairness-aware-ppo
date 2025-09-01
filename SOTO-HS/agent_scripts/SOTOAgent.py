import os
import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical
from collections import deque

class Memory:
    def __init__(self):
        self.clear_memory()
    def clear_memory(self):
        self.states, self.actions, self.rewards = [], [], []
        self.logprobs, self.state_values = [], []
        self.stream_ids = []

class SOTOAgent(nn.Module):
    """
    This agent implements the SOTO framework using a shared network (called trunk) and separate heads
    for the self-oriented and team-oriented policies.
    It adheres to the SOTO paper by using α-fairness and passing the self-oriented policy's
    action distribution to the team-oriented policy
    """
    
    def __init__(self, input_dim: int, action_space, lr: float = 5e-4, device: str = "cuda:0", **kwargs):
        super().__init__()
        self.device = device
        self.action_space = action_space
        self.is_multi_discrete = hasattr(action_space, 'nvec')

        # Hyperparameters
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.k_epochs = 5
        self.batch_size = 64
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5

        self.num_players = kwargs.get("num_players", 3)
        self.use_self_oriented_policy = True

        output_dim = int(np.sum(self.action_space.nvec)) if self.is_multi_discrete else self.action_space.n
        hidden_dim = 128

        # 1. Shared trunk for initial feature extraction
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()).to(self.device)

        # 2. Self-oriented Heads (Policy and Value)
        self.self_policy_head = nn.Linear(hidden_dim, output_dim).to(self.device)
        self.self_value_head = nn.Linear(hidden_dim, 1).to(self.device)

        # 3. Team-Oriented Heads (Policy and Value)
        # Input dimension includes the trunk's output + the self-oriented policy's action distribution size
        team_input_dim = hidden_dim + output_dim
        self.team_policy_head = nn.Linear(team_input_dim, output_dim).to(self.device)
        self.team_value_head = nn.Linear(team_input_dim, 1).to(self.device)

        # 4. A single optimizer for all network components
        self.optimizer = torch.optim.Adam([
            {'params': self.trunk.parameters()},
            {'params': self.self_policy_head.parameters()},
            {'params': self.self_value_head.parameters()},
            {'params': self.team_policy_head.parameters()},
            {'params': self.team_value_head.parameters()}
        ], lr=lr)

    def start_episode(self, ep_idx, total_eps, twophase_prop=0.50):
        beta = max(1.0 - ep_idx / (twophase_prop * total_eps), 0.0)
        self.use_self_oriented_policy = np.random.rand() < beta

    def get_distribution(self, policy_logits):
        if self.is_multi_discrete:
            split_logits = torch.split(policy_logits, self.action_space.nvec.tolist(), dim=1)
            return [Categorical(logits=l) for l in split_logits]
        else:
            return Categorical(logits=policy_logits)

    @torch.no_grad()
    def select_action(self, state: torch.Tensor):
        """
        Sample an action
        """
        
        h = self.trunk(state)

        if self.use_self_oriented_policy:
            logits = self.self_policy_head(h)
            value = self.self_value_head(h)
            stream_id = 0
        else:
            self_logits = self.self_policy_head(h)
            self_dist = torch.softmax(self_logits, dim=-1)
            team_in = torch.cat([h, self_dist], dim=-1)

            logits = self.team_policy_head(team_in)
            value = self.team_value_head(team_in)
            stream_id = 1

        dist = self.get_distribution(logits)

        if self.is_multi_discrete:
            action_parts = [d.sample() for d in dist]
            action = torch.stack(action_parts, dim=1)
            logp_parts = [d.log_prob(a) for d, a in zip(dist, action.T)]
            logp = torch.stack(logp_parts).sum()
            action_out = action.squeeze(0).cpu().numpy()
        else:
            action = dist.sample()
            logp = dist.log_prob(action)
            action_out = action.item()

        return action_out, logp.cpu(), value.squeeze().cpu(), stream_id

    def update(self, memory, avg_reward_non_impaired: float, avg_reward_impaired: float, alpha: float, eps: float = 1e-8):
        states = torch.stack(memory.states).to(self.device)
        actions = torch.tensor(np.array(memory.actions), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(self.device)
        old_logp = torch.stack(memory.logprobs).to(self.device)
        values_saved = torch.as_tensor(memory.state_values, dtype=torch.float32, device=self.device)
        stream_ids = torch.tensor(memory.stream_ids).to(self.device)

        idx_self = (stream_ids == 0).nonzero(as_tuple=True)[0]
        idx_team = (stream_ids == 1).nonzero(as_tuple=True)[0]

        loss_self = torch.tensor(0., device=self.device)
        if idx_self.numel():
            h_self = self.trunk(states[idx_self])
            adv_self = self._compute_gae(rewards[idx_self], values_saved[idx_self])
            loss_self = self._ppo_loss(self.self_policy_head, self.self_value_head, h_self,
                                       actions[idx_self], old_logp[idx_self], adv_self)

        loss_team = torch.tensor(0., device=self.device)
        if idx_team.numel():
            h_team = self.trunk(states[idx_team])
            self_logits = self.self_policy_head(h_team)
            self_action_dist = torch.softmax(self_logits, dim=-1)
            team_in = torch.cat([h_team, self_action_dist], dim=-1)

            adv_raw = self._compute_gae(rewards[idx_team], values_saved[idx_team])

            # Patient-centric fairness weighting
            fairness_ratio = 1.0
            if avg_reward_impaired > 0 and avg_reward_non_impaired > 0:
                fairness_ratio = avg_reward_non_impaired / (avg_reward_impaired + eps)

            fairness_ratio_tensor = torch.tensor(fairness_ratio, dtype=torch.float32, device=self.device)

            weights = (fairness_ratio_tensor + eps).pow(-alpha)
            adv_team = weights * adv_raw
            adv_team = (adv_team - adv_team.mean()) / (adv_team.std() + 1e-8)

            loss_team = self._ppo_loss(self.team_policy_head, self.team_value_head, team_in,
                                       actions[idx_team], old_logp[idx_team], adv_team)

        total_loss = loss_self + loss_team
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    def _ppo_loss(self, policy_head, value_head, states, actions, old_logprobs, advantages):
        """
        Calculates the PPO loss for a given set of heads and experiences
        """
        
        with torch.no_grad():
             returns = advantages + value_head(states).squeeze(-1)

        total_loss = 0
        num_batches = 0

        indices = np.arange(len(states))

        for _ in range(self.k_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                s, a, logp_old, adv, ret = (
                    states[batch_indices], actions[batch_indices], old_logprobs[batch_indices],
                    advantages[batch_indices], returns[batch_indices]
                )

                action_logits = policy_head(s)
                dist = self.get_distribution(action_logits)

                if self.is_multi_discrete:
                    logp = torch.sum(torch.stack([d.log_prob(ac.T) for d, ac in zip(dist, a.T)]), dim=0)
                    entropy = torch.sum(torch.stack([d.entropy() for d in dist]), dim=0).mean()
                else:
                    logp = dist.log_prob(a.squeeze(-1))
                    entropy = dist.entropy().mean()


                ratio = torch.exp(logp - logp_old)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = value_head(s)
                value_loss = nn.MSELoss()(value_pred.squeeze(-1), ret)

                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                total_loss += loss
                num_batches += 1

        return total_loss / max(num_batches, 1)


    def _compute_gae(self, rewards, values, lam=0.95):
        adv, gae = [], 0.
        # Ensure rewards and values are 1D arrays for easier processing
        rewards = rewards.squeeze(-1).cpu().numpy()
        values = values.squeeze(-1).cpu().numpy()

        # Append a zero for the terminal state value if it's not present
        next_values = np.append(values[1:], 0)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] - values[t]
            gae = delta + self.gamma * lam * gae
            adv.insert(0, gae)

        adv = torch.tensor(adv, dtype=torch.float32, device=self.device)
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    def get_state_dict(self):
        return {
            "trunk": self.trunk.state_dict(),
            "self_policy_head": self.self_policy_head.state_dict(),
            "self_value_head": self.self_value_head.state_dict(),
            "team_policy_head": self.team_policy_head.state_dict(),
            "team_value_head": self.team_value_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, checkpoint):
        self.trunk.load_state_dict(checkpoint['trunk'])
        self.self_policy_head.load_state_dict(checkpoint['self_policy_head'])
        self.self_value_head.load_state_dict(checkpoint['self_value_head'])
        self.team_policy_head.load_state_dict(checkpoint['team_policy_head'])
        self.team_value_head.load_state_dict(checkpoint['team_value_head'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded SOTO agent weights from checkpoint.")