import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import sys

# Path to import other scripts: it needs to be checked based on the folder structure
project_root = Path(__file__).resolve().parents[1] 
sys.path.append(str(project_root))

from agent_scripts.FairnessMetrics import CumulativeMetrics, StateValueMetrics

class Normaliser:
    """Helper class to normalize values within a dynamic range."""
    def __init__(self, decay: float = 0.99):
        self.max_value, self.min_value, self.decay = float('-inf'), float('inf'), decay
    def update(self, x: float):
        self.max_value = max(x, self.max_value * self.decay)
        self.min_value = min(x, self.min_value * self.decay)
    def normalise(self, x: float) -> float:
        value_range = self.max_value - self.min_value
        return (x - self.min_value) / value_range if value_range > 0 else 0.0

class Memory:
    """A buffer for storing experiences for PPO."""
    def __init__(self): self.clear_memory()
    def clear_memory(self):
        self.states, self.actions, self.rewards, self.logprobs, self.state_values = [], [], [], [], []

class ActorCritic(nn.Module):
    """A unified network for both policy (actor) and value (critic) estimation."""
    def __init__(self, input_dim: int, output_dim: int):
        super(ActorCritic, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(),
            nn.Linear(256, 128), nn.LeakyReLU()
        )
        self.actor_head = nn.Linear(128, output_dim)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        shared_features = self.shared_net(x)
        action_logits = self.actor_head(shared_features)
        state_value = self.critic_head(shared_features)
        return action_logits, state_value

class PPOAgentMDCF:
    """
    PPO Agent for a TWO-POLICY Counterfactual Fairness setup.
    The update uses its own memory for the PPO loss and the other world's
    memory for the fairness penalty.
    """
    def __init__(self, input_dim: int, action_space, lr: float, gamma: float, eps_clip: float,
                 k_epochs: int, batch_size: int, alpha: float, beta: float,
                 value_loss_coef: float, entropy_coef: float, device: str):
        
        self.action_space = action_space
        self.is_multi_discrete = hasattr(action_space, 'nvec')

        if self.is_multi_discrete:
            output_dim = int(np.sum(self.action_space.nvec))
        else:
            output_dim = self.action_space.n

        self.policy = ActorCritic(input_dim, output_dim).to(device).float()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma, self.eps_clip, self.k_epochs = gamma, eps_clip, k_epochs
        self.batch_size, self.device = batch_size, device
        self.alpha, self.beta = alpha, beta
        self.value_loss_coef, self.entropy_coef = value_loss_coef, entropy_coef

        self.scaler = GradScaler()
        self.reward_metrics = CumulativeMetrics()
        self.state_value_metrics = StateValueMetrics()
        self.reward_normaliser = Normaliser()
        self.state_value_normaliser = Normaliser()

    def get_distribution(self, action_logits):
        if self.is_multi_discrete:
            split_logits = torch.split(action_logits, self.action_space.nvec.tolist(), dim=1)
            return [Categorical(logits=l) for l in split_logits]
        else:
            return Categorical(logits=action_logits)

    def select_actions(self, state: torch.Tensor):
        with torch.no_grad():
            action_logits, state_value = self.policy(state)
        dist = self.get_distribution(action_logits)
        
        if self.is_multi_discrete:
            action = torch.stack([d.sample() for d in dist], dim=1)
            log_prob = torch.sum(torch.stack([d.log_prob(a) for d, a in zip(dist, action.T)]), dim=0)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.cpu().numpy(), log_prob.detach(), state_value.detach()
    
    def compute_gae(self, rewards: list, values: list) -> torch.Tensor:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        values = torch.cat(values).to(self.device).view(-1)
        values = torch.cat([values, torch.tensor([0.0], dtype=torch.float32).to(self.device)])
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] - values[i]
            gae = delta + self.gamma * 0.95 * gae
            advantages.insert(0, gae)
        advantages = torch.stack(advantages).to(self.device).view(-1)
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def update(self, memory_self: Memory, memory_other: Memory):
        # 1. Calculate CF Fairness Penalty by comparing the two worlds
        total_rewards_self = sum(r[0] for r in memory_self.rewards)
        total_rewards_other = sum(r[0] for r in memory_other.rewards)
        reward_penalty, _ = self.reward_metrics.demographic_parity(total_rewards_self, total_rewards_other)
        self.reward_normaliser.update(abs(reward_penalty))
        norm_rew_pen = self.reward_normaliser.normalise(abs(reward_penalty))

        state_values_self = torch.cat(memory_self.state_values)
        state_values_other = torch.cat(memory_other.state_values)
        sv_penalty, _ = self.state_value_metrics.demographic_parity(state_values_self, state_values_other)
        self.state_value_normaliser.update(abs(sv_penalty.item()))
        norm_sv_pen = self.state_value_normaliser.normalise(abs(sv_penalty.item()))

        combined_cf_penalty = (self.alpha * norm_rew_pen) + (self.beta * norm_sv_pen)
        combined_cf_penalty = torch.tensor(max(combined_cf_penalty, 0), device=self.device, dtype=torch.float32)

        # 2. Perform PPO update using ONLY this agent's own experience from `memory_self`
        rewards = [r[0] for r in memory_self.rewards]
        advantages = self.compute_gae(rewards, memory_self.state_values)
        
        memory_states = torch.stack(memory_self.states).to(self.device)
        memory_logprobs = torch.stack(memory_self.logprobs).to(self.device)
        discounted_rewards = advantages + torch.cat(memory_self.state_values).view(-1).to(self.device)
        
        if self.is_multi_discrete:
            memory_actions = torch.tensor(np.array(memory_self.actions), dtype=torch.int64).to(self.device)
        else:
            memory_actions = torch.tensor([a.item() for a in memory_self.actions], dtype=torch.int64).to(self.device)
        
        # 3. PPO Update Loop on `memory_self`
        for _ in range(self.k_epochs):
            for start in range(0, len(memory_states), self.batch_size):
                end = start + self.batch_size
                batch_states, batch_actions, batch_advantages, batch_old_logprobs, batch_discounted_rewards = \
                    memory_states[start:end], memory_actions[start:end], advantages[start:end].detach(), \
                    memory_logprobs[start:end].detach(), discounted_rewards[start:end]

                with autocast():
                    action_logits, predicted_state_values = self.policy(batch_states)
                    predicted_state_values = predicted_state_values.squeeze()
                    dist = self.get_distribution(action_logits)

                    if self.is_multi_discrete:
                        logprobs = torch.sum(torch.stack([d.log_prob(a) for d, a in zip(dist, batch_actions.T)]), dim=0)
                        dist_entropy = torch.sum(torch.stack([d.entropy() for d in dist]), dim=0)
                    else:
                        logprobs = dist.log_prob(batch_actions)
                        dist_entropy = dist.entropy()
                    
                    ratios = torch.exp(logprobs - batch_old_logprobs)
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                    
                    value_loss = nn.MSELoss()(predicted_state_values, batch_discounted_rewards)
                    policy_loss = -torch.min(surr1, surr2).mean()
                    entropy_loss = -self.entropy_coef * dist_entropy.mean()
                    
                    standard_ppo_loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
                    lambda_cf = standard_ppo_loss.detach() / (combined_cf_penalty + 1e-8)
                    scaled_penalty = lambda_cf * combined_cf_penalty
                    total_loss = standard_ppo_loss + scaled_penalty

                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

    def save_weights(self, file_path):
        torch.save(self.policy.state_dict(), file_path)

    def load_weights(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Weights loaded from {path}")
        
    def get_state_dict(self):
        """Returns the state dictionaries of the agent's networks."""
        return {
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def load_state_dict(self, checkpoint):
        """Loads agent state from a dictionary."""
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded agent weights from checkpoint.")