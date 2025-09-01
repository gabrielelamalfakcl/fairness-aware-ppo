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
    def __init__(self, decay=0.99):
        self.max_value, self.min_value, self.decay = float('-inf'), float('inf'), decay
    def update(self, x):
        self.max_value = max(x, self.max_value * self.decay)
        self.min_value = min(x, self.min_value * self.decay)
    def normalise(self, x):
        value_range = self.max_value - self.min_value
        return (x - self.min_value) / value_range if value_range > 0 else 0.0

class Memory:
    def __init__(self): self.clear_memory()
    def clear_memory(self):
        self.states, self.actions, self.rewards, self.logprobs, self.state_values = [], [], [], [], []

class ActorCritic(nn.Module):
    """
    A single network that contains both the policy (actor) and value (critic) heads
    """
    
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

class PPOAgentMDCSP:
    """
    A PPO Agent capable of handling both Discrete and MultiDiscrete action spaces
    We need MultiDiscrete actions when an agent should take multiple decisions at
    the same time
    PPO Agent with Conditional Statistical Parity (CSP) constraints
    """
    def __init__(self, input_dim, action_space, lr, gamma, eps_clip,
                 k_epochs, batch_size, alpha, beta,
                 value_loss_coef, entropy_coef, device):
        
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
        
        # reward normaliser for legitimate factor (priority)
        self.reward_normaliser_low = Normaliser()
        self.reward_normaliser_medium = Normaliser()
        self.reward_normaliser_high = Normaliser()
        
        self.state_value_normaliser_low = Normaliser()
        self.state_value_normaliser_medium = Normaliser()
        self.state_value_normaliser_high = Normaliser()

    def get_distribution(self, action_logits):
        if self.is_multi_discrete:
            # For MultiDiscrete, split the logits and create a distribution for each dimension
            split_logits = torch.split(action_logits, self.action_space.nvec.tolist(), dim=1)
            return [Categorical(logits=l) for l in split_logits]
        else:
            # For Discrete, it's a single distribution
            return Categorical(logits=action_logits)

    def select_actions(self, state: torch.Tensor, use_greedy: bool = False):
        with torch.no_grad():
            action_logits, state_value = self.policy(state)
        dist = self.get_distribution(action_logits)
        
        if self.is_multi_discrete:
            # For MultiDiscrete, get logits for each dimension
            split_logits = torch.split(action_logits, self.action_space.nvec.tolist(), dim=1)
            if use_greedy:
                # Select the action with the highest probability for each dimension
                action = torch.stack([torch.argmax(l, dim=1) for l in split_logits], dim=1)
            else:
                action = torch.stack([d.sample() for d in dist], dim=1)
            # Log prob calculation remains the same, based on the original distribution
            log_prob = torch.sum(torch.stack([d.log_prob(a) for d, a in zip(dist, action.T)]), dim=0)
        else: # Discrete case
            if use_greedy:
                action = torch.argmax(action_logits, dim=1)
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
        return advantages

    def update(self, memory: Memory, rewards_by_group: dict, state_values_by_group: dict):
        """
        Updates the policy using PPO loss and a CSP fairness penalty.
        """
        
        # PPO setup
        rewards = [r[0] for r in memory.rewards]
        advantages = self.compute_gae(rewards, memory.state_values)
        
        memory_states = torch.stack(memory.states).to(self.device)
        memory_logprobs = torch.stack(memory.logprobs).to(self.device)
        discounted_rewards = advantages + torch.cat(memory.state_values).view(-1).to(self.device)
        
        if self.is_multi_discrete:
            memory_actions = torch.tensor(np.array(memory.actions), dtype=torch.int64).to(self.device)
        else:
            memory_actions = torch.tensor([a.item() for a in memory.actions], dtype=torch.int64).to(self.device)

        # Fairness penalty calculation
        # Alpha (Reward) Penalties
        rew_pen_low,_,_,_ = self.reward_metrics.conditional_statistical_parity(rewards_by_group.get('low_non_impaired',0), rewards_by_group.get('low_impaired',0),0,0)
        rew_pen_med,_,_,_ = self.reward_metrics.conditional_statistical_parity(rewards_by_group.get('medium_non_impaired',0), rewards_by_group.get('medium_impaired',0),0,0)
        rew_pen_high,_,_,_ = self.reward_metrics.conditional_statistical_parity(rewards_by_group.get('high_non_impaired',0), rewards_by_group.get('high_impaired',0),0,0)
        
        self.reward_normaliser_low.update(max(rew_pen_low, 0))
        self.reward_normaliser_medium.update(max(rew_pen_med, 0))
        self.reward_normaliser_high.update(max(rew_pen_high, 0))

        norm_rew_pen = self.reward_normaliser_low.normalise(rew_pen_low) + \
                       self.reward_normaliser_medium.normalise(rew_pen_med) + \
                       self.reward_normaliser_high.normalise(rew_pen_high)
        
        # Beta (State-Value) Penalties
        def get_values_tensor(group_key):
            values = state_values_by_group.get(group_key)
            return torch.cat(values).to(self.device) if values else torch.tensor([0.0], device=self.device)

        sv_pen_low,_,_,_ = self.state_value_metrics.conditional_statistical_parity(
            get_values_tensor('low_non_impaired'), get_values_tensor('low_impaired'),
            torch.tensor([0.0]), torch.tensor([0.0])
        )
        sv_pen_med,_,_,_ = self.state_value_metrics.conditional_statistical_parity(
            get_values_tensor('medium_non_impaired'), get_values_tensor('medium_impaired'),
            torch.tensor([0.0]), torch.tensor([0.0])
        )
        sv_pen_high,_,_,_ = self.state_value_metrics.conditional_statistical_parity(
            get_values_tensor('high_non_impaired'), get_values_tensor('high_impaired'),
            torch.tensor([0.0]), torch.tensor([0.0])
        )

        self.state_value_normaliser_low.update(max(sv_pen_low.item(), 0))
        self.state_value_normaliser_medium.update(max(sv_pen_med.item(), 0))
        self.state_value_normaliser_high.update(max(sv_pen_high.item(), 0))

        norm_sv_pen = self.state_value_normaliser_low.normalise(sv_pen_low.item()) + \
                      self.state_value_normaliser_medium.normalise(sv_pen_med.item()) + \
                      self.state_value_normaliser_high.normalise(sv_pen_high.item())

        combined_csp_penalty = (self.alpha * norm_rew_pen) + (self.beta * norm_sv_pen)
        combined_csp_penalty = torch.tensor(max(combined_csp_penalty, 0), device=self.device, dtype=torch.float32)

        # PPO update
        for _ in range(self.k_epochs):
            for start in range(0, len(memory_states), self.batch_size):
                end = start + self.batch_size
                batch_states, batch_actions, batch_advantages, batch_old_logprobs, batch_discounted_rewards = \
                    memory_states[start:end], memory_actions[start:end], advantages[start:end].detach(), \
                    memory_logprobs[start:end].detach(), discounted_rewards[start:end]

                with autocast():
                    action_logits, predicted_state_values = self.policy(batch_states)
                    predicted_state_values = predicted_state_values.view(-1)
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
                    lambda_csp = standard_ppo_loss.detach() / (combined_csp_penalty + 1e-8)
                    scaled_penalty = lambda_csp * combined_csp_penalty
                    
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