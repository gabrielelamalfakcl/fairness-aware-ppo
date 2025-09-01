import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.cuda.amp import autocast, GradScaler
import numpy as np

scaler = GradScaler()

class PolicyNetwork(nn.Module):
    """
    Neural network for policy approximation
    Outputs action probabilities
    """

    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return torch.softmax(self.fc4(x), dim=-1)  # Action probabilities


class ValueNetwork(nn.Module):
    """
    Neural network for state-value estimation
    Outputs a single value representing the expected return
    """

    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)  # Single value output

class BasePPOAgent:
    """
    Base class for PPO agents handling shared logic like GAE calculation and action selection
    """

    def __init__(
        self, input_dim, output_dim, lr, gamma, eps_clip, k_epochs, batch_size, entropy_coef=0.01, value_loss_coef=0.1, device='cuda:0'):
        self.policy_net = PolicyNetwork(input_dim, output_dim).to(device).float()
        self.value_net = ValueNetwork(input_dim).to(device).float()

        self.optimizer = optim.Adam([
            {'params': self.policy_net.parameters()},
            {'params': self.value_net.parameters()}
        ], lr=lr)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.device = device
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.weights_log = []

    def compute_gae(self, rewards, values, gamma=0.99, lam=0.95):
        """
        Computes Generalized Advantage Estimation (GAE).
        """
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device).view(-1)

        # Extend values with a zero at the end for last value reference
        values = torch.cat([values, torch.tensor([0.0], dtype=torch.float32).to(self.device)])

        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] - values[i]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)

        advantages = torch.stack(advantages).to(self.device).view(-1)
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalized advantages

    def select_actions(self, states):
        """
        Selects actions based on the current policy.
        """
        action_probs = self.policy_net(states)
        dist = Categorical(action_probs)
        actions = dist.sample()
        state_values = self.value_net(states)

        return actions.tolist(), dist.log_prob(actions), state_values

    def update_weights_log(self):
        """
        Logs policy and value network weights for monitoring training stability.
        """
        weights_snapshot = {}
        for name, param in self.policy_net.named_parameters():
            weights_snapshot[name] = param.detach().cpu().numpy().copy()
        for name, param in self.value_net.named_parameters():
            weights_snapshot[name] = param.detach().cpu().numpy().copy()
        self.weights_log.append(weights_snapshot)

    def update(self, memory):
        """
        Abstract update method to be implemented in derived classes.
        """
        raise NotImplementedError


class PPOAgent(BasePPOAgent):
    """
    Proximal Policy Optimization (PPO) agent implementing the standard PPO update step.
    """

    def update(self, memory):
        """
        Updates the policy using PPO loss.
        """
        total_loss = 0
        num_updates = 0

        # Convert memory to tensors
        memory_states = torch.stack(memory.states).to(self.device)
        memory_actions = torch.tensor(np.array(memory.actions)).to(self.device)
        memory_rewards = torch.tensor(np.array(memory.rewards), dtype=torch.float32).to(self.device)
        memory_logprobs = torch.stack(memory.logprobs).to(self.device)
        memory_state_values = torch.stack(memory.state_values).to(self.device)

        # Compute advantages and discounted rewards
        advantages = self.compute_gae(memory_rewards.tolist(), memory_state_values.squeeze(-1).tolist())
        discounted_rewards = advantages + memory_state_values.squeeze(-1).detach()

        for _ in range(self.k_epochs):
            for start in range(0, len(memory_states), self.batch_size):
                end = start + self.batch_size
                batch_states = memory_states[start:end]
                batch_actions = memory_actions[start:end]
                batch_advantages = advantages[start:end].detach()
                batch_old_logprobs = memory_logprobs[start:end]
                batch_state_values = memory_state_values[start:end]
                batch_discounted_rewards = discounted_rewards[start:end]

                if batch_state_values.size(0) == 0 or batch_states.size(0) == 0:
                    continue

                # Normalize discounted rewards
                batch_discounted_rewards = (batch_discounted_rewards - batch_discounted_rewards.mean()) / \
                                           (batch_discounted_rewards.std() + 1e-8)

                with autocast():
                    probs = self.policy_net(batch_states)
                    dist = Categorical(probs)
                    logprobs = dist.log_prob(batch_actions)
                    dist_entropy = dist.entropy()
                    predicted_state_values = self.value_net(batch_states).squeeze()

                    ratios = torch.exp(logprobs - batch_old_logprobs)
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

                    mse_loss = nn.MSELoss()
                    value_loss = mse_loss(predicted_state_values, batch_discounted_rewards)
                    policy_loss = -torch.min(surr1, surr2).mean()
                    entropy_loss = -self.entropy_coef * dist_entropy.mean()

                    loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
                    total_loss += loss
                    num_updates += 1

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

        return total_loss / num_updates if num_updates > 0 else 0
