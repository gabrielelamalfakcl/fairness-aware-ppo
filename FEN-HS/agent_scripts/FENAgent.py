import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class Memory:
    def __init__(self):
        self.clear_memory()

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.state_values = []

class ActorCritic(nn.Module):    
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

class FENAgent:
    def __init__(self, input_dim, action_space, lr, gamma, eps_clip, k_epochs,
                 batch_size, num_sub_policies, value_loss_coef, entropy_coef, device):

        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # Controller
        self.controller = ActorCritic(input_dim, num_sub_policies).to(device)
        self.controller_optimizer = optim.Adam(self.controller.parameters(), lr=lr)
        self.controller_memory = Memory()

        # Sub-policies
        self.num_sub_policies = num_sub_policies
        self.sub_policies = []
        self.sub_policy_optimizers = []
        self.sub_policy_memories = []

        self.is_multi_discrete = hasattr(action_space, 'nvec')
        if self.is_multi_discrete:
            sub_policy_output_dim = int(np.sum(action_space.nvec))
        else:
            sub_policy_output_dim = action_space.n
            
        self.action_space = action_space

        for _ in range(num_sub_policies):
            sub_policy = ActorCritic(input_dim, sub_policy_output_dim).to(device)
            optimizer = optim.Adam(sub_policy.parameters(), lr=lr)
            self.sub_policies.append(sub_policy)
            self.sub_policy_optimizers.append(optimizer)
            self.sub_policy_memories.append(Memory())
    
    def _get_distribution(self, policy, action_logits):
        if policy in self.sub_policies and self.is_multi_discrete:
            split_logits = torch.split(action_logits, self.action_space.nvec.tolist(), dim=1)
            return [Categorical(logits=l) for l in split_logits]
        else:
            return Categorical(logits=action_logits)

    def select_sub_policy(self, state):
        with torch.no_grad():
            logits, state_value = self.controller(state)
        
        dist = self._get_distribution(self.controller, logits)
        sub_policy_index = dist.sample()
        log_prob = dist.log_prob(sub_policy_index)
        
        return sub_policy_index.item(), log_prob, state_value

    def select_action(self, sub_policy_index, state):
        sub_policy = self.sub_policies[sub_policy_index]
        with torch.no_grad():
            action_logits, state_value = sub_policy(state)
        
        dist = self._get_distribution(sub_policy, action_logits)
        
        if self.is_multi_discrete:
            action = torch.stack([d.sample() for d in dist], dim=1)
            log_prob = torch.sum(torch.stack([d.log_prob(a) for d, a in zip(dist, action.T)]), dim=0)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.cpu().numpy(), log_prob.detach(), state_value.detach()
            
            
    def update(self, policy_type, policy_index=None):
        if policy_type == 'controller':
            memory = self.controller_memory
            policy = self.controller
            optimizer = self.controller_optimizer
        elif policy_type == 'sub_policy':
            memory = self.sub_policy_memories[policy_index]
            policy = self.sub_policies[policy_index]
            optimizer = self.sub_policy_optimizers[policy_index]
        else:
            raise ValueError("Invalid policy_type")

        # GAE
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(self.device)
        state_values = torch.cat(memory.state_values).to(self.device).view(-1)
        
        advantages = []
        gae = 0
        # GAE for controller is simpler as it's updated less frequently
        if policy_type == 'controller':
             returns = []
             discounted_reward = 0
             for reward, is_terminal in zip(reversed(memory.rewards), reversed([False]*len(memory.rewards))):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                returns.insert(0, discounted_reward)
             returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
             advantages = returns - state_values.detach()

        else: # GAE for sub-policies
            last_value = state_values[-1].clone()
            for i in reversed(range(len(rewards))):
                delta = rewards[i] + self.gamma * last_value - state_values[i]
                gae = delta + self.gamma * 0.95 * gae
                advantages.insert(0, gae)
                last_value = state_values[i]

            advantages = torch.stack(advantages).to(self.device).view(-1)
            returns = advantages + state_values
        
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()

        if policy in self.sub_policies and self.is_multi_discrete:
            old_actions = torch.tensor(np.array(memory.actions), dtype=torch.int64).to(self.device)
        else:
            old_actions = torch.tensor(memory.actions, dtype=torch.int64).to(self.device)


        for _ in range(self.k_epochs):
            for i in range(0, len(old_states), self.batch_size):
                
                batch_states = old_states[i:i+self.batch_size]
                batch_actions = old_actions[i:i+self.batch_size]
                batch_logprobs = old_logprobs[i:i+self.batch_size]
                batch_advantages = advantages[i:i+self.batch_size]
                batch_returns = returns[i:i+self.batch_size]
                
                logits, state_values_pred = policy(batch_states)
                dist = self._get_distribution(policy, logits)

                if policy in self.sub_policies and self.is_multi_discrete:
                    logprobs = torch.sum(torch.stack([d.log_prob(a) for d, a in zip(dist, batch_actions.T)]), dim=0)
                    dist_entropy = torch.sum(torch.stack([d.entropy() for d in dist]), dim=0)
                else:
                    logprobs = dist.log_prob(batch_actions)
                    dist_entropy = dist.entropy()
                    
                ratios = torch.exp(logprobs - batch_logprobs)
                
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                
                value_loss = nn.MSELoss()(state_values_pred.view(-1), batch_returns)
                loss = -torch.min(surr1, surr2).mean() + self.value_loss_coef * value_loss - self.entropy_coef * dist_entropy.mean()
                       
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
        
        memory.clear_memory()
        return loss.item()

    def get_state_dict(self):
        state_dict = {
            'controller': self.controller.state_dict(),
            'controller_optimizer': self.controller_optimizer.state_dict(),
        }
        for i in range(self.num_sub_policies):
            state_dict[f'sub_policy_{i}'] = self.sub_policies[i].state_dict()
            state_dict[f'sub_policy_optimizer_{i}'] = self.sub_policy_optimizers[i].state_dict()
        return state_dict

    def load_state_dict(self, checkpoint):
        self.controller.load_state_dict(checkpoint['controller'])
        self.controller_optimizer.load_state_dict(checkpoint['controller_optimizer'])
        for i in range(self.num_sub_policies):
            self.sub_policies[i].load_state_dict(checkpoint[f'sub_policy_{i}'])
            self.sub_policy_optimizers[i].load_state_dict(checkpoint[f'sub_policy_optimizer_{i}'])
