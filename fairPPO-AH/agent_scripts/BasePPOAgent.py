import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.cuda.amp import GradScaler
import torch.nn.functional as F

"""
Shared functionality: policy and value networks, GAE calculation, action selection, and an abstract update method
"""

scaler = GradScaler()

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)  # This should be 128 to match the input of fc3
        self.fc3 = nn.Linear(128, 64)  # Modified to match the input of fc3
        self.fc4 = nn.Linear(64, output_dim)
        self.output_dim = output_dim
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = torch.softmax(self.fc4(x), dim=-1)  
        return x

class ValueNetwork(nn.Module):
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
        x = self.fc4(x)
        return x

class BasePPOAgent:
    def __init__(self, input_dim, output_dim, lr, gamma, eps_clip, k_epochs, batch_size, entropy_coef=0.01, value_loss_coef=0.1, device='cuda:0'):
        self.policy_net = PolicyNetwork(input_dim, output_dim).float().to(device)
        self.value_net = ValueNetwork(input_dim).float().to(device)
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
        self.loss = None
        self.weights_log = []

    def compute_gae(self, rewards, values, gamma=0.99, lam=0.95):
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device).view(-1)

        # Ensure values is 1D, and append a zero for the next value reference
        values = torch.cat([values, torch.tensor([0.0], dtype=torch.float32).to(self.device)])
        
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] - values[i]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        
        # Stack advantages into a tensor and ensure it's 1D
        advantages = torch.stack(advantages).to(self.device).view(-1)  # Ensure 1D shape
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages

    def select_actions(self, states):
        action_probs = self.policy_net(states)
        dist = Categorical(action_probs)
        actions = dist.sample()
        state_values = self.value_net(states)

        return actions.tolist(), dist.log_prob(actions), state_values

    def update_weights_log(self):
        weights_snapshot = {}
        for name, param in self.policy_net.named_parameters():
            weights_snapshot[name] = param.detach().cpu().numpy().copy()
        for name, param in self.value_net.named_parameters():
            weights_snapshot[name] = param.detach().cpu().numpy().copy()
        self.weights_log.append(weights_snapshot)
        
    def update(self, memory):
        raise NotImplementedError
        