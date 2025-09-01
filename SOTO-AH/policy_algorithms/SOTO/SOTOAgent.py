import numpy as np
from torch.distributions import Categorical
import torch
import torch.nn as nn
from collections import deque

from .BasePPOAgent import BasePPOAgent, PolicyNetwork, ValueNetwork

class SOTOAgent(BasePPOAgent):
    """
    Two subpolicies and two sub-networks: one self-oriented (maximises the agent’s own return ), 
    one team-oriented (maximises the social-welfare function)
    This allows the agent to first learn to be self-concerned before learning fair behavior.
    """
    
    def __init__(self, input_dim:  int, output_dim: int, extra_obs_dim: int, lr: float, num_players: int, **base_kwargs):
        self.extra_obs_dim = extra_obs_dim
        self.running_avg_return = [deque([0.0], maxlen=100) for _ in range(num_players)]

        super().__init__(input_dim, output_dim, lr=lr, **base_kwargs)
        
        hidden = 128
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
        ).to(self.device)

        self.self_head = PolicyNetwork(hidden, output_dim).to(self.device)
        self.team_head = PolicyNetwork(hidden + extra_obs_dim, output_dim).to(self.device)
        
        self.self_policy = self.self_head
        self.team_policy = self.team_head

        self.self_value = ValueNetwork(hidden).to(self.device)
        self.team_value = ValueNetwork(hidden + extra_obs_dim).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            [{'params': self.trunk.parameters()},
             {'params': self.self_head.parameters()},
             {'params': self.team_head.parameters()},
             {'params': self.self_value.parameters()},
             {'params': self.team_value.parameters()}],
            lr=lr)
        
    def start_episode(self, episode_idx: int, total_episodes: int, twophase_prop: float = 0.30):
        """
        Implement  the beta-switch that decides which stream will act this episode
        Early in training agents explore the selfish policy, while later they use the team-oriented one 
        Section 4.3, Learning Fair Policies in Decentralized Cooperative Multi-Agent Reinforcement Learning, Zimmer et al.
        """
        
        beta = max(1.0 - episode_idx / (twophase_prop * total_episodes), 0.0)
        self.use_greedy = (np.random.rand() < beta)

    def select_action(self, state: torch.Tensor, extra_obs: torch.Tensor):
        """
        Returns: action (int), log_prob (Tensor), state_value (Tensor), stream_id (0=self,1=team)
        """
        
        if self.use_greedy:
            h = self.trunk(state)
            probs = self.self_head(h)
            value = self.self_value(h)
            stream_id = 0
        else:
            # team policy
            h = self.trunk(state)
            team_in = torch.cat([h, extra_obs], dim=-1)
            probs = self.team_head(team_in)
            value = self.team_value(team_in)
            stream_id = 1

        dist = Categorical(probs)
        actions = dist.sample()
        logp = dist.log_prob(actions)
        values = value

        return actions.cpu().tolist(), logp.detach().cpu(), values.detach().cpu(), [stream_id]*len(actions)    
    
    def update(self, memory, alpha: float, epsilon=1e-8):
        """
        Update loss self and loss team.
        """
        
        device = self.device
        # Define tensors
        states = torch.stack(memory.states).to(device)
        extra = torch.stack(memory.extra_obs).to(device)
        actions = torch.tensor(memory.actions, dtype=torch.long).to(device)
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(device)
        old_logp = torch.stack(memory.logprobs).to(device)
        values_saved = torch.as_tensor(memory.state_values, dtype=torch.float32, device=device)
        stream_ids = torch.tensor(memory.stream_ids).to(device) # ids to sleect the streams
        player_ids = torch.tensor(memory.player_ids, device=device)

        # Two streams: self and team oriented
        idx_self = (stream_ids == 0).nonzero(as_tuple=True)[0]
        idx_team = (stream_ids == 1).nonzero(as_tuple=True)[0]

        # Self stream
        loss_self = torch.tensor(0., device=device)
        if idx_self.numel():

            # Hidden features for the self-oriented head
            h_self = self.trunk(states[idx_self])

            A_self = self.compute_gae(
                rewards[idx_self].tolist(),
                values_saved[idx_self].tolist())
            
            loss_self = self._ppo_clipped(
                net_policy=self.self_policy,
                net_value=self.self_value,
                states=h_self,
                extra_obs=None,
                actions=actions[idx_self],
                old_logp=old_logp[idx_self],
                discounted=A_self + values_saved[idx_self].detach(),
                advantages=A_self)

        # Team stream
        loss_team = torch.tensor(0., device=device)
        if idx_team.numel():
            h_team = self.trunk(states[idx_team])
            team_in = torch.cat([h_team, extra[idx_team]], dim=-1)
            A_raw = self.compute_gae(rewards[idx_team].tolist(),
                                       values_saved[idx_team].tolist())

            # alpha-fair weighting
            team_player_ids = player_ids[idx_team].cpu().tolist()
            j_list = [np.mean(self.running_avg_return[p]) for p in team_player_ids]
            j_pi   = torch.as_tensor(j_list, dtype=torch.float32, device=device)

            w = (j_pi + epsilon) ** (-alpha)
            A_team = w * A_raw # element-wise weight
            A_team = (A_team - A_team.mean()) / (A_team.std()+1e-8)

            loss_team = self._ppo_clipped(
                net_policy=self.team_policy,
                net_value=self.team_value,
                states=team_in,
                extra_obs=None,
                actions=actions[idx_team],
                old_logp=old_logp[idx_team],
                discounted=A_team + values_saved[idx_team].detach(),
                advantages=A_team)

        total_loss = loss_self + loss_team
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()
    
    def _ppo_clipped(self, net_policy, net_value, states, extra_obs, actions, old_logp, discounted, advantages):
        B = states.size(0)
        loss_tot = 0
        n_batches = 0
        for _ in range(self.k_epochs):
            for start in range(0, B, self.batch_size):
                end = start + self.batch_size
                s = states[start:end]
                a = actions[start:end]
                adv = advantages[start:end].detach()
                logp_old = old_logp[start:end]
                disc_rew = discounted[start:end]

                # forward
                probs = net_policy(s)
                dist  = Categorical(probs)
                logp  = dist.log_prob(a)
                ratio = torch.exp(logp - logp_old)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio,
                                    1-self.eps_clip,
                                    1+self.eps_clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                values = net_value(s).squeeze()
                value_loss = (values - disc_rew).pow(2).mean()

                entropy = dist.entropy().mean()
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                loss_tot += loss
                n_batches += 1
        return loss_tot / max(n_batches,1)
    
    def save_weights(self, path: str) -> None:
        """
        Save both streams and the optimiser in one checkpoint.
        """
        
        torch.save({
            "self_policy" : self.self_policy.state_dict(),
            "team_policy" : self.team_policy.state_dict(),
            "self_value"  : self.self_value.state_dict(),
            "team_value"  : self.team_value.state_dict(),
            "optimizer"   : self.optimizer.state_dict(),
            # optional bookkeeping
            "running_avg_return": [list(q) for q in self.running_avg_return]
        }, path)

    def load_weights(self, path: str, map_location=None) -> None:
        """
        Load both streams and the optimiser in one checkpoint.
        """
        
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.self_policy.load_state_dict(ckpt["self_policy"])
        self.team_policy.load_state_dict(ckpt["team_policy"])
        self.self_value .load_state_dict(ckpt["self_value"])
        self.team_value .load_state_dict(ckpt["team_value"])
        self.optimizer  .load_state_dict(ckpt["optimizer"])
        if "running_avg_return" in ckpt:
            saved_lists = ckpt["running_avg_return"] # list[list[float]]
            for dq, saved in zip(self.running_avg_return, saved_lists):
                dq.clear() # keep same deque object
                dq.extend(saved) # refill with saved numbers
        print(f"SOTO checkpoint loaded from {path}")


