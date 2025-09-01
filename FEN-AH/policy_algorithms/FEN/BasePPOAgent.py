import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.cuda.amp import autocast, GradScaler
import numpy as np

scaler = GradScaler()

class PolicyNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LeakyReLU(),
            nn.Linear(256, 128),    nn.LeakyReLU(),
            nn.Linear(128,  64),    nn.LeakyReLU(),
            nn.Linear( 64, out_dim))
    def forward(self, x):           
        return torch.softmax(self.net(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LeakyReLU(),
            nn.Linear(256, 128),    nn.LeakyReLU(),
            nn.Linear(128,  64),    nn.LeakyReLU(),
            nn.Linear( 64, 1))
    def forward(self, x):           
        return self.net(x).squeeze(-1)


class BasePPOAgent:
    """
    FEN needs a policy-gradient optimiser 
    PPO is a the optimiser we use to be consistent with other wrappers/methods
    """
    
    def __init__(self, input_dim, output_dim, lr, gamma, eps_clip,
                    k_epochs, batch_size,
                    entropy_coef=0.01, value_loss_coef=0.1,
                    device='cuda:0', *,
                    build_default_nets: bool = True):

        self.device = torch.device(device)

        if build_default_nets:
            self.policy_net = PolicyNetwork(input_dim, output_dim).to(self.device)
            self.value_net  = ValueNetwork(input_dim).to(self.device)

            self.optimizer  = optim.Adam(
                [{'params': self.policy_net.parameters()},
                 {'params': self.value_net.parameters()}], lr=lr)
        else:
            # subclasses (FEN) will create their own stuff
            self.policy_net = self.value_net = self.optimizer = None

        # common hyper-params
        self.gamma, self.eps_clip = gamma, eps_clip
        self.k_epochs, self.batch_size = k_epochs, batch_size
        self.entropy_coef, self.value_loss_coef = entropy_coef, value_loss_coef

        # optional: keep around snapshots for debugging
        self.weights_log = []

    def compute_gae(self, rewards, values, lam: float = .95):
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        values  = torch.as_tensor(values , dtype=torch.float32, device=self.device)
        values  = torch.cat([values, values.new_zeros(1)])      # V_{t+1}

        adv, gae = [], 0.
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] - values[t]
            gae   = delta + self.gamma * lam * gae
            adv.insert(0, gae)
        adv = torch.stack(adv)
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    def select_actions(self, states):
        assert self.policy_net is not None, "select_actions() unused in FEN."
        probs = self.policy_net(states)
        dist  = Categorical(probs)
        acts  = dist.sample()
        vals  = self.value_net(states)
        return acts, dist.log_prob(acts), vals

    def update_weights_log(self):
        if self.policy_net is None:         # FEN case
            return
        snap = {n: p.detach().cpu().numpy()
                for n, p in self.policy_net.named_parameters()}
        snap.update({f"V.{n}": p.detach().cpu().numpy()
                     for n, p in self.value_net.named_parameters()})
        self.weights_log.append(snap)

    def update(self, memory):   raise NotImplementedError
    
# A minimal plain-PPO agent 
class PPOAgent(BasePPOAgent):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw, build_default_nets=True)

    def update(self, memory):
        """
        Standard clipped-PPO update (unchanged – relies on the
        default nets we created in __init__ above).
        """
        # --- tensorise memory ---
        S  = torch.stack(memory.states).to(self.device)
        A  = torch.as_tensor(np.array(memory.actions)).to(self.device)
        R  = torch.as_tensor(memory.rewards, dtype=torch.float32).to(self.device)
        LP = torch.stack(memory.logprobs).to(self.device)
        V  = torch.stack(memory.state_values).to(self.device)

        adv = self.compute_gae(R.tolist(), V.squeeze(-1).tolist())
        ret = adv + V.squeeze(-1).detach()

        total_loss, n_upd = 0., 0
        for _ in range(self.k_epochs):
            for i in range(0, len(S), self.batch_size):
                s, a = S[i:i+self.batch_size], A[i:i+self.batch_size]
                old_lp, ad = LP[i:i+self.batch_size], adv[i:i+self.batch_size].detach()
                tgt = ret[i:i+self.batch_size]

                with autocast():
                    probs = self.policy_net(s)
                    dist  = Categorical(probs)
                    lp    = dist.log_prob(a)
                    ratio = torch.exp(lp - old_lp)
                    surr1, surr2 = ratio * ad, torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * ad
                    policy_loss  = -torch.min(surr1, surr2).mean()

                    v_pred = self.value_net(s)
                    value_loss = (v_pred - tgt).pow(2).mean()

                    entropy = dist.entropy().mean()
                    loss = (policy_loss
                            + self.value_loss_coef * value_loss
                            - self.entropy_coef  * entropy)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer); scaler.update()

                total_loss += loss.item();   n_upd += 1

        return total_loss / n_upd if n_upd else 0.