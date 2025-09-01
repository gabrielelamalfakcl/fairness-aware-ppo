from .BasePPOAgent import BasePPOAgent

class PPOAgent(BasePPOAgent):
    def __init__(self, input_dim, output_dim, lr, gamma, eps_clip, k_epochs, batch_size, entropy_coef=0.01, value_loss_coef=0.1, device='cuda:0'):
        super().__init__(input_dim, output_dim, lr, gamma, eps_clip, k_epochs, batch_size, entropy_coef, value_loss_coef, device)

    def update(self, memory):
        total_loss = 0
        num_updates = 0

        memory_states = torch.stack(memory.states).to(self.device)
        memory_actions = torch.tensor(np.array(memory.actions)).to(self.device)
        memory_rewards = torch.tensor(np.array(memory.rewards), dtype=torch.float32).to(self.device)
        memory_logprobs = torch.stack(memory.logprobs).to(self.device)
        memory_state_values = torch.stack(memory.state_values).to(self.device)

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

                batch_discounted_rewards = (batch_discounted_rewards - batch_discounted_rewards.mean()) / (batch_discounted_rewards.std() + 1e-8)

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
                    entropy_loss = -0.01 * dist_entropy.mean()

                    loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
                    total_loss += loss
                    num_updates += 1

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

        return total_loss / num_updates if num_updates > 0 else 0
