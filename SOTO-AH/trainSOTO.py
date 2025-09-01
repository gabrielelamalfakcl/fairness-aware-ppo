import torch, argparse, os
import numpy as np
from tqdm import tqdm
import wandb
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]  
FAIRPPO = ROOT / "SOTO-AH"   
sys.path.insert(0, str(ROOT))  
sys.path.insert(0, str(FAIRPPO))  

from policy_algorithms.SOTO.SOTOAgent import SOTOAgent
from policy_algorithms.SOTO.Memory import Memory
from environments.AllelopaticHarvest.Environment import Environment
from environments.AllelopaticHarvest.BerryRegrowth import LinearRegrowth

p = argparse.ArgumentParser()
p.add_argument('--alpha', type=float, default=2.0, help="Fairness parameter alpha")
p.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
p.add_argument('--gamma', type=float, default=0.99, help="Discount factor gamma")
p.add_argument('--entropy_coef', type=float, default=0.05, help="Entropy bonus coefficient")
p.add_argument('--twophase_prop', type=float, default=0.5, help="Proportion of training for beta-annealing phase")
p.add_argument('--num_episodes', type=int, default=500, help="Number of training episodes")
p.add_argument('--max_timesteps', type=int, default=1500, help="Max timesteps per episode")
p.add_argument('--cuda', default='cuda:0', help="CUDA device")
args = p.parse_args()

# wandb
wandb.init(
    project="SOTO-Training-Experimental",
    config={
        "alpha": args.alpha,
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "entropy_coef": args.entropy_coef,
        "twophase_prop": args.twophase_prop,
        "episodes": args.num_episodes,
        "timesteps": args.max_timesteps
    },
    name=f"SOTO_alpha={args.alpha}_lr={args.lr}_ent={args.entropy_coef}"
)

# config
cfg = dict(
    input_dim=10,
    output_dim=9,
    lr=args.lr,
    gamma=args.gamma,
    eps_clip=0.2,
    k_epochs=5,
    batch_size=256,
    extra_obs_dim=3,
    entropy_coef=args.entropy_coef,
    device=torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
)

# Initialise the environment
env = Environment(
    x_dim=25, y_dim=15, max_steps=args.max_timesteps,
    num_players=40, num_bushes=60,
    red_player_percentage=0.5, blue_player_percentage=0.5,
    red_bush_percentage=0.5, blue_bush_percentage=0.5,
    disability_percentage=0.5,
    max_lifespan=120, spont_growth_rate=2, regrowth_rate=3
)

# Initialise the agent (SOTOAgent)
agent = SOTOAgent(
    input_dim=cfg['input_dim'],
    output_dim=cfg['output_dim'],
    extra_obs_dim=cfg['extra_obs_dim'],
    lr=cfg['lr'],
    gamma=cfg['gamma'],
    eps_clip=cfg['eps_clip'],
    k_epochs=cfg['k_epochs'],
    batch_size=cfg['batch_size'],
    entropy_coef=cfg['entropy_coef'], # <-- Pass the new hyperparameter
    device=cfg['device'],
    num_players=env.num_players
)

for ep in tqdm(range(args.num_episodes)):
    state = env.reset()
    agent.start_episode(ep, args.num_episodes, twophase_prop=args.twophase_prop)
    mem = Memory()
    ep_returns = np.zeros(env.num_players)

    # Track per-group cumulative rewards
    ep_group_returns = {
        "non_sensitive_red": 0.0,
        "non_sensitive_blue": 0.0,
        "sensitive_red": 0.0,
        "sensitive_blue": 0.0
    }

    for t in range(args.max_timesteps):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=cfg['device'])
        global_feat = env.global_features() # (3,)
        extra_np = np.repeat(global_feat[None, :], env.num_players, 0) # (N,3)
        extra_tensor = torch.as_tensor(extra_np, dtype=torch.float32, device=cfg['device'])

        acts, logp, vals, sids = agent.select_action(state_tensor, extra_tensor)
        
        next_state, rew, done = env.step(actions=acts, environment=env, regrowth_rate=env.regrowth_rate, regrowth_function=LinearRegrowth().regrowth,
                                         max_lifespan=env.max_lifespan, spont_growth_rate=env.spont_growth_rate)
        
        # Store memory
        for i in range(env.num_players):
            mem.states.append(state_tensor[i].cpu())
            mem.extra_obs.append(extra_tensor[i].cpu())
            mem.actions.append(acts[i])
            mem.logprobs.append(logp[i])
            mem.state_values.append(vals[i].item())
            mem.rewards.append(rew[i])
            mem.stream_ids.append(sids[i])
            mem.player_ids.append(i)

            # Group returns
            sensitive = state[i][2] == 1
            color = state[i][4]
            if sensitive and color == 1:
                ep_group_returns["sensitive_red"] += rew[i]
            elif sensitive and color == 2:
                ep_group_returns["sensitive_blue"] += rew[i]
            elif not sensitive and color == 1:
                ep_group_returns["non_sensitive_red"] += rew[i]
            elif not sensitive and color == 2:
                ep_group_returns["non_sensitive_blue"] += rew[i]

        ep_returns += rew
        state = next_state
        if done: break

    for i in range(env.num_players):
        agent.running_avg_return[i].append(ep_returns[i])
    agent.update(mem, alpha=args.alpha)

    # Wandb
    wandb.log({
        "Episode": ep,
        "Reward/non_sensitive_red": ep_group_returns["non_sensitive_red"],
        "Reward/non_sensitive_blue": ep_group_returns["non_sensitive_blue"],
        "Reward/sensitive_red": ep_group_returns["sensitive_red"],
        "Reward/sensitive_blue": ep_group_returns["sensitive_blue"],
        "Total Episode Reward": np.sum(ep_returns)
    })

    # Save weights 
    weights_dir = f"soto_weights/alpha_{args.alpha}"
    os.makedirs(weights_dir, exist_ok=True)
    ckpt_path = os.path.join(weights_dir, "soto_agent.pth")
    agent.save_weights(ckpt_path)
    print("Weights saved to", ckpt_path)

wandb.finish()
