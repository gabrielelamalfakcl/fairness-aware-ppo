# -*- coding: utf-8 -*-
import argparse
import torch
import numpy as np
from tqdm import tqdm
import wandb
import os
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  
FAIRPPO = ROOT / "FEN-AH"   
sys.path.insert(0, str(ROOT))  
sys.path.insert(0, str(FAIRPPO))  

from environments.AllelopaticHarvest.Environment import Environment
from environments.AllelopaticHarvest.BerryRegrowth import LinearRegrowth
from policy_algorithms.FEN.FENAgent import FENAgent
from policy_algorithms.FEN.Memory import MemoryFEN

p = argparse.ArgumentParser()
p.add_argument("--num_episodes", type=int, default=500)
p.add_argument("--max_timesteps", type=int, default=1500)
p.add_argument("--save_every", type=int, default=100, help="ckpt frequency (eps)")
p.add_argument("--cuda", default="cuda:0")
p.add_argument("--resume_ckpt", type=str, default=None)
args = p.parse_args()

device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

# wandb
wandb.init(project="FEN-Training",
           config=dict(episodes=args.num_episodes,
                       timesteps=args.max_timesteps),
           name="FEN-Training-Run")

# initialise the environment
env = Environment(x_dim=25, y_dim=15, max_steps=args.max_timesteps,
                  num_players=40, num_bushes=60,
                  red_player_percentage=.5, blue_player_percentage=.5,
                  red_bush_percentage=.5, blue_bush_percentage=.5,
                  disability_percentage=.5,
                  max_lifespan=120, spont_growth_rate=2, regrowth_rate=3)

# initialise the agent (FENAgent)
agent = FENAgent(input_dim=10, output_dim=9,
                 k_sub=2, T_macro=10,
                 device=device)

start_ep = 0
if args.resume_ckpt:
    agent.load_weights(args.resume_ckpt, map_location=device)
    start_ep = (int(args.resume_ckpt.rsplit('_ep',1)[1].split('.pth')[0]) + 1)

for ep in tqdm(range(start_ep, args.num_episodes), ascii=True):
    obs = env.reset()

    mem = MemoryFEN()
    done = False
    ep_returns = np.zeros(env.num_players)
    
    t_step = 0

    grp = {k:0.0 for k in ["non_sensitive_red","non_sensitive_blue",
                           "sensitive_red","sensitive_blue"]} # logging purposes

    while not done:
        t_step += 1
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

        # act
        acts, logp_a, _ = agent.select_action(obs_t, mem)

        # store generic transition data
        mem.states.extend(obs_t.cpu())
        mem.actions.extend(acts) 
        mem.logprobs.extend(logp_a.cpu()) 

        # env step
        nxt_obs, rew, done = env.step(acts, env, env.regrowth_rate, LinearRegrowth().regrowth, env.max_lifespan, env.spont_growth_rate)

        mem.rewards.extend(rew.tolist())
        
        rew_t = torch.as_tensor(rew, device=device)
        
        # Calculates a running average for utility
        if t_step == 1:
            # On the first step, utility is just the first reward
            agent.u_i = rew_t.clone()
            # The gossip value (u_bar_i) is initialized with the agent's own utility
            agent.u_bar_i = rew_t.clone()
        else:
            # For all other steps, update u_i using the running average formula
            agent.u_i += (rew_t - agent.u_i) / t_step

        # The gossip function is called every macro step to get a consensus average
        if agent.t_since_z == 0: 
            neighbours = [list(range(env.num_players)) for _ in range(env.num_players)]
            new_bar = agent.u_bar_i.clone()
            for i, nbrs in enumerate(neighbours):
                d_i = max(len(nbrs), 1)
                for j in nbrs:
                    w_ij = 1.0 / (max(d_i, len(neighbours[j])) + 1)
                    # The gossip update rule remains the same
                    new_bar[i] += w_ij * (agent.u_bar_i[j] - agent.u_bar_i[i]) 
            agent.u_bar_i = new_bar

        # fairness counters
        sens = (obs_t[:,2] == 1).cpu().numpy()
        red  = (obs_t[:,4] == 1).cpu().numpy()

        # group returns for wandb
        for i,r in enumerate(rew):
            if sens[i] and red[i]:   
                grp["sensitive_red"] += r
            elif sens[i] and not red[i]: 
                grp["sensitive_blue"] += r
            elif not sens[i] and red[i]: 
                grp["non_sensitive_red"] += r
            else: 
                grp["non_sensitive_blue"]+= r

        ep_returns += rew
        obs = nxt_obs
    # end episode

    loss = agent.update(mem, beta_kl=0.5) 

    # wandb log
    log_data = {
        "Episode": ep,
        "Loss": loss,
        "Total_Reward": ep_returns.sum()
    }

    reward_data = {f"Reward/{k}": v for k, v in grp.items()}

    # Update the main dictionary with the reward data
    log_data.update(reward_data)

    # Log the final, combined dictionary
    wandb.log(log_data)
    
    # checkpoint
    if ep % args.save_every == 0 or ep == args.num_episodes-1:
        ckpt_dir = "fen_weights/default"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = f"{ckpt_dir}/ckpt_ep{ep}.pth"
        agent.save_weights(ckpt_path)

wandb.finish()
print("Training complete")