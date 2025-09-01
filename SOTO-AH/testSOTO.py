import argparse, pickle
import numpy as np
import torch
from tqdm import trange
import sys, pathlib, os

ROOT = pathlib.Path(__file__).resolve().parents[1]  
FAIRPPO = ROOT / "SOTO-AH"   
sys.path.insert(0, str(ROOT))  
sys.path.insert(0, str(FAIRPPO))  

from policy_algorithms.SOTO.SOTOAgent import SOTOAgent
from environments.AllelopaticHarvest.Environment import Environment
from environments.AllelopaticHarvest.BerryRegrowth import LinearRegrowth
from fairnessmetrics.FairnessMetrics import CumulativeMetrics


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha",        type=float, required=True,
                   help="alpha used **during training** (needed only to find the ckpt)")
    p.add_argument("--num_episodes", type=int,   required=True)
    p.add_argument("--max_timesteps",type=int,   required=True)
    p.add_argument("--cuda",         type=str,   default="0")
    p.add_argument("--all_sensitive",type=int,   choices=[0,1], default=None,
                   help="0 ⇒ spawn only non-sensitive players, "
                        "1 ⇒ only sensitive players, "
                        "omit ⇒ 50/50 mix (regular DP/CSP test)")
    p.add_argument("--save", action="store_true",
                   help="pickle fairness + reward stats to disc")
    return p.parse_args()


def load_soto(weights_dir, device, env):
    ckpt_path = os.path.join(weights_dir, "soto_agent.pth")

    agent = SOTOAgent(input_dim = 10,
                      output_dim = 9,
                      extra_obs_dim = 3,      
                      lr = 5e-4, # not used for test               
                      gamma = 0.99,
                      eps_clip = 0.2,
                      k_epochs = 5,
                      batch_size = 256,
                      device = device,
                      num_players = env.num_players)

    agent.load_weights(ckpt_path, map_location=device)
    
    # At test time, we force the agent to use the team-oriented policy, as a final learnt policy
    agent.use_greedy = False
    return agent


def evaluate(agent, env, n_ep, max_t):
    num2action = {0: "stay", 1: "move_up", 2: "move_down", 3: "move_left", 4: "move_right",
        5: "eat_bush", 6: "change_bush_color", 7: "interact_with_nearby_player",
        8: "ripe_bush"
    }
    F = CumulativeMetrics()

    # logs
    cum_rew = {k:[] for k in ["non_sensitive_red","non_sensitive_blue",
                              "sensitive_red","sensitive_blue"]}
    action_freq = {"non_sensitive":{a:0 for a in num2action.values()},
                   "sensitive":    {a:0 for a in num2action.values()}}
    dp, dpN, csp, cspN = [], [], [], []

    for ep in trange(n_ep, desc="test"):
        state = env.reset()

        ep_rew = {k:0.0 for k in cum_rew} 

        for t in range(max_t):
            s_t   = torch.as_tensor(state, dtype=torch.float32,
                                    device=agent.device)
            gfeat = env.global_features()
            extra = np.repeat(gfeat[None,:], env.num_players, 0)
            extra = torch.as_tensor(extra, dtype=torch.float32,
                                     device=agent.device)

            acts, _, _ , _ = agent.select_action(s_t, extra)
            next_state, r, done = env.step(acts, env,
                                           env.regrowth_rate,
                                           LinearRegrowth().regrowth,
                                           env.max_lifespan,
                                           env.spont_growth_rate)

            for i, rew in enumerate(r):
                sens = "sensitive" if state[i][2]==1 else "non_sensitive"
                pref = "red" if state[i][4]==1 else "blue"
                ep_rew[f"{sens}_{pref}"] += rew
                action_freq[sens][num2action[acts[i]]] += 1

            state = next_state
            if done: break

        # episode finished – store fairness metrics
        for k in cum_rew: cum_rew[k].append(ep_rew[k])

        d, dn = F.demographic_parity( ep_rew["non_sensitive_red"]+
                                      ep_rew["non_sensitive_blue"],
                                      ep_rew["sensitive_red"]+
                                      ep_rew["sensitive_blue"])
        dp.append(d); dpN.append(dn)

        g1, g2, g1n, g2n = F.conditional_statistical_parity(
                ep_rew["non_sensitive_red"], ep_rew["sensitive_red"],
                ep_rew["non_sensitive_blue"], ep_rew["sensitive_blue"])
        csp.append({"CSP_G1":g1, "CSP_G2":g2})
        cspN.append({"CSP_G1":g1n,"CSP_G2":g2n})

    return dp, dpN, csp, cspN, cum_rew, action_freq


if __name__ == "__main__":
    args = cli()

    # Build environment
    # Default: 50% agents with and 50% without sensitive attribute
    sens_pct = 0.5                               
    if args.all_sensitive is not None:
        sens_pct = 1.0 if args.all_sensitive else 0.0
    
    env_cfg = dict(x_dim=25, y_dim=15, num_players=40, num_bushes=60,
                   red_player_percentage=.5, blue_player_percentage=.5,
                   red_bush_percentage=.5, blue_bush_percentage=.5,
                   disability_percentage=sens_pct, max_steps=args.max_timesteps,
                   max_lifespan=120, spont_growth_rate=2, regrowth_rate=3)
    env = Environment(**env_cfg)

    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() else "cpu")
    weights_dir = f"soto_weights/alpha_{args.alpha}"
    agent = load_soto(weights_dir, device, env)

    dp, dpN, csp, cspN, cum_rew, act_freq = evaluate(
            agent, env, args.num_episodes, args.max_timesteps)
    
    if args.save:
        outdir = "fairness_results_folder"
        os.makedirs(outdir, exist_ok=True)

        # Determine filename: add suffix if it's a counterfactual world
        suffix = ""
        if args.all_sensitive == 1:
            suffix = "_world1"
        elif args.all_sensitive == 0:
            suffix = "_world0"

        outfile = os.path.join(outdir, f"fairness_results_alpha={args.alpha}{suffix}.pkl")
        with open(outfile, "wb") as f:
            pickle.dump({"DP":dp, "DPN":dpN,
                         "CSP":csp, "CSPN":cspN,
                         "cumulative_rewards":cum_rew,
                         "action_frequencies":act_freq}, f)
        print("saved", outfile)
