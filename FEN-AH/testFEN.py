# testFEN.py – evaluate a trained FEN policy
import argparse, os, pickle
import numpy as np
import torch
from tqdm import trange
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  
FAIRPPO = ROOT / "FEN-AH"   
sys.path.insert(0, str(ROOT))  
sys.path.insert(0, str(FAIRPPO))  

from policy_algorithms.FEN.FENAgent import FENAgent
from environments.AllelopaticHarvest.Environment import Environment
from environments.AllelopaticHarvest.BerryRegrowth import LinearRegrowth
from fairnessmetrics.FairnessMetrics import CumulativeMetrics
from policy_algorithms.FEN.Memory import MemoryFEN

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="Direct path to the trained agent checkpoint (.pth file)")
    p.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to test for.")
    p.add_argument("--max_timesteps", type=int, default=1500, help="Max timesteps per episode.")
    p.add_argument("--cuda", default="0")
    p.add_argument("--all_sensitive", type=int, choices=[0,1], default=None,
                   help="0 => only non-sensitive agents, 1 => only sensitive")
    p.add_argument("--save", action="store_true",
                   help="Pickle fairness + reward stats to disk")
    return p.parse_args()

def load_fen(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    agent = FENAgent(input_dim=10, output_dim=9,
                     k_sub=2, T_macro=10,
                     device=device)
    agent.load_weights(ckpt_path, map_location=device)
    agent.eval_mode = True
    return agent

def evaluate(agent, env, n_ep, max_t):
    """
    Runs the trained FEN agent in the environment for a number of episodes
    """
    
    num2action = {0: "stay", 1: "move_up", 2: "move_down", 3: "move_left", 4: "move_right", 
                  5: "eat_bush", 6: "change_bush_color", 7: "interact_with_nearby_player", 
                  8: "ripe_bush"}

    F = CumulativeMetrics() # to calculate cond stat parity

    cum_rew = {k:[] for k in ["non_sensitive_red","non_sensitive_blue",
                              "sensitive_red","sensitive_blue"]}
    action_freq = {"non_sensitive":{a:0 for a in num2action.values()},
                   "sensitive":    {a:0 for a in num2action.values()}}
    dp, dpN, csp, cspN = [], [], [], []

    for ep in trange(n_ep, desc="test"):
        state = env.reset()
        mem = MemoryFEN()
        ep_rew = {k:0.0 for k in cum_rew}

        for t in range(max_t):
            s_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device)

            with torch.no_grad():
                actions, logp_a, _ = agent.select_action(s_t, mem)
                
            mem.states.extend(s_t.cpu())
            mem.actions.extend(actions)
            mem.logprobs.extend(logp_a.cpu())
            
            # actions, environment, regrowth_rate, regrowth_function, max_lifespan, spont_growth_rate)
            next_state, rewards, done = env.step(
                                    actions,
                                    env,
                                    env.regrowth_rate,
                                    LinearRegrowth().regrowth,
                                    env.max_lifespan,
                                    env.spont_growth_rate)
            
            mem.rewards.extend(rewards.tolist())

            for i, r in enumerate(rewards):
                sens = "sensitive" if state[i][2] == 1 else "non_sensitive"
                col  = "red" if state[i][4] == 1 else "blue"
                ep_rew[f"{sens}_{col}"] += r
                action_freq[sens][num2action[actions[i]]] += 1

            state = next_state
            if done: break

        for k in cum_rew:
            cum_rew[k].append(ep_rew[k])

        # Fairness metrics (analysis purpose only)
        d, dn = F.demographic_parity(ep_rew["non_sensitive_red"] + ep_rew["non_sensitive_blue"],
                                     ep_rew["sensitive_red"] + ep_rew["sensitive_blue"])
        dp.append(d)
        dpN.append(dn)

        g1, g2, g1n, g2n = F.conditional_statistical_parity(
                ep_rew["non_sensitive_red"], ep_rew["sensitive_red"],
                ep_rew["non_sensitive_blue"], ep_rew["sensitive_blue"])
        csp.append({"CSP_G1": g1, "CSP_G2": g2})
        cspN.append({"CSP_G1": g1n, "CSP_G2": g2n})

    return dp, dpN, csp, cspN, cum_rew, action_freq


if __name__ == "__main__":
    args = cli()
    sens_pct = 0.5 # percentage of sensitive agents
    if args.all_sensitive is not None:
        sens_pct = 1.0 if args.all_sensitive else 0.0

    env_cfg = dict(x_dim=25, y_dim=15, num_players=40, num_bushes=60,
                   red_player_percentage=.5, blue_player_percentage=.5,
                   red_bush_percentage=.5, blue_bush_percentage=.5,
                   disability_percentage=sens_pct, max_steps=args.max_timesteps,
                   max_lifespan=120, spont_growth_rate=2, regrowth_rate=3)
    env = Environment(**env_cfg)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    agent = load_fen(args.ckpt_path, device)
    
    dp, dpN, csp, cspN, cum_rew, act_freq = evaluate(
        agent, env, args.num_episodes, args.max_timesteps)

    if args.save:
        outdir = "fairness_results_folder"
        os.makedirs(outdir, exist_ok=True)
        ckpt_name = os.path.basename(args.ckpt_path).split('.pth')[0]
        suffix = ""
        if args.all_sensitive == 1:
            suffix = "_allsensitive"
        elif args.all_sensitive == 0:
            suffix = "_nonsensitive"
        
        outfile = os.path.join(outdir, f"fairness_results.pkl")
        with open(outfile, "wb") as f:
            pickle.dump({"DP":dp, "DPN":dpN,
                         "CSP":csp, "CSPN":cspN,
                         "cumulative_rewards":cum_rew,
                         "action_frequencies":act_freq}, f)
        print("Saved analysis to:", outfile)
