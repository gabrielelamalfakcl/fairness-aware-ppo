# test_counterfactual.py – evaluate two separately trained FEN policies
import argparse
import os
import pickle
import numpy as np
import torch
from tqdm import trange
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]  
FAIRPPO = ROOT / "FEN-AH"   
sys.path.insert(0, str(ROOT))  
sys.path.insert(0, str(FAIRPPO))  

from policy_algorithms.FEN.FENAgent import FENAgent
from environments.AllelopaticHarvest.Environment import Environment
from fairnessmetrics.FairnessMetrics import CumulativeMetrics

def cli():
    p = argparse.ArgumentParser(description="Evaluate two separately trained FEN agents in their respective worlds.")
    p.add_argument("--ckpt_world0", type=str, required=True,
                   help="Path to the checkpoint for the agent trained in the 'all non-sensitive' world.")
    p.add_argument("--ckpt_world1", type=str, required=True,
                   help="Path to the checkpoint for the agent trained in the 'all sensitive' world.")
    p.add_argument("--num_episodes", type=int, default=100)
    p.add_argument("--max_timesteps", type=int, default=1500)
    p.add_argument("--cuda", default="0")
    p.add_argument("--save", action="store_true", help="Save fairness metrics to a .pkl file")
    return p.parse_args()

def load_agent(ckpt_path, device):
    """
    Loads a single FEN agent from a checkpoint
    """
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Ensure agent is initialized with the correct 9-action output space
    agent = FENAgent(input_dim=10, output_dim=9, k_sub=2, T_macro=10, device=device)
    agent.load_weights(ckpt_path, map_location=device)
    agent.eval_mode = True
    print(f"Loaded agent from: {ckpt_path}")
    return agent

def run_world(agent, env, n_ep, max_t):
    """Runs a given agent in a given environment for n_ep episodes."""
    num2action = {
        0: "stay", 1: "move_up", 2: "move_down", 3: "move_left", 4: "move_right",
        5: "eat_bush", 6: "change_bush_color", 7: "interact_with_nearby_player",
        8: "ripe_bush"
    }
    F = CumulativeMetrics()

    all_rewards = []
    
    for _ in trange(n_ep, leave=False, desc="Evaluating World"):
        state = env.reset()
        ep_rew = 0.0

        for _ in range(max_t):
            s_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
            with torch.no_grad():
                actions, _, _ = agent.select_action(s_t, None)
        
            s_next, r, done, *_ = env.step(actions)

            ep_rew += np.sum(r)
            state = s_next
            if done: break
        
        all_rewards.append(ep_rew)

    return {"total_rewards": all_rewards}

if __name__ == "__main__":
    args = cli()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # Load the two specialist agents
    agent_0 = load_agent(args.ckpt_world0, device)
    agent_1 = load_agent(args.ckpt_world1, device)

    def make_env(sens_pct):
        return Environment(
            x_dim=25, y_dim=15, num_players=40, num_bushes=60,
            red_player_percentage=.5, blue_player_percentage=.5,
            red_bush_percentage=.5, blue_bush_percentage=.5,
            disability_percentage=sens_pct,
            max_steps=args.max_timesteps,
            max_lifespan=120, spont_growth_rate=2, regrowth_rate=3
        )

    # Run agent_0 in the factual world (0% sensitive)
    print("\nRunning Agent 0 in World 0 (all non-sensitive agents)")
    env0 = make_env(sens_pct=0.0)
    res_world0 = run_world(agent_0, env0, args.num_episodes, args.max_timesteps)
    
    # Run agent_1 in the counterfactual world (100% sensitive)
    print("\nRunning Agent 1 in World 1 (all sensitive agents)")
    env1 = make_env(sens_pct=1.0)
    res_world1 = run_world(agent_1, env1, args.num_episodes, args.max_timesteps)
    
    # Calculate final results 
    avg_reward_w0 = np.mean(res_world0["total_rewards"])
    avg_reward_w1 = np.mean(res_world1["total_rewards"])
    cf_gap = avg_reward_w0 - avg_reward_w1


    if args.save:
        outdir = "fairness_data_FEN_CF"
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, "counterfactual_results.pkl")
        with open(outfile, "wb") as f:
            pickle.dump({
                "world0_rewards": res_world0["total_rewards"],
                "world1_rewards": res_world1["total_rewards"],
                "cf_gap": cf_gap
            }, f)
        print(f"Saved counterfactual analysis to: {outfile}")
