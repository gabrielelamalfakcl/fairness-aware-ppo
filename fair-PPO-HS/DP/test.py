import argparse
import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import os
import random
from pathlib import Path
import sys

# Path to import other scripts: it needs to be checked based on the folder structure
try:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
except (IndexError, NameError):
    project_root = Path.cwd()
    sys.path.append(str(project_root))

from environments.Hospital.hospital_pz_env import HospitalMAS
from PPOAgentMDDP import PPOAgentMDDP
from environments.Hospital.utils import generate_arrival_templates

def test_ppo(agents: dict, config: dict, test_rng: random.Random, env: HospitalMAS):
    """
    Test of Fair-PPO (DP)
    """
    
    for agent in agents.values():
        agent.policy.eval()

    sim_minutes = config["sim_hours"] * 60
    all_episode_results = []

    print(f"Running {config['num_test_episodes']} test episodes...")
    for episode in tqdm(range(config["num_test_episodes"])):
        template_seed = test_rng.randint(0, 1_000_000_000)
        unique_template = generate_arrival_templates(n_templates=1, sim_minutes=sim_minutes, seed=template_seed)[0]
        env.arrival_template_override = unique_template
        
        episode_seed = test_rng.randint(0, 1_000_000_000)
        obs, _ = env.reset(seed=episode_seed)
        done = {"__all__": False}
        
        while not done["__all__"]:
            action_dict = {}
            with torch.no_grad():
                for agent_name, agent_obs in obs.items():
                    if agent_obs.sum() > 0:
                        state_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0).to(config["device"])
                        action, _, _ = agents[agent_name].select_actions(state_tensor, use_greedy=False) # Use greedy for deterministic testing
                        action_dict[agent_name] = action[0]
            
            obs, rewards, terminated, truncated, infos = env.step(action_dict)
            done = terminated

        # Episode metric calculation
        sim = env.sim
        finished_patients = [p for p in sim.patients if p.status in ("done", "exited")]

        finished_impaired_count = len([p for p in finished_patients if p.true_impairment != "none"])
        finished_non_impaired_count = len([p for p in finished_patients if p.true_impairment == "none"])

        wt_escort_imp = [p.escort_wait_time for p in finished_patients if p.true_impairment != "none" and hasattr(p, 'escort_wait_time')]
        wt_escort_non = [p.escort_wait_time for p in finished_patients if p.true_impairment == "none" and hasattr(p, 'escort_wait_time')]
        avg_wt_escort_imp = np.mean(wt_escort_imp) if wt_escort_imp else 0.0
        avg_wt_escort_non = np.mean(wt_escort_non) if wt_escort_non else 0.0
        escort_wait_fairness_gap = abs(avg_wt_escort_non - avg_wt_escort_imp)

        patient_rewards = sim.drain_patient_rewards()
        impaired_rewards = [patient_rewards.get(p.id, 0.0) for p in finished_patients if p.true_impairment != "none"]
        non_imp_rewards  = [patient_rewards.get(p.id, 0.0) for p in finished_patients if p.true_impairment == "none"]
        avg_reward_impaired = np.mean(impaired_rewards) if impaired_rewards else 0.0
        avg_reward_non_impaired = np.mean(non_imp_rewards) if non_imp_rewards else 0.0
        reward_fairness_gap = abs(avg_reward_non_impaired - avg_reward_impaired)

        routing_accuracy = np.mean(sim.routing_decisions) if sim.routing_decisions else 0.0
        avg_escort_wait_time = np.mean(sim.escort_wait_times) if sim.escort_wait_times else 0.0
        avg_helper_travel_time = np.mean(sim.helper_travel_times) if sim.helper_travel_times else 0.0
        avg_ward_wait_time = np.mean(sim.ward_wait_times) if sim.ward_wait_times else 0.0
        doctor_moves = sim.doctor_moves_count
        
        routing_outcomes = sim.routing_outcomes
        total_routings = len(routing_outcomes)
        perfect_pct = routing_outcomes.count('perfect') * 100 / total_routings if total_routings > 0 else 0
        acceptable_pct = routing_outcomes.count('acceptable') * 100 / total_routings if total_routings > 0 else 0
        wrong_pct = routing_outcomes.count('wrong') * 100 / total_routings if total_routings > 0 else 0

        all_episode_results.append({
            "episode_seed": episode_seed,
            "total_patients_finished": len(finished_patients),
            "finished_impaired_count": finished_impaired_count,
            "finished_non_impaired_count": finished_non_impaired_count,
            "avg_reward_impaired": avg_reward_impaired,
            "avg_reward_non_impaired": avg_reward_non_impaired,
            "reward_fairness_gap": reward_fairness_gap,
            "avg_escort_wait_impaired": avg_wt_escort_imp,
            "avg_escort_wait_non_impaired": avg_wt_escort_non,
            "escort_wait_fairness_gap": escort_wait_fairness_gap,
            "triage_routing_accuracy": routing_accuracy,
            "escort_avg_patient_wait_time": avg_escort_wait_time,
            "escort_avg_helper_travel_time": avg_helper_travel_time,
            "manager_avg_ward_wait_time": avg_ward_wait_time,
            "manager_doctor_moves": doctor_moves,
            "routing_perfect_pct": perfect_pct,
            "routing_acceptable_pct": acceptable_pct,
            "routing_wrong_pct": wrong_pct,
        })

    # Save results
    os.makedirs(config["output_dir"], exist_ok=True)
    df = pd.DataFrame(all_episode_results)
    
    output_filename = f"fairness_results_{config['alpha']}_{config['beta']}.pkl"
    output_path = os.path.join(config["output_dir"], output_filename)
    df.to_pickle(output_path)

    print(f"\nTest results DataFrame saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Fair PPO (DP) testing for Hospital MAS")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value of the trained model.")
    parser.add_argument("--beta", type=float, required=True, help="Beta value of the trained model.")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing the final .pth weight files.")
    parser.add_argument("--output_dir", type=str, default="fairness_results_DP", help="Root directory to save the test result files.")
    parser.add_argument("--num_test_episodes", type=int, default=100, help="Number of episodes to run for testing.")
    parser.add_argument("--cuda", type=str, default="cuda:0", help="CUDA device.")
    parser.add_argument("--test_seed", type=int, default=123, help="Seed for the test run for reproducibility.")
    args = parser.parse_args()

    config = {
        "alpha": args.alpha, "beta": args.beta, "num_test_episodes": args.num_test_episodes,
        "weights_dir": args.weights_dir, "output_dir": args.output_dir,
        "device": torch.device(args.cuda if torch.cuda.is_available() else "cpu"),
        "test_seed": args.test_seed, "gamma": 0.99,
        "n_clerks": 30, "m_nurses": 60, "k_robots": 30, "n_triage": 30,
        "n_swing_doctors": 18, "n_patients": 300, "sim_hours": 12,
    }
    
    env_keys = ["n_clerks", "n_patients", "m_nurses", "k_robots", "n_triage", "n_swing_doctors", "sim_hours", "gamma"]
    env_config = {key: config[key] for key in env_keys if key in config}
    main_env = HospitalMAS(verbose=False, **env_config)
    
    test_rng = random.Random(config["test_seed"])

    agents = {}
    agent_input_dims = {"escort_dispatcher": 6, "triage_router": 18, "doctor_manager": 19}
    
    print(f"Loading weights from: {config['weights_dir']}")
    for name, input_dim in agent_input_dims.items():
        agents[name] = PPOAgentMDDP(input_dim=input_dim, action_space=main_env.action_spaces[name], lr=0, gamma=0, eps_clip=0, k_epochs=0, batch_size=0, alpha=0, beta=0, value_loss_coef=0, entropy_coef=0, device=config["device"])
        weights_path = os.path.join(config['weights_dir'], f'{name}_final_weights.pth')
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=config["device"])
            agents[name].policy.load_state_dict(state_dict)
            print(f"Successfully loaded weights for {name}")
        else:
            print(f"ERROR: Weight file not found for {name} at {weights_path}")
            sys.exit(1)

    test_ppo(agents, config, test_rng, main_env) 
    print('Testing finished.')