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

# Path to import other scripts
try:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
except (IndexError, NameError):
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from environments.Hospital.hospital_pz_env import HospitalMAS
from PPOAgentMDCF import PPOAgentMDCF
from environments.Hospital.utils import generate_arrival_templates, build_arrival_schedule
from environments.Hospital.players import Patient

def test_ppo_cf(agents_w0, agents_w1, config, test_rng, env_w0, env_w1, test_patient_schedules_w0, test_patient_schedules_w1):
    """
    Tests the trained Fair-PPO CF agents with detailed metric collection,
    mirroring the structure of the DP test script.
    """
    for agent in agents_w0.values(): agent.policy.eval()
    for agent in agents_w1.values(): agent.policy.eval()

    all_results = []

    for episode in tqdm(range(config["num_test_episodes"]), desc="Running Test Episodes"):
        # Set patient schedules
        env_w0.set_patient_schedule(test_patient_schedules_w0[episode])
        env_w1.set_patient_schedule(test_patient_schedules_w1[episode])

        episode_seed = test_rng.randint(0, 1_000_000_000)
        obs_w0, _ = env_w0.reset(seed=episode_seed)
        obs_w1, _ = env_w1.reset(seed=episode_seed)
        
        done_w0, done_w1 = {"__all__": False}, {"__all__": False}

        while not done_w0["__all__"] and not done_w1["__all__"]:
            # Step world 0 (counterfactual)
            action_dict_w0 = {}
            with torch.no_grad():
                for agent_name, agent_obs in obs_w0.items():
                    if agent_obs.sum() > 0:
                        state_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0).to(config["device"])
                        action, _, _ = agents_w0[agent_name].select_actions(state_tensor)
                        action_dict_w0[agent_name] = action[0]
            obs_w0, _, terminated_w0, _, _ = env_w0.step(action_dict_w0)
            done_w0 = terminated_w0

            # Step world 1 (counterfactual)
            action_dict_w1 = {}
            with torch.no_grad():
                for agent_name, agent_obs in obs_w1.items():
                    if agent_obs.sum() > 0:
                        state_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0).to(config["device"])
                        action, _, _ = agents_w1[agent_name].select_actions(state_tensor)
                        action_dict_w1[agent_name] = action[0]
            obs_w1, _, terminated_w1, _, _ = env_w1.step(action_dict_w1)
            done_w1 = terminated_w1

        # Metric calculation for both worlds
        def get_detailed_metrics(env):
            sim = env.sim
            results = {}
            finished_patients = [p for p in sim.patients if p.status in ("done", "exited")]
            patient_rewards = sim._patient_rewards

            # Group-based metrics
            finished_impaired = [p for p in finished_patients if p.true_impairment != "none"]
            finished_non_impaired = [p for p in finished_patients if p.true_impairment == "none"]
            
            impaired_rewards = [patient_rewards.get(p.id, 0.0) for p in finished_impaired]
            non_imp_rewards = [patient_rewards.get(p.id, 0.0) for p in finished_non_impaired]

            results["total_patients_finished"] = len(finished_patients)
            results["finished_impaired_count"] = len(finished_impaired)
            results["finished_non_impaired_count"] = len(finished_non_impaired)
            results["avg_reward_impaired"] = np.mean(impaired_rewards) if impaired_rewards else 0.0
            results["avg_reward_non_impaired"] = np.mean(non_imp_rewards) if non_imp_rewards else 0.0
            results["reward_fairness_gap"] = abs(results["avg_reward_non_impaired"] - results["avg_reward_impaired"])

            # Task-specific metrics
            results["triage_routing_accuracy"] = np.mean(sim.routing_decisions) if sim.routing_decisions else 0.0
            results["escort_avg_patient_wait_time"] = np.mean(sim.escort_wait_times) if sim.escort_wait_times else 0.0
            results["manager_avg_ward_wait_time"] = np.mean(sim.ward_wait_times) if sim.ward_wait_times else 0.0
            results["manager_doctor_moves"] = sim.doctor_moves_count
            
            return results

        metrics_w0 = get_detailed_metrics(env_w0)
        metrics_w1 = get_detailed_metrics(env_w1)

        # Combine results into a single dictionary for this episode
        episode_summary = {"episode_seed": episode_seed}
        for k, v in metrics_w0.items(): episode_summary[f"W0_{k}"] = v
        for k, v in metrics_w1.items(): episode_summary[f"W1_{k}"] = v
        
        # Overall counterfactual fairness gap (total reward W0 vs W1)
        total_reward_w0 = metrics_w0["avg_reward_impaired"] * metrics_w0["finished_impaired_count"] + \
                          metrics_w0["avg_reward_non_impaired"] * metrics_w0["finished_non_impaired_count"]
        total_reward_w1 = metrics_w1["avg_reward_impaired"] * metrics_w1["finished_impaired_count"] + \
                          metrics_w1["avg_reward_non_impaired"] * metrics_w1["finished_non_impaired_count"]
        episode_summary["CF_Reward_Gap"] = abs(total_reward_w0 - total_reward_w1)
        
        all_results.append(episode_summary)

    # Save results
    os.makedirs(config["output_dir"], exist_ok=True)
    df = pd.DataFrame(all_results)
    
    output_filename = f"fairness_results_CF_{config['alpha']}_{config['beta']}.pkl"
    output_path = os.path.join(config["output_dir"], output_filename)
    df.to_pickle(output_path)

    print(f"\nTest results DataFrame saved to {output_path}")
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test trained Fair-PPO CF agents.")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value of the trained model.")
    parser.add_argument("--beta", type=float, required=True, help="Beta value of the trained model.")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing the final .pth weight files for the specified alpha/beta.")
    parser.add_argument("--output_dir", type=str, default="test_results_CF", help="Root directory to save the test result files.")
    parser.add_argument("--num_test_episodes", type=int, default=100, help="Number of episodes to run for testing.")
    parser.add_argument("--test_seed", type=int, default=42, help="Seed for the test run for reproducibility.")
    parser.add_argument("--cuda", type=str, default="cuda:0", help="CUDA device.")
    args = parser.parse_args()

    config = {
        "alpha": args.alpha, "beta": args.beta,
        "weights_dir": args.weights_dir, "output_dir": args.output_dir,
        "num_test_episodes": args.num_test_episodes,
        "device": torch.device(args.cuda if torch.cuda.is_available() else "cpu"),
        "test_seed": args.test_seed,
        "gamma": 0.99, "n_clerks": 30, "m_nurses": 60, "k_robots": 30, "n_triage": 30,
        "n_swing_doctors": 18, "n_patients": 300, "sim_hours": 12,
    }
    
    test_rng = random.Random(config["test_seed"])
    
    # Generate Test Patient Schedules
    print(f"Generating {config['num_test_episodes']} sets of test patient schedules...")
    arrival_templates = generate_arrival_templates(n_templates=20, sim_minutes=config["sim_hours"] * 60, seed=config["test_seed"])
    
    test_patient_schedules_w0, test_patient_schedules_w1 = [], []
    illnesses = ["emergency", "cardio", "general", "pediatric", "psychiatric", "xray"]
    priorities = ["low", "medium", "high"]
    impair_lvl = ["none", "low", "high"]
    
    for i in range(config["num_test_episodes"]):
        template = test_rng.choice(arrival_templates)
        arrival_times = build_arrival_schedule(config["n_patients"], template["peaks"], sim_minutes=config["sim_hours"] * 60)
        
        episode_schedule_w0, episode_schedule_w1 = [], []
        for patient_idx, arrival_time in enumerate(arrival_times):
            base_priority = test_rng.choice(priorities)
            base_illness = test_rng.choice(illnesses)
            base_impairment = test_rng.choices(impair_lvl, weights=[0.6, 0.25, 0.15])[0]

            patient_w0 = Patient(400 + patient_idx, priority=base_priority, position="outside", illness=base_illness, impairment_level=base_impairment, arrival_time=arrival_time)
            episode_schedule_w0.append(patient_w0)

            if base_impairment == "none": cf_impairment = "high"
            elif base_impairment == "high": cf_impairment = "none"
            else: cf_impairment = "low"
            
            patient_w1 = Patient(400 + patient_idx, priority=base_priority, position="outside", illness=base_illness, impairment_level=cf_impairment, arrival_time=arrival_time)
            episode_schedule_w1.append(patient_w1)

        test_patient_schedules_w0.append(episode_schedule_w0)
        test_patient_schedules_w1.append(episode_schedule_w1)

    env_keys = ["n_clerks", "n_patients", "m_nurses", "k_robots", "n_triage", "n_swing_doctors", "sim_hours", "gamma"]
    env_config = {key: config[key] for key in env_keys}
    env_w0 = HospitalMAS(verbose=False, **env_config)
    env_w1 = HospitalMAS(verbose=False, **env_config)
    
    env_w0.reset(seed=config["run_seed"])
    env_w1.reset(seed=config["run_seed"] + 1) # Use a different seed for the second env
    
    agents_w0, agents_w1 = {}, {}
    agent_input_dims = {"escort_dispatcher": 6, "triage_router": 18, "doctor_manager": 19}
    
    print(f"Loading weights from: {config['weights_dir']}")
    for name, input_dim in agent_input_dims.items():
        agent_config = {'lr': 0, 'gamma': 0, 'eps_clip': 0, 'k_epochs': 0, 'batch_size': 0, 'alpha': 0, 'beta': 0, 'value_loss_coef': 0, 'entropy_coef': 0, 'device': config["device"]}
        
        # Load W0 agent
        agents_w0[name] = PPOAgentMDCF(input_dim=input_dim, action_space=env_w0.action_spaces[name], **agent_config)
        weights_path_w0 = os.path.join(config['weights_dir'], f'w0_{name}_weights.pth')
        if os.path.exists(weights_path_w0):
            agents_w0[name].policy.load_state_dict(torch.load(weights_path_w0, map_location=config["device"]))
        else:
            sys.exit(f"ERROR: Weight file not found for W0 agent '{name}' at {weights_path_w0}")

        # Load W1 agent
        agents_w1[name] = PPOAgentMDCF(input_dim=input_dim, action_space=env_w1.action_spaces[name], **agent_config)
        weights_path_w1 = os.path.join(config['weights_dir'], f'w1_{name}_weights.pth')
        if os.path.exists(weights_path_w1):
            agents_w1[name].policy.load_state_dict(torch.load(weights_path_w1, map_location=config["device"]))
        else:
            sys.exit(f"ERROR: Weight file not found for W1 agent '{name}' at {weights_path_w1}")

    print("All agent weights loaded successfully.")
    results_df = test_ppo_cf(agents_w0, agents_w1, config, test_rng, env_w0, env_w1, test_patient_schedules_w0, test_patient_schedules_w1)