import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random
from pathlib import Path
import sys

# Path to import other scripts: it needs to be checked based on the folder structure
try:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
except (IndexError, NameError):
    project_root = Path.cwd()
    sys.path.append(str(project_root))

from agent_scripts.SOTOAgent import SOTOAgent
from environments.Hospital.hospital_pz_env import HospitalMAS
from environments.Hospital.utils import generate_arrival_templates

def test_soto(agents: dict, config: dict, test_rng: random.Random, env: HospitalMAS):
    """
    Test SOTO (alpha-fairness hyperparams)
    """
    
    for agent in agents.values():
        agent.eval() # Set agents to evaluation mode for deterministic behavior (reproducibility - see random seed)

    sim_minutes = config["sim_hours"] * 60
    all_episode_results = []

    print(f"Running {config['num_test_episodes']} test episodes...")
    for episode in tqdm(range(config["num_test_episodes"])):
        for agent in agents.values():
            agent.start_episode(episode, config['num_test_episodes'])
            
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
                        # SOTOAgent's select_action is deterministic in eval mode
                        action, _, _, _ = agents[agent_name].select_action(state_tensor)
                        action_dict[agent_name] = action
            
            obs, rewards, terminated, truncated, infos = env.step(action_dict)
            done = terminated

        # Metric calculation
        sim = env.sim
        
        # total finished patients, total finished patients with impairment, total finished patients without impairment  
        finished_patients = [p for p in sim.patients if p.status in ("done", "exited")]
        finished_impaired_count = len([p for p in finished_patients if p.true_impairment != "none"])
        finished_non_impaired_count = len([p for p in finished_patients if p.true_impairment == "none"])
        
        finished_impaired_LP_count = len([p for p in finished_patients if p.true_impairment != "none" and p.true_priority == "low"])
        finished_impaired_MP_count = len([p for p in finished_patients if p.true_impairment != "none" and p.true_priority == "medium"])
        finished_impaired_HP_count = len([p for p in finished_patients if p.true_impairment != "none" and p.true_priority == "high"])

        finished_non_impaired_LP_count = len([p for p in finished_patients if p.true_impairment == "none" and p.true_priority == "low"])
        finished_non_impaired_MP_count = len([p for p in finished_patients if p.true_impairment == "none" and p.true_priority == "medium"])
        finished_non_impaired_HP_count = len([p for p in finished_patients if p.true_impairment == "none" and p.true_priority == "high"])


        wt_escort_imp = [p.escort_wait_time for p in finished_patients if p.true_impairment != "none" and hasattr(p, 'escort_wait_time')]
        wt_escort_non = [p.escort_wait_time for p in finished_patients if p.true_impairment == "none" and hasattr(p, 'escort_wait_time')]
        avg_wt_escort_imp = np.mean(wt_escort_imp) if wt_escort_imp else 0.0
        avg_wt_escort_non = np.mean(wt_escort_non) if wt_escort_non else 0.0
        escort_wait_fairness_gap = abs(avg_wt_escort_non - avg_wt_escort_imp)

        patient_rewards = sim.drain_patient_rewards()
        impaired_rewards = [patient_rewards.get(p.id, 0.0) for p in finished_patients if p.true_impairment != "none"]
        non_imp_rewards  = [patient_rewards.get(p.id, 0.0) for p in finished_patients if p.true_impairment == "none"]
        
        impaired_rewards_LP = [patient_rewards.get(p.id, 0.0) for p in finished_patients if p.true_impairment != "none" and p.true_priority == "low"]
        impaired_rewards_MP = [patient_rewards.get(p.id, 0.0) for p in finished_patients if p.true_impairment != "none" and p.true_priority == "medium"]
        impaired_rewards_HP = [patient_rewards.get(p.id, 0.0) for p in finished_patients if p.true_impairment != "none" and p.true_priority == "high"]
 
        non_impaired_rewards_LP = [patient_rewards.get(p.id, 0.0) for p in finished_patients if p.true_impairment == "none" and p.true_priority == "low"]
        non_impaired_rewards_MP = [patient_rewards.get(p.id, 0.0) for p in finished_patients if p.true_impairment == "none" and p.true_priority == "medium"]
        non_impaired_rewards_HP = [patient_rewards.get(p.id, 0.0) for p in finished_patients if p.true_impairment == "none" and p.true_priority == "high"]
        
        avg_reward_impaired = np.mean(impaired_rewards) if impaired_rewards else 0.0
        avg_reward_non_impaired = np.mean(non_imp_rewards) if non_imp_rewards else 0.0
        
        avg_reward_impaired_LP = np.mean(impaired_rewards_LP) if impaired_rewards_LP else 0.0
        avg_reward_impaired_MP = np.mean(impaired_rewards_MP) if impaired_rewards_MP else 0.0
        avg_reward_impaired_HP = np.mean(impaired_rewards_HP) if impaired_rewards_HP else 0.0

        avg_reward_non_impaired_LP = np.mean(non_impaired_rewards_LP) if non_impaired_rewards_LP else 0.0
        avg_reward_non_impaired_MP = np.mean(non_impaired_rewards_MP) if non_impaired_rewards_MP else 0.0
        avg_reward_non_impaired_HP = np.mean(non_impaired_rewards_HP) if non_impaired_rewards_HP else 0.0

        reward_fairness_gap = abs(avg_reward_non_impaired - avg_reward_impaired)
        
        reward_fairness_gap_LP = abs(avg_reward_non_impaired_LP - avg_reward_impaired_LP)
        reward_fairness_gap_MP = abs(avg_reward_non_impaired_MP - avg_reward_impaired_MP)
        reward_fairness_gap_HP = abs(avg_reward_non_impaired_HP - avg_reward_impaired_HP)

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
            
        # Wandb
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
            
            "finished_impaired_LP_count": finished_impaired_LP_count,
            "finished_impaired_MP_count": finished_impaired_MP_count,
            "finished_impaired_HP_count": finished_impaired_HP_count,
            "finished_non_impaired_LP_count": finished_non_impaired_LP_count,
            "finished_non_impaired_MP_count": finished_non_impaired_MP_count,
            "finished_non_impaired_HP_count": finished_non_impaired_HP_count,
            
            "reward_fairness_gap_LP": reward_fairness_gap_LP,
            "reward_fairness_gap_MP": reward_fairness_gap_MP,
            "reward_fairness_gap_HP": reward_fairness_gap_HP,
            
        })

    # Save results
    os.makedirs(config["output_dir"], exist_ok=True)
    df = pd.DataFrame(all_episode_results)
    
    ckpt_name = Path(config["ckpt_path"]).stem
    output_filename = f"soto_results_{ckpt_name}_alpha={config['alpha']}.pkl"
    output_path = os.path.join(config["output_dir"], output_filename)
    df.to_pickle(output_path)

    print(f"\nTest results DataFrame saved to {output_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SOTO testing for Hospital MAS")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value of the trained model.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Full path to the .pth checkpoint file.")
    parser.add_argument("--output_dir", type=str, default="soto_test_results", help="Directory to save the test result files.")
    parser.add_argument("--num_test_episodes", type=int, default=500, help="Number of episodes to run for testing.")
    parser.add_argument("--cuda", type=str, default="cuda:0", help="CUDA device.")
    parser.add_argument("--test_seed", type=int, default=42, help="Seed for the test run for reproducibility.")
    args = parser.parse_args()

    config = {
        "alpha": args.alpha, "num_test_episodes": args.num_test_episodes,
        "ckpt_path": args.ckpt_path, "output_dir": args.output_dir,
        "device": torch.device(args.cuda if torch.cuda.is_available() else "cpu"),
        "test_seed": args.test_seed, "gamma": 0.99,
        "n_clerks": 30, "m_nurses": 60, "k_robots": 30, "n_triage": 30,
        "n_swing_doctors": 18, "n_patients": 300, "sim_hours": 12,
    }
    

    env_keys = ["n_clerks", "n_patients", "m_nurses", "k_robots", "n_triage", "n_swing_doctors", "sim_hours", "gamma"]
    env_config = {key: config[key] for key in env_keys if key in config}
    main_env = HospitalMAS(verbose=False, **env_config)

    agents = {}
    agent_input_dims = {"escort_dispatcher": 6, "triage_router": 18, "doctor_manager": 19}
    
    print("Initializing SOTO agents...")
    for name, input_dim in agent_input_dims.items():
        agents[name] = SOTOAgent(
            input_dim=input_dim,
            action_space=main_env.action_spaces[name],
            num_players=len(agent_input_dims),
            device=config["device"]
        )
    
    print(f"Loading checkpoint from: {config['ckpt_path']}")
    if os.path.exists(config['ckpt_path']):
        checkpoint = torch.load(config['ckpt_path'], map_location=config["device"])
        for name, agent in agents.items():
            if name in checkpoint:
                agent.load_state_dict(checkpoint[name])
                print(f"Successfully loaded weights for {name}")
            else:
                print(f"Agent '{name}' not found in checkpoint file.")
                sys.exit(1)
    else:
        print(f"Checkpoint file not found at '{config['ckpt_path']}'")
        sys.exit(1)
        
    test_rng = random.Random(config["test_seed"])
    test_soto(agents, config, test_rng, main_env)
    print('Testing finished.')