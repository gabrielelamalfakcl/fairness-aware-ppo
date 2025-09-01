import argparse
import torch
import numpy as np
from tqdm import tqdm
import wandb
import os
import random
from collections import defaultdict, deque
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
from PPOAgentMDCSP import PPOAgentMDCSP, Memory
from environments.Hospital.utils import generate_arrival_templates 

def sort_templates_by_difficulty(templates):
    """
    Sorts arrival templates based on a difficulty score based on the number and duration of peaks
    """
    
    def get_difficulty_score(template):
        total_peak_duration = sum(end - start for start, end in template["peaks"])
        return len(template["peaks"]) * total_peak_duration

    return sorted(templates, key=get_difficulty_score)

def train_ppo_csp(agents, config, arrival_templates, start_episode, env: HospitalMAS):
    """
    Trains Fair-PPO for the Hospital simulation using Conditional Stat Parity
    with curriculum learning approach.
    """
    
    memories = {name: Memory() for name in agents.keys()}
    run_rng = random.Random(config["run_seed"])
    
    # Curriculum Learning Setup
    sorted_templates = sort_templates_by_difficulty(arrival_templates)
    n_total_templates = len(sorted_templates)
    phase1_end, phase2_end = 300, 600
    easy_pool = sorted_templates[:max(1, int(n_total_templates * 0.3))]
    medium_pool = sorted_templates[:max(1, int(n_total_templates * 0.7))]
    hard_pool = sorted_templates
    
    # Efficient rolling averages
    recent_rewards = deque(maxlen=100)
    recent_finished = deque(maxlen=100)
    recent_gaps = deque(maxlen=100)
    all_gap_high = deque(maxlen=100)
    all_gap_medium = deque(maxlen=100)
    all_gap_low = deque(maxlen=100)
        
            
    for episode in tqdm(range(start_episode, config["num_episodes"])):
        # Curriculum Learning
        if episode < phase1_end:
            env.arrival_template_override = random.choice(easy_pool)
        elif episode < phase2_end:
            env.arrival_template_override = random.choice(medium_pool)
        else:
            env.arrival_template_override = random.choice(hard_pool)
        
        episode_seed = run_rng.randint(0, 1_000_000_000)
        obs, _ = env.reset(seed=episode_seed)
        done = {"__all__": False}
        total_episode_reward = 0.0
        state_values_by_group = defaultdict(list)
        
        # Run the simulation for one episode
        while not done["__all__"]:
            step_data = {}
            last_obs = obs
            for agent_name, agent_obs in last_obs.items():
                if agent_obs.sum() > 0:
                    state_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0).to(config["device"])
                    action, log_prob, state_value = agents[agent_name].select_actions(state_tensor)
                    step_data[agent_name] = {"action": action[0], "log_prob": log_prob, "state_value": state_value}
                    
                    if agent_name == "triage_router":
                        patient_being_routed = env.sim._payload.get("patient")
                        if patient_being_routed:
                            mem_idx = len(memories[agent_name].states)
                            patient_being_routed.triage_memory_idx = mem_idx
            
            action_dict = {name: data["action"] for name, data in step_data.items()}
            obs, rewards, terminated, truncated, infos = env.step(action_dict)
            done = terminated
            total_episode_reward += sum(rewards.values())

            # Update tensors at the end of an episode
            for agent_name, data in step_data.items():
                mem = memories[agent_name]
                mem.states.append(torch.tensor(last_obs[agent_name], dtype=torch.float32))
                mem.actions.append(data["action"])
                mem.logprobs.append(data["log_prob"])
                
                mem.rewards.append([rewards.get(agent_name, 0.0)])
                mem.state_values.append(data["state_value"])
                
                if agent_name in ["triage_router", "escort_dispatcher"] and env.sim._payload and "patient" in env.sim._payload:
                    patient = env.sim._payload["patient"]
                    if hasattr(patient, 'priority'):
                        impairment_status = "impaired" if patient.true_impairment != "none" else "non_impaired"
                        group_key = f"{patient.priority.lower()}_{impairment_status}"
                        state_values_by_group[group_key].append(data["state_value"])
                    
        # A) Episode metric calculation
        patient_rewards = env.sim.drain_patient_rewards()
        finished = [p for p in env.sim.patients if p.status in ("done", "exited")]
        finished_impaired_count = len([p for p in finished if p.true_impairment != "none"])
        finished_non_impaired_count = len([p for p in finished if p.true_impairment == "none"])

        impaired_rewards = [patient_rewards[p.id] for p in finished if p.true_impairment != "none"]
        non_imp_rewards  = [patient_rewards[p.id] for p in finished if p.true_impairment == "none"]
        avg_reward_impaired = np.mean(impaired_rewards) if impaired_rewards else 0.0
        avg_reward_non_impaired = np.mean(non_imp_rewards) if non_imp_rewards else 0.0
        
        finished_count = len(finished)
        fairness_gap = abs(avg_reward_non_impaired - avg_reward_impaired)
        
        recent_rewards.append(total_episode_reward)
        recent_finished.append(finished_count)
        recent_gaps.append(fairness_gap)
        
        # B) Agent-specific sask metrics
        sim = env.sim
        routing_accuracy = np.mean(sim.routing_decisions) if sim.routing_decisions else 0.0
        avg_escort_wait_time = np.mean(sim.escort_wait_times) if sim.escort_wait_times else 0.0
        avg_helper_travel_time = np.mean(sim.helper_travel_times) if sim.helper_travel_times else 0.0
        avg_ward_wait_time = np.mean(sim.ward_wait_times) if sim.ward_wait_times else 0.0
        doctor_moves = sim.doctor_moves_count
        
        routing_outcomes = env.sim.routing_outcomes
        total_routings = len(routing_outcomes)
        if total_routings > 0:
            perfect_pct = routing_outcomes.count('perfect') * 100 / total_routings
            acceptable_pct = routing_outcomes.count('acceptable') * 100 / total_routings
            wrong_pct = routing_outcomes.count('wrong') * 100 / total_routings
        else:
            perfect_pct, acceptable_pct, wrong_pct = 0, 0, 0
        
        rewards_by_group_lists = defaultdict(list)
        for p in finished:
            if hasattr(p, 'true_priority') and p.id in patient_rewards:
                impairment_status = "impaired" if p.true_impairment != "none" else "non_impaired"
                group_key = f"{p.true_priority.lower()}_{impairment_status}"
                rewards_by_group_lists[group_key].append(patient_rewards[p.id])
        
        avg_rewards_by_group = {group: np.mean(rewards) if rewards else 0.0 for group, rewards in rewards_by_group_lists.items()}
        gap_high = abs(avg_rewards_by_group.get('high_impaired', 0.0) - avg_rewards_by_group.get('high_non_impaired', 0.0))
        gap_medium = abs(avg_rewards_by_group.get('medium_impaired', 0.0) - avg_rewards_by_group.get('medium_non_impaired', 0.0))
        gap_low = abs(avg_rewards_by_group.get('low_impaired', 0.0) - avg_rewards_by_group.get('low_non_impaired', 0.0))
        finished_count = sum(1 for p in env.sim.patients if p.status in ["done", "exited"])

        all_gap_high.append(gap_high)
        all_gap_medium.append(gap_medium)
        all_gap_low.append(gap_low)
        
        # Wandb
        log_data = {
            "Episode": episode,
            "Per-Episode/Total Reward": total_episode_reward,
            "Per-Episode/Finished Patients": finished_count,
            "Counts/Finished Impaired": finished_impaired_count,
            "Counts/Finished Non-Impaired": finished_non_impaired_count,
            "Per-Episode/Fairness Gap": fairness_gap,
            
            "Rolling-Average/Avg Total Reward": np.mean(recent_rewards),
            "Rolling-Average/Avg Finished Patients": np.mean(recent_finished),
            "Rolling-Average/Avg Fairness Gap": np.mean(recent_gaps),
            
            "Reward Details/Finished Impaired (Avg)": avg_reward_impaired,
            "Reward Details/Finished Non-Impaired (Avg)": avg_reward_non_impaired,
            
            # Add parity within the sub-groups (by priority)
            "Cumulative/Avg Gap (High Prio)": np.mean(all_gap_high),
            "Cumulative/Avg Gap (Medium Prio)": np.mean(all_gap_medium),
            "Cumulative/Avg Gap (Low Prio)": np.mean(all_gap_low),
            
            # Add new task-specific metrics
            "Task-Metrics/Triage_Routing_Accuracy": routing_accuracy,
            "Task-Metrics/Escort_Avg_Patient_Wait_Time": avg_escort_wait_time,
            "Task-Metrics/Escort_Avg_Helper_Travel_Time": avg_helper_travel_time,
            "Task-Metrics/Manager_Avg_Ward_Wait_Time": avg_ward_wait_time,
            "Task-Metrics/Manager_Doctor_Moves": doctor_moves,
            
            "Routing-Breakdown/Perfect_Match_Pct": perfect_pct,
            "Routing-Breakdown/Acceptable_Match_Pct": acceptable_pct,
            "Routing-Breakdown/Wrong_Match_Pct": wrong_pct,
        }
        
        for group, avg_reward in avg_rewards_by_group.items():
            # Example group: "high_non_impaired"
            parts = group.split('_')
            priority = parts[0].capitalize()
            status = "Non-Impaired" if "non" in group else "Impaired"
            log_key = f"AvgRewardByGroup/{priority} Prio/{status}"
            log_data[log_key] = avg_reward

        for group, rewards_list in rewards_by_group_lists.items():
            parts = group.split('_')
            priority = parts[0].capitalize()
            status = "Non-Impaired" if "non" in group else "Impaired"
            log_key = f"CountByGroup/{priority} Prio/{status}"
            log_data[log_key] = len(rewards_list)

        for agent_name, agent_obj in agents.items():
            if len(memories[agent_name].states) > 0:
                loss = agent_obj.update(memories[agent_name], avg_rewards_by_group, state_values_by_group)
                if loss is not None:
                    log_data[f"Loss/{agent_name}"] = loss.item() if hasattr(loss, 'item') else loss

        wandb.log(log_data)

        for mem in memories.values(): 
            mem.clear_memory()

        if episode > 0 and (episode % config["save_every"] == 0 or episode == config["num_episodes"] - 1):
            ckpt_dir = f"ppo_weights_dp/alpha={config['alpha']}_beta={config['beta']}"
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = f"{ckpt_dir}/ckpt_ep{episode}.pth"
            checkpoint = {name: agent.get_state_dict() for name, agent in agents.items()}
            torch.save(checkpoint, ckpt_path)
                    
if __name__ == '__main__':
    try:
        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
             sys.path.append(str(project_root))
    except (IndexError, NameError):
        print("Could not add project root to path.")
        
    parser = argparse.ArgumentParser(description="Run Fair PPO (CSP) training for Hospital MAS")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha for reward fairness")
    parser.add_argument("--beta", type=float, required=True, help="Beta for state-value fairness")
    parser.add_argument("--num_episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--save_every", type=int, default=100, help="Frequency of saving checkpoints.")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to checkpoint file to resume.")
    parser.add_argument("--wandb_run_name", type=str, default="hospital-csp-run", help="Run name for wandb")
    parser.add_argument("--cuda", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--run_seed", type=int, default=None, help="Seed for the entire training run for reproducibility.")
    args = parser.parse_args()
   
    config = {
        "alpha": args.alpha, "beta": args.beta, "num_episodes": args.num_episodes,
        "save_every": args.save_every, "device": torch.device(args.cuda if torch.cuda.is_available() else "cpu"),
        "run_seed": args.run_seed,"ppo_lr": 1e-5, "gamma": 0.99, "eps_clip": 0.2, "k_epochs": 10,
        "batch_size": 256, "value_loss_coef": 0.5, "entropy_coef": 0.01,
        "n_clerks": 30, "m_nurses": 60, "k_robots": 30, "n_triage": 30,
        "n_swing_doctors": 18, "n_patients": 300, "sim_hours": 12,
        "wandb_project": "Hospital-FairPPO-CSP", 
        "wandb_run_name": args.wandb_run_name
    }
    
    arrival_templates = generate_arrival_templates(n_templates=5, sim_minutes=config["sim_hours"] * 60)

    env_keys = ["n_clerks", "n_patients", "m_nurses", "k_robots", "n_triage", "n_swing_doctors", "sim_hours", "gamma"]
    env_config = {key: config[key] for key in env_keys if key in config}
    env_config['seed'] = None  
    env_config['render_mode'] = False 
    env_config['force_impairment'] = None 
    env_config['verbose'] = False 

    main_env = HospitalMAS(**env_config)
    
    agent_config_keys = ["ppo_lr", "gamma", "eps_clip", "k_epochs", "batch_size", "value_loss_coef", "entropy_coef", "alpha", "beta", "device"]
    agent_config = {key: config[key] for key in agent_config_keys}
    agent_config['lr'] = agent_config.pop('ppo_lr') # Rename key for agent constructor

    agents = {}
    agent_input_dims = {"escort_dispatcher": 6, "triage_router": 18, "doctor_manager": 19}
    for name, input_dim in agent_input_dims.items():
        agents[name] = PPOAgentMDCSP(
            input_dim=input_dim,
            action_space=main_env.action_spaces[name],
            **agent_config
        )

    start_episode = 0
    if args.resume_ckpt:
        print(f"Resuming training from checkpoint: {args.resume_ckpt}")
        checkpoint = torch.load(args.resume_ckpt, map_location=config["device"])
        for name, agent in agents.items():
            if name in checkpoint: agent.load_state_dict(checkpoint[name])
        try:
            start_episode = int(args.resume_ckpt.split('_ep')[1].split('.pth')[0]) + 1
        except:
            print("Warning: Could not parse episode number from checkpoint. Starting from episode 0.")
    
    wandb.init(
        project=config["wandb_project"],
        name=f"{config['wandb_run_name']}-alpha_{config['alpha']}-beta_{config['beta']}",
        config=config, resume="allow"
    )

    for agent in agents.values():
        wandb.watch(agent.policy, log="all", log_freq=100, log_graph=True)
    
    train_ppo_csp(agents, config, arrival_templates, start_episode, main_env)
    
    # Save the weights   
    weights_dir = f"fairPPO_weights/csp/alpha={config['alpha']}_beta={config['beta']}"
    os.makedirs(weights_dir, exist_ok=True)
    for agent_name, agent_obj in agents.items():
        weights_file_path = os.path.join(weights_dir, f'{agent_name}_weights.pth')
        agent_obj.save_weights(weights_file_path)

    wandb.finish()
    print('Training finished.')