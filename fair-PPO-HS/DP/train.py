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
from PPOAgentMDDP import PPOAgentMDDP, Memory
from environments.Hospital.utils import generate_arrival_templates

def write_patient_journey_report(env, episode, log_all_patients=True):
    """
    Writes a detailed journey log for all patients to a file
    """
    
    # log journey directory
    log_dir = "journey_logs_all"
    os.makedirs(log_dir, exist_ok=True)
    
    report_path = os.path.join(log_dir, f"episode_{episode}_report.txt")
    
    with open(report_path, "w") as f:
        f.write(f" PATIENT JOURNEY REPORT FOR EPISODE {episode} \n\n")
        
        # Sorting by patients'  ID
        all_patients_sorted = sorted(env.sim.patients, key=lambda p: p.id)
        
        for patient in all_patients_sorted:
            f.write("="*60 + "\n")
            f.write(f"Patient ID: {patient.id}\n")
            f.write(f"- Ground Truth Priority: {patient.true_priority}\n")
            f.write(f"- Ground Truth Impairment: {patient.true_impairment}\n")
            f.write(f"- Ground Truth Illness: {patient.true_illness}\n")
            f.write(f"- Final Status: {patient.status}\n")
            f.write(f"- Total Reward: {env.sim._patient_rewards.get(patient.id, 0.0):.2f}\n")
            f.write("-"*60 + "\n")
            f.write("Event Log:\n")
            for time, event in getattr(patient, "event_log", []):
                f.write(f"- [t={time:03d}] {event}\n")
            f.write("\n")
            
def sort_templates_by_difficulty(templates):
    """
    Sorts arrival templates based on a difficulty score based on the number and duration of peaks
    """
    
    def get_difficulty_score(template):
        total_peak_duration = sum(end - start for start, end in template["peaks"])
        return len(template["peaks"]) * total_peak_duration

    return sorted(templates, key=get_difficulty_score)

def train_ppo(agents, config, arrival_templates, start_episode, env: HospitalMAS):
    """
    Trains Fair-PPO for the Hospital simulation using Demographic Parity
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
    
    # Rolling averages
    recent_rewards = deque(maxlen=100)
    recent_finished = deque(maxlen=100)
    recent_gaps = deque(maxlen=100)
    
    # Main training loop  
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
        sv_for_impaired, sv_for_non_impaired = [], []
        
        # Run the simulation for one episode
        while not done["__all__"]:
            step_data = {}
            last_obs = obs
            for agent_name, agent_obs in last_obs.items():
                if agent_obs.sum() > 0:
                    state_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0).to(config["device"])
                    action, log_prob, state_value = agents[agent_name].select_actions(state_tensor)
                    step_data[agent_name] = {"action": action[0], "log_prob": log_prob, "state_value": state_value}
                    
                    if agent_name == "triage_router" and env.sim._payload and "patient" in env.sim._payload:
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
                    if patient.true_impairment != "none": sv_for_impaired.append(data["state_value"])
                    else: sv_for_non_impaired.append(data["state_value"])
        
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
            
            "Task-Metrics/Triage_Routing_Accuracy": routing_accuracy,
            "Task-Metrics/Escort_Avg_Patient_Wait_Time": avg_escort_wait_time,
            "Task-Metrics/Escort_Avg_Helper_Travel_Time": avg_helper_travel_time,
            "Task-Metrics/Manager_Avg_Ward_Wait_Time": avg_ward_wait_time,
            "Task-Metrics/Manager_Doctor_Moves": doctor_moves,
            
            "Routing-Breakdown/Perfect_Match_Pct": perfect_pct,
            "Routing-Breakdown/Acceptable_Match_Pct": acceptable_pct,
            "Routing-Breakdown/Wrong_Match_Pct": wrong_pct,
        }

        for agent_name, agent_obj in agents.items():
            if len(memories[agent_name].states) > 0:
                loss = agent_obj.update(memories[agent_name], avg_reward_non_impaired, avg_reward_impaired, sv_for_non_impaired, sv_for_impaired)
                if loss is not None:
                    log_data[f"Loss/{agent_name}"] = loss.item() if hasattr(loss, 'item') else loss
        
        wandb.log(log_data)

        for mem in memories.values(): 
            mem.clear_memory()
        
        # checkpoint weights save
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

    parser = argparse.ArgumentParser(description="Run Fair PPO (DP) training for Hospital MAS")
    parser.add_argument("--alpha", type=float, default=0.0, help="Alpha for reward fairness")
    parser.add_argument("--beta", type=float, default=0.0, help="Beta for state-value fairness")
    parser.add_argument("--num_episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--save_every", type=int, default=100, help="Frequency of saving checkpoints.")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to checkpoint file to resume.")
    parser.add_argument("--wandb_run_name", type=str, default="hospital-dp-run", help="Run name for wandb")
    parser.add_argument("--cuda", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--run_seed", type=int, default=42, help="Seed for the entire training run for reproducibility.")
    args = parser.parse_args()
   
    config = {
        "alpha": args.alpha, "beta": args.beta, "num_episodes": args.num_episodes,
        "save_every": args.save_every, "device": torch.device(args.cuda if torch.cuda.is_available() else "cpu"),
        "run_seed": args.run_seed, "ppo_lr": 1e-5, "gamma": 0.99, "eps_clip": 0.2, "k_epochs": 10,
        "batch_size": 256, "value_loss_coef": 0.5, "entropy_coef": 0.01,
        "n_clerks": 30, "m_nurses": 60, "k_robots": 30, "n_triage": 30,
        "n_swing_doctors": 18, "n_patients": 300, "sim_hours": 12,
        "wandb_project": "[FINAL]Hospital-FairPPO-DP", 
        "wandb_run_name": args.wandb_run_name
    }
    
    arrival_templates = generate_arrival_templates(n_templates=5, sim_minutes=config["sim_hours"] * 60) # arrival templates to generate

    env_keys = ["n_clerks", "n_patients", "m_nurses", "k_robots", "n_triage", "n_swing_doctors", "sim_hours", "gamma"]
    env_config = {key: config[key] for key in env_keys if key in config}
    env_config['seed'] = None
    env_config['render_mode'] = False
    env_config['force_impairment'] = None
    env_config['verbose'] = False

    main_env = HospitalMAS(**env_config)
    
    agent_config_keys = ["ppo_lr", "gamma", "eps_clip", "k_epochs", "batch_size", "value_loss_coef", "entropy_coef", "alpha", "beta", "device"]
    agent_config = {key: config[key] for key in agent_config_keys}
    agent_config['lr'] = agent_config.pop('ppo_lr')

    agents = {}
    agent_input_dims = {"escort_dispatcher": 6, "triage_router": 18, "doctor_manager": 19}
    
    # For each learning agent, instantiate a multi discrete choice agent
    for name, input_dim in agent_input_dims.items():
        agents[name] = PPOAgentMDDP(
            input_dim=input_dim,
            action_space=main_env.action_spaces[name],
            **agent_config
        )

    start_episode = 0
    if args.resume_ckpt:
        checkpoint = torch.load(args.resume_ckpt, map_location=config["device"])
        for name, agent in agents.items():
            if name in checkpoint: agent.load_state_dict(checkpoint[name])
        try:
            start_episode = int(args.resume_ckpt.split('_ep')[1].split('.pth')[0]) + 1
        except:
            print("Warning: Could not parse episode number from checkpoint. Starting from episode 0.")
    
    wandb.init(
        project=config["wandb_project"],
        name=f"{config['wandb_run_name']}-alpha={config['alpha']}-beta={config['beta']}",
        config=config, resume="allow"
    )

    for agent_name, agent in agents.items():
        wandb.watch(agent.policy, log="all", log_freq=100, log_graph=True)
    
    train_ppo(agents, config, arrival_templates, start_episode, main_env)

    # Save the weights        
    weights_dir = f"fairPPO_weights/dp/alpha={config['alpha']}_beta={config['beta']}"
    os.makedirs(weights_dir, exist_ok=True)
    for agent_name, agent_obj in agents.items():
        weights_file_path = os.path.join(weights_dir, f'{agent_name}_final_weights.pth')
        torch.save(agent_obj.policy.state_dict(), weights_file_path)

    wandb.finish()
    print('Training finished.')