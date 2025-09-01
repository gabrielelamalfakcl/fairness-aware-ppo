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
from PPOAgentMDCF import PPOAgentMDCF, Memory
from environments.Hospital.utils import generate_arrival_templates, build_arrival_schedule
from environments.Hospital.players import Patient

def sort_templates_by_difficulty(templates):
    """
    Sorts arrival templates based on a difficulty score.
    A higher score means more patients arriving in shorter, more intense peaks.
    """
    def get_difficulty_score(template):
        if not template["peaks"]:
            return 0
        total_peak_duration = sum(end - start for start, end in template["peaks"])
        # Avoid division by zero; a shorter duration is harder
        return len(template["peaks"]) / (total_peak_duration + 1e-6)

    return sorted(templates, key=get_difficulty_score)

def train_ppo_cf(agents_w0, agents_w1, config, arrival_templates, start_episode, env_w0: HospitalMAS, env_w1: HospitalMAS, all_patient_schedules_w0, all_patient_schedules_w1):
    """
    Trains two PPO policies with curriculum learning and detailed logging.
    """
    memories_w0 = {name: Memory() for name in agents_w0.keys()}
    memories_w1 = {name: Memory() for name in agents_w1.keys()}
    run_rng = random.Random(config["run_seed"])

    # Curriculum Learning Setup
    sorted_templates = sort_templates_by_difficulty(arrival_templates)
    n_total_templates = len(sorted_templates)
    phase1_end = int(config["num_episodes"] * 0.3)
    phase2_end = int(config["num_episodes"] * 0.6)
    easy_pool = sorted_templates[:max(1, int(n_total_templates * 0.3))]
    medium_pool = sorted_templates[:max(1, int(n_total_templates * 0.7))]
    hard_pool = sorted_templates
    
    # Rolling Averages Setup
    recent_rewards_w0 = deque(maxlen=100)
    recent_rewards_w1 = deque(maxlen=100)
    recent_finished_w0 = deque(maxlen=100)
    recent_finished_w1 = deque(maxlen=100)
    recent_gaps = deque(maxlen=100)

    for episode in tqdm(range(start_episode, config["num_episodes"])):
        # Curriculum Learning: Select difficulty based on episode
        if episode < phase1_end:
            template = run_rng.choice(easy_pool)
        elif episode < phase2_end:
            template = run_rng.choice(medium_pool)
        else:
            template = run_rng.choice(hard_pool)

        # Set the patient schedules for this episode
        # The schedules themselves already contain the counterfactual swap
        env_w0.set_patient_schedule(all_patient_schedules_w0[episode])
        env_w1.set_patient_schedule(all_patient_schedules_w1[episode])
        
        episode_seed = run_rng.randint(0, 1_000_000_000)
        obs_w0, _ = env_w0.reset(seed=episode_seed)
        obs_w1, _ = env_w1.reset(seed=episode_seed)
        
        done_w0, done_w1 = {"__all__": False}, {"__all__": False}
        total_episode_reward_w0, total_episode_reward_w1 = 0.0, 0.0

        while not done_w0["__all__"] and not done_w1["__all__"]:
            # Step World 0 (Factual)
            step_data_w0 = {}
            last_obs_w0 = obs_w0
            for agent_name, agent_obs in last_obs_w0.items():
                if agent_obs.sum() > 0:
                    state_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0).to(config["device"])
                    action, log_prob, state_value = agents_w0[agent_name].select_actions(state_tensor)
                    step_data_w0[agent_name] = {"action": action[0], "log_prob": log_prob, "state_value": state_value}
            
            action_dict_w0 = {name: data["action"] for name, data in step_data_w0.items()}
            obs_w0, rewards_w0, terminated_w0, truncated_w0, infos_w0 = env_w0.step(action_dict_w0)
            done_w0 = terminated_w0

            total_episode_reward_w0 += sum(rewards_w0.values())
            for agent_name, data in step_data_w0.items():
                mem = memories_w0[agent_name]
                mem.states.append(torch.tensor(last_obs_w0[agent_name], dtype=torch.float32))
                mem.actions.append(data["action"])
                mem.logprobs.append(data["log_prob"])
                mem.rewards.append([rewards_w0.get(agent_name, 0.0)])
                mem.state_values.append(data["state_value"])

            # Step World 1 (Counterfactual)
            step_data_w1 = {}
            last_obs_w1 = obs_w1
            for agent_name, agent_obs in last_obs_w1.items():
                if agent_obs.sum() > 0:
                    state_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0).to(config["device"])
                    action, log_prob, state_value = agents_w1[agent_name].select_actions(state_tensor)
                    step_data_w1[agent_name] = {"action": action[0], "log_prob": log_prob, "state_value": state_value}
            
            action_dict_w1 = {name: data["action"] for name, data in step_data_w1.items()}
            obs_w1, rewards_w1, terminated_w1, truncated_w1, infos_w1 = env_w1.step(action_dict_w1)
            done_w1 = terminated_w1

            total_episode_reward_w1 += sum(rewards_w1.values())
            for agent_name, data in step_data_w1.items():
                mem = memories_w1[agent_name]
                mem.states.append(torch.tensor(last_obs_w1[agent_name], dtype=torch.float32))
                mem.actions.append(data["action"])
                mem.logprobs.append(data["log_prob"])
                mem.rewards.append([rewards_w1.get(agent_name, 0.0)])
                mem.state_values.append(data["state_value"])
        
        # End of Episode Logging
        finished_w0 = sum(1 for p in env_w0.sim.patients if p.status in ["done", "exited"])
        finished_w1 = sum(1 for p in env_w1.sim.patients if p.status in ["done", "exited"])
        fairness_gap = abs(total_episode_reward_w0 - total_episode_reward_w1)

        recent_rewards_w0.append(total_episode_reward_w0)
        recent_rewards_w1.append(total_episode_reward_w1)
        recent_finished_w0.append(finished_w0)
        recent_finished_w1.append(finished_w1)
        recent_gaps.append(fairness_gap)

        # Task-Specific Metrics
        def get_task_metrics(env):
            sim = env.sim
            metrics = {}
            metrics["Triage_Routing_Accuracy"] = np.mean(sim.routing_decisions) if sim.routing_decisions else 0.0
            metrics["Escort_Avg_Patient_Wait_Time"] = np.mean(sim.escort_wait_times) if sim.escort_wait_times else 0.0
            metrics["Manager_Avg_Ward_Wait_Time"] = np.mean(sim.ward_wait_times) if sim.ward_wait_times else 0.0
            metrics["Manager_Doctor_Moves"] = sim.doctor_moves_count
            return metrics

        task_metrics_w0 = get_task_metrics(env_w0)
        task_metrics_w1 = get_task_metrics(env_w1)

        log_data = {
            "Episode": episode,
            "Per-Episode/Total Reward W0 (Factual)": total_episode_reward_w0,
            "Per-Episode/Total Reward W1 (Counterfactual)": total_episode_reward_w1,
            "Per-Episode/Finished Patients W0": finished_w0,
            "Per-Episode/Finished Patients W1": finished_w1,
            "Per-Episode/Fairness Gap (Total Reward)": fairness_gap,
            
            "Rolling-Average/Avg Reward W0": np.mean(recent_rewards_w0),
            "Rolling-Average/Avg Reward W1": np.mean(recent_rewards_w1),
            "Rolling-Average/Avg Finished W0": np.mean(recent_finished_w0),
            "Rolling-Average/Avg Finished W1": np.mean(recent_finished_w1),
            "Rolling-Average/Avg Fairness Gap": np.mean(recent_gaps),
        }

        # Add task metrics for both worlds to the log
        for k, v in task_metrics_w0.items(): log_data[f"Task-Metrics-W0/{k}"] = v
        for k, v in task_metrics_w1.items(): log_data[f"Task-Metrics-W1/{k}"] = v

        # Agent Updates
        for name in agents_w0.keys():
            if len(memories_w0[name].states) > 0 and len(memories_w1[name].states) > 0:
                loss0 = agents_w0[name].update(memories_w0[name], memories_w1[name])
                if loss0 is not None: log_data[f"Loss/W0_{name}"] = loss0
                
                loss1 = agents_w1[name].update(memories_w1[name], memories_w0[name])
                if loss1 is not None: log_data[f"Loss/W1_{name}"] = loss1
        
        wandb.log(log_data)

        for mem in memories_w0.values(): mem.clear_memory()
        for mem in memories_w1.values(): mem.clear_memory()

        if episode > 0 and (episode % config["save_every"] == 0 or episode == config["num_episodes"] - 1):
            ckpt_dir = f"ppo_weights_cf/alpha={config['alpha']}_beta={config['beta']}"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({name: agent.get_state_dict() for name, agent in agents_w0.items()}, f"{ckpt_dir}/ckpt_w0_ep{episode}.pth")
            torch.save({name: agent.get_state_dict() for name, agent in agents_w1.items()}, f"{ckpt_dir}/ckpt_w1_ep{episode}.pth")
            print(f"\nSaved checkpoints to {ckpt_dir}")
            
    env_w0.close()
    env_w1.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Two-Policy CF PPO training for Hospital MAS")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha for reward fairness")
    parser.add_argument("--beta", type=float, required=True, help="Beta for state-value fairness")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--save_every", type=int, default=100, help="Frequency of saving checkpoints.")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to checkpoint file to resume.")
    parser.add_argument("--wandb_run_name", type=str, default="hospital-cf-run", help="Run name for wandb")
    parser.add_argument("--cuda", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--run_seed", type=int, default=42, help="Seed for the entire training run for reproducibility")
    args = parser.parse_args()
   
    config = {
        "alpha": args.alpha, "beta": args.beta, "num_episodes": args.num_episodes,
        "save_every": args.save_every, "device": torch.device(args.cuda if torch.cuda.is_available() else "cpu"),
        "run_seed": args.run_seed, "ppo_lr": 1e-5, "gamma": 0.99, "eps_clip": 0.2, "k_epochs": 10,
        "batch_size": 256, "value_loss_coef": 0.5, "entropy_coef": 0.01,
        "n_clerks": 30, "m_nurses": 60, "k_robots": 30, "n_triage": 30,
        "n_swing_doctors": 18, "n_patients": 300, "sim_hours": 12,
        "wandb_project": "[FINAL]Hospital-FairPPO-CF", 
        "wandb_run_name": args.wandb_run_name
    }
    
    random.seed(config["run_seed"])
    np.random.seed(config["run_seed"])
    torch.manual_seed(config["run_seed"])

    print("Generating arrival templates...")
    arrival_templates = generate_arrival_templates(n_templates=50, sim_minutes=config["sim_hours"] * 60, seed=config["run_seed"])

    print("Generating base patient templates for all episodes...")
    all_patient_schedules_w0 = []
    all_patient_schedules_w1 = []
    
    illnesses = ["emergency", "cardio", "general", "pediatric", "psychiatric", "xray"]
    priorities = ["low", "medium", "high"]
    impair_lvl = ["none", "low", "high"]
    
    for i in range(config["num_episodes"]):
        template = random.choice(arrival_templates)
        arrival_times = build_arrival_schedule(config["n_patients"], template["peaks"], sim_minutes=config["sim_hours"] * 60)
        
        episode_schedule_w0 = []
        episode_schedule_w1 = []

        for patient_idx, arrival_time in enumerate(arrival_times):
            base_priority = random.choice(priorities)
            base_illness = random.choice(illnesses)
            base_impairment = random.choices(impair_lvl, weights=[0.6, 0.25, 0.15])[0]

            patient_w0 = Patient(400 + patient_idx, priority=base_priority, position="outside", illness=base_illness, impairment_level=base_impairment, arrival_time=arrival_time)
            episode_schedule_w0.append(patient_w0)

            if base_impairment == "none":
                cf_impairment = "high"
            elif base_impairment == "high":
                cf_impairment = "none"
            else:
                cf_impairment = "low"
            
            patient_w1 = Patient(400 + patient_idx, priority=base_priority, position="outside", illness=base_illness, impairment_level=cf_impairment, arrival_time=arrival_time)
            episode_schedule_w1.append(patient_w1)

        all_patient_schedules_w0.append(episode_schedule_w0)
        all_patient_schedules_w1.append(episode_schedule_w1)

    print(f"Generated {config['num_episodes']} sets of factual/counterfactual patient schedules.")
    
    env_config_keys = ["n_clerks", "n_patients", "m_nurses", "k_robots", "n_triage", "n_swing_doctors", "sim_hours", "gamma"]
    env_config = {key: config[key] for key in env_config_keys}

    env_w0 = HospitalMAS(**env_config, verbose=False)
    env_w1 = HospitalMAS(**env_config, verbose=False)
    
    env_w0.reset(seed=config["run_seed"])
    env_w1.reset(seed=config["run_seed"] + 1) # Use a different seed for the second env
    
    agent_config_keys = ["ppo_lr", "gamma", "eps_clip", "k_epochs", "batch_size", "value_loss_coef", "entropy_coef", "alpha", "beta", "device"]
    agent_config = {key: config[key] for key in agent_config_keys}
    agent_config['lr'] = agent_config.pop('ppo_lr')

    agents_w0, agents_w1 = {}, {}
    agent_input_dims = {"escort_dispatcher": 6, "triage_router": 18, "doctor_manager": 19}
    for name, input_dim in agent_input_dims.items():
        agents_w0[name] = PPOAgentMDCF(input_dim=input_dim, action_space=env_w0.action_spaces[name], **agent_config)
        agents_w1[name] = PPOAgentMDCF(input_dim=input_dim, action_space=env_w1.action_spaces[name], **agent_config)

    start_episode = 0
    
    wandb.init(
        project=config["wandb_project"],
        name=f"{config['wandb_run_name']}-alpha_{config['alpha']}-beta_{config['beta']}-seed_{config['run_seed']}",
        config=config, resume="allow"
    )

    for agent in agents_w0.values(): wandb.watch(agent.policy, log="all", log_freq=100)
    for agent in agents_w1.values(): wandb.watch(agent.policy, log="all", log_freq=100)
    
    train_ppo_cf(agents_w0, agents_w1, config, arrival_templates, start_episode, env_w0, env_w1, all_patient_schedules_w0, all_patient_schedules_w1)    
    
    weights_dir = f"fairPPO_weights/cf/alpha={config['alpha']}_beta={config['beta']}"
    os.makedirs(weights_dir, exist_ok=True)
    for agent_name, agent_obj in agents_w0.items():
        agent_obj.save_weights(os.path.join(weights_dir, f'w0_{agent_name}_weights.pth'))
    for agent_name, agent_obj in agents_w1.items():
        agent_obj.save_weights(os.path.join(weights_dir, f'w1_{agent_name}_weights.pth'))
    
    wandb.finish()
    print('Training finished.')
