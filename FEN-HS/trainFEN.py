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
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
except (IndexError, NameError):
    project_root = Path.cwd()
    sys.path.append(str(project_root))

from environments.Hospital.hospital_pz_env import HospitalMAS
from agent_scripts.FENAgent import FENAgent
from environments.Hospital.utils import generate_arrival_templates

def sort_templates_by_difficulty(templates: list[dict]) -> list[dict]:
    """
    Sorts arrival templates based on a difficulty score.
    Difficulty is defined by the number and total duration of peaks.
    """
    def get_difficulty_score(template):
        total_peak_duration = sum(end - start for start, end in template["peaks"])
        return len(template["peaks"]) * total_peak_duration

    return sorted(templates, key=get_difficulty_score)

def train_fen(agents: dict, config: dict, arrival_templates: list, start_episode: int, env: HospitalMAS):
    """
    Trains FEN agents for the HospitalMAS environment with patient-group fairness.
    """
    
    run_rng = random.Random(config["run_seed"])

    # Curriculum Learning setup
    sorted_templates = sort_templates_by_difficulty(arrival_templates)
    n_total_templates = len(sorted_templates)
    phase1_end, phase2_end = 300, 600
    easy_pool = sorted_templates[:max(1, int(n_total_templates * 0.3))]
    medium_pool = sorted_templates[:max(1, int(n_total_templates * 0.7))]
    hard_pool = sorted_templates

    recent_rewards = deque(maxlen=100)
    recent_finished = deque(maxlen=100)
    recent_gaps = deque(maxlen=100)

    for episode in tqdm(range(start_episode, config["num_episodes"])):
        # Episode initialization
        if episode < phase1_end:
            env.arrival_template_override = random.choice(easy_pool)
        elif episode < phase2_end:
            env.arrival_template_override = random.choice(medium_pool)
        else:
            env.arrival_template_override = random.choice(hard_pool)

        episode_seed = run_rng.randint(0, 1_000_000_000)
        obs, _ = env.reset(seed=episode_seed)
        done = {"__all__": False}
        episode_total_reward = 0.0
        current_sub_policy = {name: 0 for name in agents}

        pending_controller_experiences = []

        # Main loop
        while not done["__all__"]:
            # Controller selects sub-policies for the next T steps
            sub_policy_selections = {}
            for agent_name, agent_obs in obs.items():
                if agent_obs.sum() > 0:  # Agent is active
                    state_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0).to(config["device"])

                    sp_idx, sp_log_prob, sp_state_value = agents[agent_name].select_sub_policy(state_tensor)
                    current_sub_policy[agent_name] = sp_idx
                    sub_policy_selections[agent_name] = sp_idx

                    # Store experience temporarily; reward will be added at episode end
                    pending_controller_experiences.append(
                        (agent_name, state_tensor.squeeze(0), sp_idx, sp_log_prob, sp_state_value)
                    )

            # Sub-policies interact with the environment for T steps
            for _ in range(config["T_steps"]):
                if done["__all__"]:
                    break

                step_actions = {}
                step_data_for_reward = {}

                # 1. Select actions using the chosen sub-policies
                for agent_name, agent_obs in obs.items():
                    if agent_obs.sum() > 0:
                        state_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0).to(config["device"])
                        sub_policy_idx = current_sub_policy[agent_name]

                        action, log_prob, state_value = agents[agent_name].select_action(sub_policy_idx, state_tensor)
                        step_actions[agent_name] = action[0]

                        # Store sub-policy experience data
                        step_data_for_reward[agent_name] = {
                            'sub_policy_idx': sub_policy_idx, 'log_prob': log_prob, 'state_value': state_value,
                            'state_tensor': state_tensor.squeeze(0), 'action': action[0]
                        }

                # 2. Step the environment
                obs, env_rewards, terminated, _, infos = env.step(step_actions)
                done = terminated
                episode_total_reward += sum(env_rewards.values())

                # 3. Store rewards and experience in sub-policy memory
                for agent_name, data in step_data_for_reward.items():
                    sub_policy_idx = data['sub_policy_idx']
                    mem = agents[agent_name].sub_policy_memories[sub_policy_idx]

                    mem.states.append(data['state_tensor'])
                    mem.actions.append(data['action'])
                    mem.logprobs.append(data['log_prob'])
                    mem.state_values.append(data['state_value'])

                    # Policy 0 (efficiency) gets direct environment reward
                    if sub_policy_idx == 0:
                        mem.rewards.append(env_rewards.get(agent_name, 0.0))
                    # Other policies (fairness/diversity) get the controller's log_prob
                    else:
                        # This value represents log p(z|o): each sub-policy is rewarded for being easily distinguishable 
                        # by the controller based on the states it visits
                        controller_exp = pending_controller_experiences[-1] # Get the most recent controller decision
                        
                        # Ensure the experience belongs to the correct agent before assigning reward
                        if controller_exp[0] == agent_name:
                            # controller_exp = (agent_name, state, action, log_prob, state_value)
                            # The log_prob of the controller's action is our information-theoretic reward
                            info_theoretic_reward = controller_exp[3].item()
                            mem.rewards.append(info_theoretic_reward)
                        else:
                            # Fallback in case of a mismatch, though this shouldn't happen
                            mem.rewards.append(0.0)

        # Calculate Fairness Reward and Update
        # 1. Calculate final utilities for patient groups
        patient_rewards = env.sim.drain_patient_rewards()
        finished_patients = [p for p in env.sim.patients if p.status in ("done", "exited")]

        impaired_rewards = [patient_rewards.get(p.id, 0.0) for p in finished_patients if p.true_impairment != "none"]
        non_imp_rewards = [patient_rewards.get(p.id, 0.0) for p in finished_patients if p.true_impairment == "none"]

        avg_impaired_utility = np.mean(impaired_rewards) if impaired_rewards else 0.0
        avg_non_impaired_utility = np.mean(non_imp_rewards) if non_imp_rewards else 0.0

        all_finished_rewards = impaired_rewards + non_imp_rewards
        avg_system_utility = np.mean(all_finished_rewards) if all_finished_rewards else 0.0

        # 2. Calculate the Fair-Efficient Reward
        c = 100.0  # Normalization constant for system utility
        epsilon = 0.1
        
        # Calculate fairness score for each group
        if avg_system_utility > 1e-6:
            fair_score_impaired = (avg_system_utility / c) / (epsilon + abs((avg_impaired_utility / avg_system_utility) - 1))
            fair_score_non_impaired = (avg_system_utility / c) / (epsilon + abs((avg_non_impaired_utility / avg_system_utility) - 1))
            final_fairness_reward = (fair_score_impaired + fair_score_non_impaired) / 2.0
        else:
            final_fairness_reward = 0.0

        # 3. Assign the final calculated reward to all of the controller's experiences for this episode
        for exp in pending_controller_experiences:
            agent_name, state, action, log_prob, state_value = exp
            agents[agent_name].controller_memory.states.append(state)
            agents[agent_name].controller_memory.actions.append(action)
            agents[agent_name].controller_memory.logprobs.append(log_prob)
            agents[agent_name].controller_memory.state_values.append(state_value)
            agents[agent_name].controller_memory.rewards.append(final_fairness_reward)

        # wandb
        finished_count = len(finished_patients)
        fairness_gap = abs(avg_non_impaired_utility - avg_impaired_utility)

        recent_rewards.append(episode_total_reward)
        recent_finished.append(finished_count)
        recent_gaps.append(fairness_gap)

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

        log_data = {
            "Episode": episode,
            "Per-Episode/Total Env Reward": episode_total_reward,
            "Per-Episode/Final Fairness Reward": final_fairness_reward,
            "Per-Episode/Finished Patients": finished_count,
            "Per-Episode/Fairness Gap": fairness_gap,

            "Rolling-Average/Avg Total Reward": np.mean(recent_rewards),
            "Rolling-Average/Avg Finished Patients": np.mean(recent_finished),
            "Rolling-Average/Avg Fairness Gap": np.mean(recent_gaps),

            "Reward Details/Impaired (Avg Utility)": avg_impaired_utility,
            "Reward Details/Non-Impaired (Avg Utility)": avg_non_impaired_utility,

            "Task-Metrics/Triage_Routing_Accuracy": routing_accuracy,
            "Task-Metrics/Escort_Avg_Patient_Wait_Time": avg_escort_wait_time,
            "Task-Metrics/Escort_Avg_Helper_Travel_Time": avg_helper_travel_time,
            "Task-Metrics/Manager_Avg_Ward_Wait_Time": avg_ward_wait_time,
            "Task-Metrics/Manager_Doctor_Moves": doctor_moves,
            
            "Routing-Breakdown/Perfect_Match_Pct": perfect_pct,
            "Routing-Breakdown/Acceptable_Match_Pct": acceptable_pct,
            "Routing-Breakdown/Wrong_Match_Pct": wrong_pct,
        }

        # Update agents and add their losses to the log
        for agent_name, agent_obj in agents.items():
            if len(agent_obj.controller_memory.states) > 0:
                loss = agent_obj.update('controller')
                if loss is not None: log_data[f"Loss/Controller_{agent_name}"] = loss

            for i in range(agent_obj.num_sub_policies):
                if len(agent_obj.sub_policy_memories[i].states) > 0:
                    loss = agent_obj.update('sub_policy', policy_index=i)
                    if loss is not None: log_data[f"Loss/SubPolicy_{i}_{agent_name}"] = loss

        wandb.log(log_data)

        # Save checkpoint
        if episode > 0 and (episode % config["save_every"] == 0 or episode == config["num_episodes"] - 1):
            ckpt_dir = args.ckpt_dir
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = f"{ckpt_dir}/ckpt_ep{episode}.pth"
            checkpoint = {name: agent.get_state_dict() for name, agent in agents.items()}
            torch.save(checkpoint, ckpt_path)
            print(f"\nSaved checkpoint to {ckpt_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run FEN training for Hospital MAS")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--save_every", type=int, default=100, help="Frequency of saving checkpoints.")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to checkpoint file to resume.")
    parser.add_argument("--wandb_run_name", type=str, default="hospital-fen-run", help="Run name for wandb")
    parser.add_argument("--cuda", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--run_seed", type=int, default=42, help="Seed for the entire training run.")
    parser.add_argument("--T_steps", type=int, default=50, help="Frequency of controller decisions (macro-step size).")
    parser.add_argument("--ckpt_dir", type=str, default="fen_weights", help="Directory to save checkpoints.")
    args = parser.parse_args()

    config = {
        "num_episodes": args.num_episodes, "save_every": args.save_every,
        "device": torch.device(args.cuda if torch.cuda.is_available() else "cpu"),
        "run_seed": args.run_seed, "ppo_lr": 1e-5, "gamma": 0.99, "eps_clip": 0.1, "k_epochs": 10,
        "batch_size": 64, "value_loss_coef": 0.5, "entropy_coef": 0.01,
        "n_clerks": 30, "m_nurses": 60, "k_robots": 30, "n_triage": 30,
        "n_swing_doctors": 18, "n_patients": 300, "sim_hours": 12,
        "wandb_project": "Hospital-FEN", "wandb_run_name": args.wandb_run_name,
        "num_sub_policies": 4,
        "T_steps": args.T_steps,
    }

    arrival_templates = generate_arrival_templates(n_templates=5, sim_minutes=config["sim_hours"] * 60)

    env_keys = ["n_clerks", "n_patients", "m_nurses", "k_robots", "n_triage", "n_swing_doctors", "sim_hours", "gamma"]
    env_config = {key: config[key] for key in env_keys if key in config}
    env_config['seed'] = None
    env_config['render_mode'] = False
    env_config['force_impairment'] = None
    env_config['verbose'] = False
    main_env = HospitalMAS(**env_config)

    agent_config_keys = ["ppo_lr", "gamma", "eps_clip", "k_epochs", "batch_size",
                         "value_loss_coef", "entropy_coef", "num_sub_policies", "device"]
    agent_config = {key: config[key] for key in agent_config_keys}
    agent_config['lr'] = agent_config.pop('ppo_lr')

    agents = {}
    agent_input_dims = {"escort_dispatcher": 6, "triage_router": 18, "doctor_manager": 19}
    for name, input_dim in agent_input_dims.items():
        agents[name] = FENAgent(
            input_dim=input_dim,
            action_space=main_env.action_spaces[name],
            **agent_config
        )

    start_episode = 0
    if args.resume_ckpt:
        checkpoint = torch.load(args.resume_ckpt, map_location=config["device"])
        for name, agent in agents.items():
            if name in checkpoint: agent.load_state_dict(checkpoint[name])
        start_episode = int(args.resume_ckpt.split('_ep')[1].split('.pth')[0]) + 1

    wandb.init(
        project=config["wandb_project"],
        name=f"{config['wandb_run_name']}",
        config=config, resume="allow"
    )

    for agent in agents.values():
        wandb.watch(agent.controller, log="all", log_freq=100)

    train_fen(agents, config, arrival_templates, start_episode, main_env)

    wandb.finish()
    print('Training finished.')