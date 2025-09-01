# Project Title: Fairness Aware Reinforcement Learning via Proximal Policy Optimization

## Description
Implementation of Fair-PPO, with simulation tests (Allelopathic Harvest (AH) and HospitalSim (HS)) and benchmarks (PPO, FEN, SOTO).

## Folders Structure
* **`environments`**: Full implementation of AllelopathicHarvest (AH) and HospitalSim (HS).
* **`fair-PPO-AH`**: Fair-PPO implementation for AH with different fairness metrics:
    * **Demographic Parity (DP)**
    * **Conditional Statistical Parity (CSP)**
    * **Counterfactual Fairness (CF)**

### BENCHMARK-AH
* **`FEN-AH`**: FEN implementation in AH (see the paper "Learning Fairness in Multi-Agent Systems", Jiang et al., 2019).
* **`SOTO-AH`**: SOTO implementation in the AH (see the paper "Learning Fair Policies in Decentralized Cooperative Multi-Agent Reinforcement Learning", Zimmer et al., 2021).
	
## How to Train the Algorithms
Examples with GPU. See parsing args to add more.

```python
# fair-PPO-AH/DP/train.py: 
	python3 train.py --alpha 0.5 --beta 0.5 --num_episodes 1000 --max_timesteps 3000 --wandb_run_name "fairppo_dp_run1" --cuda "cuda:0"
# fair-PPO-AH/CSP/train.py:
	python3 train.py --alpha 0.5 --beta 0.5 --num_episodes 1000 --max_timesteps 3000 --wandb_run_name "fairppo_run_1" --cuda "cuda:0"
# fair-PPO-AH/CF/train.py:
	python3 train.py --alpha 0.5 --beta 0.5 --num_episodes 1000 --max_timesteps 3000 --wandb_run_name "counterfactual_ppo_run_1" --cuda "cuda:0"
# FEN-AH/trainFEN.py:
	python3 trainFEN.py --num_episodes 1000 --max_timesteps 3000 --wandb_run_name "fen_run_1" --cuda "cuda:0"
# SOTO-AH/train.py:
	python3 trainSOTO.py --num_episodes 1000 --max_timesteps 2000 --wandb_run_name "soto_run_1" --cuda "cuda:0"
	
# fair-PPO-HS/DP/train.py: 
	python3 train.py --alpha 0.5 --beta 0.5 --num_episodes 1000 --save_every 200 --wandb_run_name "fairppo_dp_run1" --cuda "cuda:0"
# fair-PPO-HS/CSP/train.py:
	python3 train.py --alpha 0.5 --beta 0.5 --num_episodes 1000 --save_every 200 --wandb_run_name "fairppo_dp_run1" --cuda "cuda:0"
# fair-PPO-HS/CF/train.py:
	python3 train.py --alpha 0.5 --beta 0.5 --num_episodes 1000 --save_every 200 --wandb_run_name "counterfactual_ppo_run_1" --cuda "cuda:0"
# FEN-HS/trainFEN.py:
	python3 trainFEN.py --num_episodes 2000 --T_steps 50 --save_every 250 --wandb_run_name "fen_run_1" --cuda "cuda:0"
# SOTO-HS/train.py:
	python3 trainSOTO.py --alpha 0.5 --num_episodes 1000 --save_every 200 --wandb_run_name "soto_run_1" --cuda "cuda:0"
	
See requirements.txt file to see the necessary packages to install (pip install -r requirements.txt).

## Results
In folder "Analysis", we report the scripts to obtain the results in the paper. Please check the input/output folders and the result files format.
Figures and tables are in the respective folders "Figures-Tables" or "Tables". The training results are produced through wandb.
