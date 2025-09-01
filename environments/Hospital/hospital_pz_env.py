from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import numpy as np, random

from environments.Hospital.environment import Environment
from environments.Hospital.players import (AcceptanceAgents, Nurse, Robot,
                     Doctor, Patient, TriageAgent)
from environments.Hospital.utils import build_arrival_schedule

N_DOCTOR_PER_DEPT = 10 # number of fixed doctors per department

class HospitalMAS(ParallelEnv):
    """
    Three learning agents thorugh RL: escort_dispatcher, triage_router, doctor_manager
    """

    metadata = {"name": "hospital_mas_v0"}
    _AGENTS   = ["escort_dispatcher", "triage_router", "doctor_manager"]

    # static observation spaces (fixed vector length)
    observation_spaces = {
        "escort_dispatcher": spaces.Box(0.0, 1.0, (6,),  np.float32),
        "triage_router": spaces.Box(0.0, 1.0, (18,),  np.float32),
        "doctor_manager": spaces.Box(0.0, 1.0, (19,), np.float32),
    }

    def __init__(
        self, *,
        n_clerks=4, n_patients=300, m_nurses=10, k_robots=8, n_triage=4,
        sim_hours=8, seed=None, render_mode=False, verbose=True,
        n_swing_doctors=4, force_impairment=None, gamma=0.99
    ):
        super().__init__()

        self.cfg = dict(locals())
        self.cfg.pop("self")

        # run-time settings
        self.rng = random.Random(seed)
        self.max_minutes = sim_hours * 60
        self.render_mode = render_mode
        self.verbose = verbose
        self.n_swing = n_swing_doctors
        self.force_impairment = force_impairment

        # dynamic action spaces: multiple actions at the same time 
        self.action_spaces = {
            "escort_dispatcher": spaces.MultiDiscrete([50]*20),
            "triage_router": spaces.Discrete(6),
            "doctor_manager": spaces.MultiDiscrete([7]*self.n_swing),
        }

        # internal state
        self.sim = None # Environment() reference
        self._cached_obs = None
        self.agents = [] # active agents this episode
        
    def set_patient_schedule(self, patient_list: list):
        """
        Allows the training script to inject a pre-defined list of patients
        This method is used ONLY in counterfactual penalty calculation of Fair-PPO
        """
        
        self.sim.arrival_schedule = []
        self.sim.patients = []
        
        for p in patient_list:
            self.sim.schedule_patient(p)

    # PettingZoo
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)

        # 1 fresh simulator 
        self.sim = Environment(
            rl_mode  = True,
            visualise= self.render_mode,
            verbose  = self.verbose,
            gamma    = self.cfg["gamma"]
        )
        self.sim.simulation_limit = self.max_minutes
        C = self.cfg

        # 2 staff 
        for i in range(C["n_clerks"]):
            self.sim.acceptance_staff.append(AcceptanceAgents(600+i)) # create acceptance agents (entrance) based on n clerks
        
        fixed = [
            ("emergency", "Resuscitation"),
            ("cardio", "AcuteCare"),
            ("general", "PromptCare"),
            ("pediatric", "Pediatrics"),
            ("psychiatric", "Psychiatry"),
            ("xray", "Imaging"),
        ]
        n_per_dept = N_DOCTOR_PER_DEPT # number of fixed doctors per department
        doctor_id = 100
        for spec, room in fixed:
            for _ in range(n_per_dept):
                self.sim.assign_doctor(Doctor(doctor_id, spec, room)) # assign doctors to rooms
                doctor_id += 1

        for i in range(C["m_nurses"]):
            self.sim.assign_nurse(Nurse(200+i))
        for i in range(C["k_robots"]):
            self.sim.assign_robot(Robot(300+i))
        for i in range(C["n_triage"]):
            self.sim.triage_agents.append(TriageAgent(700+i))

        # swing doctors (managed by doctor_manager)
        self.sim.swing_doctors = []
        for i in range(self.n_swing):
            d = Doctor(150+i, "swing", "StaffArea", swing=True)
            self.sim.assign_doctor(d)
            self.sim.swing_doctors.append(d)

        # 3 patients schedule
        if not self.sim.patients: # this condition checks there are no external injections of patients (counterfactual case)
            illnesses = ["emergency","cardio","general",
                        "pediatric","psychiatric","xray"]
            priorities = ["low","medium","high"]
            impair_lvl = ["none","low","high"]

            if hasattr(self, "arrival_template_override") and self.arrival_template_override:
                peaks = self.arrival_template_override["peaks"]
            else:
                # If not, use the original random generation logic as a fallback
                peaks = [(self.rng.randint(0, 420),
                        self.rng.randint(60, 120) + self.rng.randint(0, 420))]
                
            arr_ts = build_arrival_schedule(C["n_patients"], peaks, sim_minutes=self.max_minutes, λ_peak=self.rng.uniform(10,18), λ_off =self.rng.uniform(1,5))  
            self.sim.arrival_times = arr_ts # store arrival times for plot

            for i, at in enumerate(arr_ts):
                if self.force_impairment == "none":
                    impairment_for_this_patient = "none"
                elif self.force_impairment == "impaired":
                    # If forcing impairment, high impairment
                    impairment_for_this_patient = "high"
                else:
                    # Default behavior: random choice based on weights
                    impairment_for_this_patient = self.rng.choices(impair_lvl, weights=[0.6,0.25,0.15])[0] # chances of assigning none, low, high impairment level

                self.sim.schedule_patient(Patient(400+i, priority = self.rng.choice(priorities), 
                                                position ="outside", illness = self.rng.choice(illnesses), 
                                                impairment_level = impairment_for_this_patient, 
                                                arrival_time = at))

        # 4 run until the first “hook”
        self._advance_until_hook()
        self.agents = self._AGENTS.copy()
        return self._cached_obs, {a: {} for a in self.agents}

    def step(self, action_dict):
        # If the simulator is paused for a decision but the caller
        # forgot to send one, make a random move so time goes on.
        if self.sim._hook == "awaiting_routing" and "triage_router" not in action_dict:
             action_dict["triage_router"] = self.action_spaces["triage_router"].sample()

        if self.sim._hook == "awaiting_dispatch" and "escort_dispatcher" not in action_dict:
            action_dict["escort_dispatcher"] = self.action_spaces["escort_dispatcher"].sample()

        if self.sim._hook == "awaiting_doctor_move" and "doctor_manager" not in action_dict:
            action_dict["doctor_manager"] = self.action_spaces["doctor_manager"].sample()
        
        # Initialize rewards dictionary that will be populated by direct-return actions
        rewards = {a: 0.0 for a in self._AGENTS}
        
        # 1. apply actions
        if "escort_dispatcher" in action_dict:
            rewards["escort_dispatcher"] = self.sim.apply_dispatch_action(action_dict["escort_dispatcher"])

        if "triage_router" in action_dict:
            rewards["triage_router"] = self.sim.apply_routing_action(action_dict["triage_router"])

        if "doctor_manager" in action_dict:
            rewards["doctor_manager"] = self.sim.apply_manager_action(action_dict["doctor_manager"])

        self._advance_until_hook()

        # 1. Get the general, bundled reward (can be used for logging)
        general_reward_val = self.sim.pop_reward()

        # 2. Get the specific, attributable reward events
        reward_events = self.sim.drain_reward_events()

        # 3. Set rewards and info
        done = self.sim.current_time >= self.max_minutes
        
        terminations = {a: done for a in self.agents}; terminations["__all__"]=done
        truncations = {a: False for a in self.agents}
        
        # Pass the events out in the info dict
        infos = {a: {"reward_events": reward_events} for a in self.agents}

        if done:
            self.agents = []

        return self._cached_obs, rewards, terminations, truncations, infos

    def render(self): pass
    def close (self):
        if self.sim and self.render_mode:
            self.sim.finalise_episode(save_logs=True, make_gif=True)

    # advance core engine until it pauses for an agent decision
    def _advance_until_hook(self):
        while True:
            hook = self.sim.run_one_tick()
            done_by_clock = self.sim.current_time >= self.max_minutes # stop the sim due to time exceeding
            done_by_activity = (not self.sim.arrival_schedule and not self.sim.event_queue and 
                                all(p.status == "exited" for p in self.sim.patients)) # stop the sim due to no activity
            
            if hook or done_by_clock or done_by_activity:
                break

        # convert hook into observation vectors
        obs_ed = (np.array(self.sim.build_dispatch_obs(),np.float32)
                    if hook=="awaiting_dispatch" else np.zeros(6,np.float32))
        obs_tr = (np.array(self.sim.build_router_obs(self.sim._payload["patient"]), np.float32)
                    if hook=="awaiting_routing" else np.zeros(18, np.float32))
        obs_dm = (np.array(self.sim.build_manager_obs(),np.float32)
                    if hook=="awaiting_doctor_move" else np.zeros(19,np.float32))
        self._cached_obs = {
            "escort_dispatcher": obs_ed,
            "triage_router": obs_tr,
            "doctor_manager": obs_dm,
        }
