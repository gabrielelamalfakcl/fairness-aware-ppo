import heapq, random
import numpy as np
from collections import defaultdict

from environments.Hospital.players import *
from environments.Hospital.visual import *
from environments.Hospital.players import SYMPTOM_LIST


# Assign priority of assistance
PRIO_ENTRANCE = 3
PRIO_TRIAGE = 2
PRIO_DEPT = 1
PRIO_TO_IDX = {"low":0, "medium":1, "high":2}

# Setting the constants
PROCESSING_TIME_BASE = 5 # Base time to process patient at acceptance
PROCESSING_TIME_ASSIST_BONUS = 2 # Additional time if patient needs assistance
TRIAGE_DURATION = 10 # Minutes needed for triage
TREATMENT_DURATION = 30 # Minutes for treatment
FLOOR_TRAVEL_PENALTY = 5 # Penalty per floor in minutes

# We provide small rewards to avoid delayed rewards
PATIENT_STAGE_POTENTIAL = {
    "waiting": 0,
    "being_accepted": 1,
    "moving_to_triage": 2, 
    "waiting_for_escort": 2,
    "in_triage_queue": 3,
    "being_triaged": 4,
    "triaged": 5,
    "moving_to_ward": 5.5, 
    "being_treated": 6,
    "done": 7,
    "waiting_for_exit": 8,
    "moving_to_exit": 8.5,
    "exited": 9
}

# Defines the quality of sending a patient with a given illness to a specific ward.
# The value is a reward multiplier. Wards not listed are considered incorrect.
ILLNESS_WARD_MAP = {
    "emergency":   {"Resuscitation": 1.0},
    "cardio":      {"AcuteCare": 1.0, "PromptCare": 0.4, "Resuscitation": 0.7},
    "xray":        {"Imaging": 1.0, "PromptCare": 0.5},
    "psychiatric": {"Psychiatry": 1.0, "PromptCare": 0.2},
    "pediatric":   {"Pediatrics": 1.0, "PromptCare": 0.6},
    "general":     {"PromptCare": 1.0, "AcuteCare": 0.5}
}

class Environment:
    def __init__(self, visualise=False, verbose=False, priority_error_prob=0.0, rl_mode=False, gamma=0.99):
        self.verbose = verbose
        self.gamma = gamma

        self.rl_mode = rl_mode
        self._hook = None
        self._payload = None

        self._reward_buffer = 0.0

        self.rooms = {
            "Entrance": [],
            "Triage": [],
            "Resuscitation": [],
            "AcuteCare": [],
            "PromptCare": [],
            "Pediatrics": [],
            "Psychiatry": [],
            "Imaging": [],
            "StaffArea": [],
            "Exit": []
        }

        self.visualise = visualise
        self.snapshots = []

        self._event_counter = 0 # event counter

        # Entrance/Acceptance
        self.priority_error_prob = priority_error_prob # just for a deterministic model

        self.entrance_queue = []
        self.arrival_schedule = []
        self.acceptance_staff = []

        self.patients = []
        self.doctors = []
        self.swing_doctors = []
        self.nurses = []
        self.triage_agents = []
        self.robots = []
        self.available_nurses = []
        self.assistance_requests = []
        self._request_counter = 0
        self.available_robots = []
        self.event_queue = []
        self.current_time = 0
        self.completed_patients = []
        self.triage_queue = []

        self.print_every = 1

        # time to travel from one room to another
        self.room_travel_times = {
            ("Entrance", "Triage"): 2,
            ("Triage", "Resuscitation"): 3,
            ("Triage", "AcuteCare"): 4,
            ("Triage", "PromptCare"): 2.5,
            ("Triage", "Pediatrics"): 4.5,
            ("Triage", "Psychiatry"): 5,
            ("Triage", "Imaging"): 6,
            ("Imaging", "AcuteCare"): 3,
            ("PromptCare", "Exit"): 2,
            ("AcuteCare", "Exit"): 3.5,
            ("Psychiatry", "Exit"): 3,
            ("Pediatrics", "Exit"): 3,
            ("StaffArea", "Triage"): 2,
            ("Triage", "StaffArea"): 2,
            # Added reverse paths for patient self-movement
            ("Resuscitation", "Triage"): 3,
            ("AcuteCare", "Triage"): 4,
            ("PromptCare", "Triage"): 2.5,
            ("Pediatrics", "Triage"): 4.5,
            ("Psychiatry", "Triage"): 5,
            ("Imaging", "Triage"): 6
        }


        # room positions inside hospital
        self.room_positions = {
            "Entrance": (0, 0),
            "Triage": (2, 0),
            "Exit": (8, 0),

            "AcuteCare": (2, 4),
            "PromptCare": (4, 4),
            "Imaging": (6, 4),

            "Pediatrics": (2, 8),
            "Psychiatry": (4, 8),
            "Resuscitation": (6, 8),
            "StaffArea": (0, 8)
        }

        self.room_floors = {
            "Entrance": 0,
            "Triage": 0,
            "Resuscitation": 0,
            "AcuteCare": 1,
            "PromptCare": 1,
            "Pediatrics": 2,
            "Psychiatry": 2,
            "Imaging": 3,
            "StaffArea": 0,
            "Exit": 0
        }

        self.simulation_limit = 720 # 12h #480 8h

        self._patient_rewards = defaultdict(float)
        self._reward_events = []
        self.manager_shaping_penalty = 0.0
        
        # additional data for metrics evaluation
        self.routing_decisions = []
        self.routing_outcomes = []
        self.escort_wait_times = []
        self.helper_travel_times = []
        self.doctor_moves_count = 0
        self.ward_wait_times = []

    def _apply_shaped_reward(self, patient, old_status):
        """
        Calculates and applies a potential-based reward for a status change
        """

        if old_status not in PATIENT_STAGE_POTENTIAL or patient.status not in PATIENT_STAGE_POTENTIAL:
            return

        old_potential = PATIENT_STAGE_POTENTIAL[old_status]
        new_potential = PATIENT_STAGE_POTENTIAL[patient.status]

        # Only give a reward if the patient has progressed to a more valuable state
        # This is done to avoid a delayed reward signal
        if new_potential > old_potential:
            shaped_reward = (self.gamma * new_potential) - old_potential
            self.credit_patient(patient, shaped_reward)

    def _move_agent_to_room(self, agent_to_move, new_room_name):
        """
        Safely moves an agent to a new room, ensuring it's removed from all other room lists first.
        """
        
        if agent_to_move is None: return
        # Defensive removal from all other room lists
        for room_list in self.rooms.values():
            if agent_to_move in room_list:
                room_list.remove(agent_to_move)

        # Place agent in the new room
        if new_room_name and new_room_name in self.rooms:
            self.rooms[new_room_name].append(agent_to_move)
            agent_to_move.position = new_room_name
        elif new_room_name is None:
            # agent is in transit, not in any specific room
            agent_to_move.position = "in_transit"

    def credit_patient_dispatch(self, patient, reward_value):
        """
        Dispatch rewards. Credits the patient and creates a reward event for the dispatcher.
        """
        
        self.credit_patient(patient, reward_value)   
        self._reward_events.append({
            "agent_id": "escort_dispatcher",
            "patient_id": patient.id,
            "reward": reward_value
        })
    
    
    def credit_patient_routing(self, patient, reward_value):
        """
        Routing rewards. It credits the patient and creates a reward event to be passed
        """

        # The original credit_patient still works for general rewards
        self.credit_patient(patient, reward_value)

        # Create an event to be processed by the training loop
        self._reward_events.append({
            "agent_id": "triage_router",
            "patient_id": patient.id,
            "reward": reward_value
        })

    def credit_patient_completion(self, patient, reward_value):
        """
        Creates a reward event for the doctor_manager upon treatment completion.
        """
        self.credit_patient(patient, reward_value)

        self._reward_events.append({
            "agent_id": "doctor_manager",
            "patient_id": patient.id,
            "reward": reward_value
        })
        
    def credit_manager_action(self, reward_value):
        """
        Credits the doctor_manager agent for a move action.
        """
        self._reward_buffer += reward_value
        
        self._reward_events.append({
            "agent_id": "doctor_manager",
            "patient_id": None, # Action is not on a single patient
            "reward": reward_value
        })

    def drain_reward_events(self):
        """
        Pops all buffered reward events
        """
        events, self._reward_events = self._reward_events, []
        return events

    def credit_patient(self, patient, delta):
        self._patient_rewards[patient.id] += delta
        self._reward_buffer += delta

    def drain_patient_rewards(self):
        d, self._patient_rewards = self._patient_rewards, defaultdict(float)
        return d

    def build_dispatch_obs(self):
        """
        Escort dispatcher observation input
        """
        
        high_prio_reqs = 0
        med_prio_reqs = 0
        low_prio_reqs = 0
        
        # 1. Priority requests (3 features)
        for req in self.assistance_requests:
            patient_priority = req[2].true_priority
            if patient_priority == 'high':
                high_prio_reqs += 1
            elif patient_priority == 'medium':
                med_prio_reqs += 1
            else:
                low_prio_reqs += 1
        
        # Normalize the counts
        norm_high = high_prio_reqs / 10.0
        norm_med = med_prio_reqs / 10.0
        norm_low = low_prio_reqs / 10.0
        
        # 2. Max waiting time (1 feature)            
        times = [self.current_time - req[0] for req in self.assistance_requests]
        max_wait = max(times, default=0)/60
        
        # 3. Idle nurse/robots proportion (1 feature)            
        idle_ratio_nurse  = (sum(h.role=="idle" for h in self.nurses) / max(len(self.nurses), 1))
        idle_ratio_robot  = (sum(r.role=="idle" for r in self.robots) / max(len(self.robots), 1))

        # final observation vector
        final_obs = [norm_high, norm_med, norm_low, max_wait, # max waiting time in hours
            idle_ratio_nurse, idle_ratio_robot,
        ]
        return final_obs

    def build_router_obs(self, patient_to_route):
        """
        Triage router observation input
        """
        
        wards = ["PromptCare", "AcuteCare", "Psychiatry", "Pediatrics", "Resuscitation", "Imaging"]

        # 1. Patient Symptoms (12 features)
        patient_symptoms_vector = [patient_to_route.symptoms[s] for s in SYMPTOM_LIST]

        # 2. Expected Wait Time per Ward (6 features)
        expected_wait_times = []
        for w in wards:
            # Patients in the queue for this ward
            q = [p for p in self.rooms[w] if isinstance(p, Patient) and p.status == "triaged"]
            # Idle doctors in this ward
            idle_docs = sum(1 for d in self.rooms[w] if isinstance(d, Doctor) and d.busy_until <= self.current_time)
            
            # Calculate expected wait, adding a small epsilon to avoid division by zero
            # This value will be high if the queue is long or doctors are busy
            expected_wait = len(q) / (idle_docs + 0.1)
            expected_wait_times.append(expected_wait / 60.0) # Normalize by an hour

        # final observation vector
        final_obs = patient_symptoms_vector + expected_wait_times
        return final_obs

    def build_manager_obs(self):
        """
        Doctor manager observation input
        """
        
        wards = ["PromptCare","AcuteCare","Psychiatry",
                 "Pediatrics","Resuscitation","Imaging"]

        obs = []
        for w in wards:
            # 1. Queue Lenght (6 features)
            q = [p for p in self.rooms[w] if isinstance(p, Patient) and p.status != "done"]
            # 2. Doctors Available (6 features)
            docs = [d for d in self.rooms[w] if isinstance(d, Doctor)]
            idle = sum(d.busy_until <= self.current_time for d in docs)
            # 3. Mean Patient Priority (6 features)
            urg = np.mean([{"low":0,"medium":0.5,"high":1}[p.true_priority] for p in q]) if q else 0
            obs.extend([len(q)/20, idle/max(len(docs),1), urg])

        # 4. Free Swing Doctors (1 feature)
        free_swing = sum(d.busy_until <= self.current_time for d in self.swing_doctors)
        obs.append(free_swing/len(self.swing_doctors or [1]))
        return obs

    def _pop_heap_index(self, h, idx):
        """
        Remove element at *index* idx from a heap list h and re-heapify.
        """
        
        h[idx] = h[-1] # move last to hole
        h.pop()
        if idx < len(h):
            heapq._siftup(h, idx)
            heapq._siftdown(h, 0, idx)

    def apply_dispatch_action(self, assignment):
        """
        Apply the assignment decision from the escort_dispatcher agent.
        The assignment is a list mapping each idle helper to a request index.
        """
        
        # print(f"DEBUG: apply_dispatch_action called at time {self.current_time}")
        idle_helpers = self._payload["idle_helpers"]

        # Use a set to track requests that have been fulfilled in this tick
        fulfilled_request_indices = set()

        total_dispatch_reward = 0.0
        if self.assistance_requests and all(a >= len(self.assistance_requests) for a in assignment):
            # Penalize based on the number of waiting requests.
            total_dispatch_reward -= len(self.assistance_requests) * 2.0
        
        for helper_idx, request_idx in enumerate(assignment):
            if helper_idx >= len(idle_helpers):
                break

            # Action > num_requests means the agent wants this helper to stay idle
            if request_idx >= len(self.assistance_requests):
                continue

            # Check if another helper has already been assigned this request
            if request_idx in fulfilled_request_indices:
                continue

            helper = idle_helpers[helper_idx]
            _, _, patient, room = self.assistance_requests[request_idx]
            
            # 1. Proximity-based reward: Reward for assigning the closest helper
            # This directly measures dispatching efficiency
            travel_time = self.get_travel_time(helper.position, room, helper.speed)
            self.helper_travel_times.append(travel_time)
            # print(f"DEBUG: Appended to helper_travel_times. New length: {len(self.helper_travel_times)}")
            # The reward is higher for shorter travel times (max 10)
            proximity_reward = max(0, 10 - travel_time)
            total_dispatch_reward += proximity_reward
            
            # 2. Priority-based reward: Reward for serving high priority requests
            # This provides a positive incentive for correct prioritization
            if patient.true_priority == 'high':
                total_dispatch_reward += 15
            elif patient.true_priority == 'medium':
                total_dispatch_reward += 5

            self._send_helper(helper, room, patient)
            fulfilled_request_indices.add(request_idx)

        # Remove fulfilled requests from the queue (in reverse order to not mess up indices)
        for request_idx in sorted(list(fulfilled_request_indices), reverse=True):
            self.assistance_requests.pop(request_idx)

        self._hook = self._payload = None
        
        return total_dispatch_reward

    def apply_routing_action(self, action):
        """
        Apply the routing decision from the escort_dispatcher agent.
        The assignment is a list mapping each idle helper to a request index.
        """
        
        if self._hook != "awaiting_routing" or self._payload is None:
            print(f"[WARNING] Ignoring routing action: no routing hook")
            return 0.0

        patient = self._payload["patient"]

        patient.target_room = self.router_choices[action]
        patient.pred_illness = patient.target_room

        if self.verbose:
            print(f"[ROUTING] Patient {patient.id} assigned to {patient.target_room}")

        reward = self._finish_triage(patient)

        self._hook = self._payload = None
        
        return reward

    def apply_manager_action(self, action):
        """
        Apply the manager decision from the doctor_manager agent.
        """
        
        wards = ["PromptCare","AcuteCare","Psychiatry",
                 "Pediatrics","Resuscitation","Imaging"]
        
        # Define urgency weights for reward calculation
        urgency_weights = {"low": 0.5, "medium": 1.0, "high": 2.0}
        
        total_manager_reward = self.manager_shaping_penalty
        self.manager_shaping_penalty = 0.0

        for doc, dest in zip(self.swing_doctors, action):
            if dest == 6: # no-op
                continue
            target_room = wards[dest]
            if doc.position == target_room:
                continue
            
            # Calculate the immediate reward for making this decision.
            # The reward is based on the number and priority of patients in the target queue.
            q = [p for p in self.rooms[target_room] if isinstance(p, Patient) and p.status == "triaged"]
            if q:
                queue_pressure = 0
                for p in q:
                    # Patient.arrival_to_room_time is set when they first enter the ward queue.
                    wait_time = self.current_time - (p.arrival_to_room_time or self.current_time)
                    urgency = urgency_weights.get(p.true_priority, 0.5)
                    queue_pressure += wait_time * urgency
                
                # Apply a scaled positive reward for sending a doctor to a high-pressure ward.
                # The division by 10 is a scaling factor to keep reward values reasonable.
                total_manager_reward += (queue_pressure / 5.0)

            # Apply a small penalty for the cost of moving a doctor.
            total_manager_reward -= 1.0

            # remove from current room list
            self._move_agent_to_room(doc, None)

            # travel event
            travel = 10 # fixed transfer time: a simplification (this can be modelled with distance/speed)
            doc.busy_until = self.current_time + travel
            doc.position   = f"moving→{target_room}"
            self.push_event(doc.busy_until, "doctor_arrived", doc, order_key=0)
            doc.target_room = target_room
            
            if dest != 6 and doc.position != target_room:
                self.doctor_moves_count += 1
            
        return total_manager_reward

    def _instant_penalty(self):
        return 0.0

    def run_one_tick(self):
        self._reward_buffer = 0.0
        self._hook = None

        # 1. Process events for the current tick
        processed = 0
        while (self.event_queue and
                self.event_queue[0][0] <= self.current_time and
                processed < 1000):

            top = heapq.heappop(self.event_queue)
            ev_time, _, _, ev_type, subject = top

            if self.verbose:
                if not hasattr(self, "_dbg_cnt") or ev_time != self.current_time:
                    self._dbg_cnt = 0
                if self._dbg_cnt < 10:
                    print(f"[t={self.current_time}]  {ev_type:17}  subj:",
                        getattr(subject, "id", subject))
                    self._dbg_cnt += 1

            # Call the appropriate handler for the event
            if ev_type == "acceptance_done":
                self.handle_acceptance_done(subject)
            elif ev_type == "triage_done":
                self.handle_triage_done(subject)
            elif ev_type == "treatment_complete":
                self.handle_treatment_complete(subject)
            elif ev_type == "helper_arrived":
                self.handle_helper_arrived(subject)
            elif ev_type == "doctor_arrived":
                self.handle_doctor_arrived(subject)
            elif ev_type == "helper_return_to_base":
                self.handle_helper_return_to_base(subject)
            elif ev_type == "ready_to_exit":
                self.handle_ready_to_exit(subject)
            # Add handlers for new patient arrival events
            elif ev_type == "patient_arrived_at_triage":
                self.handle_patient_arrived_at_triage(subject)
            elif ev_type == "patient_arrived_at_ward":
                self.handle_patient_arrived_at_ward(subject)
            elif ev_type == "patient_arrived_at_exit":
                self.handle_patient_arrived_at_exit(subject)


            if self._hook:
                break

            processed += 1

        if processed == 1000:
            print(f"[WARN] 1,000 events processed at t={self.current_time}, possible tight loop.")

        if self._hook:
            return self._hook

        # 2. Main logic step for the current tick
        self.intake_step()
        if self._hook: return self._hook

        self.triage_step()
        if self._hook: return self._hook

        self.treatment_step()

        self.dispatch_helpers()
        if self._hook: return self._hook

        if (self.rl_mode and
            any(d.busy_until <= self.current_time for d in self.swing_doctors) and
            self.current_time % 5 == 0):
            self._hook = "awaiting_doctor_move"
            self._payload = None

        # Apply a small penalty for any swing doctor that is idle in a ward with no waiting patients
        # This is to encourage allocation with a logic
        treatment_wards = ["PromptCare", "AcuteCare", "Psychiatry", "Pediatrics", "Resuscitation", "Imaging"]
        for doc in self.swing_doctors:
            is_idle = doc.busy_until <= self.current_time
            in_idleable_ward = doc.position in treatment_wards
            
            if is_idle and in_idleable_ward:
                # Check if there are any patients waiting for treatment in the doctor's current ward.
                is_queue_empty = not any(p for p in self.rooms[doc.position] if isinstance(p, Patient) and p.status == 'triaged')
                if is_queue_empty:
                    # Apply a small, continuous penalty for being idle in an empty ward.
                    self.manager_shaping_penalty -= 0.5

        # Define penalty weights based on patient priority
        priority_penalty_weights = {"high": 0.3, "medium": 0.15, "low": 0.05}

        for patient in self.patients:
            # Penalize waiting for a doctor in a treatment ward
            if patient.status == 'triaged':
                # patient.arrival_to_room_time is set when they arrive at the ward
                wait_time = self.current_time - (patient.arrival_to_room_time or self.current_time)
                if wait_time > 15: # Apply penalty after a 15-minute grace period
                    base_penalty = -0.1 # Base penalty per minute
                    scaled_penalty = base_penalty * priority_penalty_weights.get(patient.true_priority, 0.05)
                    self.credit_patient(patient, scaled_penalty)
                    
            # Penalize waiting in the triage queue
            elif patient.status == 'in_triage_queue':
                wait_time = self.current_time - getattr(patient, 'arrival_to_triage_queue_time', self.current_time)
                if wait_time > 10: # Apply penalty after a 10-minute grace period
                    base_penalty = -0.1 # Base penalty per minute
                    scaled_penalty = base_penalty * priority_penalty_weights.get(patient.true_priority, 0.05)
                    self.credit_patient(patient, scaled_penalty)

        # 3. Finalise the tick
        self._reward_buffer -= self._instant_penalty()

        # if self.verbose:
        #     if self.current_time % self.print_every == 0:
        #         debug_tick(self)

        if self.visualise and self.current_time % 2 == 0:
            self.record_snapshot()

        self.current_time += 1

        for p in self.patients:
            if p.status == "waiting_for_escort" and p.true_impairment == "high":
                wait_time = self.current_time - (p.arrival_to_room_time or self.current_time)
                if wait_time > 5:
                    self.credit_patient(p, -2 * (wait_time - 5))

        return self._hook

    def push_event(self, ev_time, ev_type, subject, order_key=0):
        heapq.heappush(
            self.event_queue,
            (ev_time, order_key, self._event_counter, ev_type, subject)
        )
        self._event_counter += 1

    def _initiate_patient_move(self, patient, start_room, end_room, arrival_event_type):
        """
        A generic function to handle the start of a patient's self-ambulating
        It calculates travel time, sets the patient's state, and schedules an arrival event
        """
        
        travel_time = self.get_travel_time(start_room, end_room, patient.speed)
        arrival_time = self.current_time + travel_time

        # Remove patient from their current room and set them in transit
        self._move_agent_to_room(patient, None)
        patient.busy_until = arrival_time
        patient.target_room = end_room

        # Schedule the corresponding arrival event
        self.push_event(arrival_time, arrival_event_type, patient)

        # For visualization purposes
        self.simulate_movement(patient, start_room, end_room)
        self.log_patient_event(patient, f"Started moving from {start_room} to {end_room}. Will arrive at t={arrival_time:.1f}.")

    def _send_helper(self, helper, room, patient=None):
        """
        Move a nurse or robot to a destination to help a patient
        """
        
        orig_room = helper.position
        self._move_agent_to_room(helper, None)
        helper.role = "moving"
        helper.target_room = room
        helper.assigned_patient = patient

        travel = self.get_travel_time(orig_room, room, helper.speed)
        arrive_time = self.current_time + travel
        helper.busy_until = arrive_time

        self.push_event(arrive_time, "helper_arrived", helper)
        self.simulate_movement(helper, orig_room, room)

        if self.verbose:
            print(f"{helper.name} {helper.id} moving {orig_room} → {room}")

    def assign_nurse(self, nurse):
        """
        Nurse in StaffArea and mark it idle
        """
        
        self.nurses.append(nurse)
        nurse.role = "idle"
        self._move_agent_to_room(nurse, "StaffArea")
        self.available_nurses.append(nurse)

    def assign_robot(self, robot):
        """
        Robot in StaffArea and mark it idle
        """
        
        self.robots.append(robot)
        robot.role = "idle"
        self._move_agent_to_room(robot, "StaffArea")
        self.available_robots.append(robot)

    def assign_doctor(self, doctors):
        """
        Place each Doctor into the correct ward according to specialty.
        """
        
        if isinstance(doctors, Doctor):
            doctors = [doctors]

        specialty_to_room = {
            "general": "PromptCare",
            "cardio": "AcuteCare",
            "neuro": "Psychiatry",
            "pediatric": "Pediatrics",
            "emergency": "Resuscitation",
            "psychiatric":"Psychiatry",
            "xray": "Imaging",
        }

        for doc in doctors:
            self.doctors.append(doc)
            room = specialty_to_room.get(doc.specialty, "PromptCare")
            self._move_agent_to_room(doc, room)

    def get_travel_time(self, start, end, agent_speed=75):
        """
        Get travel time based on speed and distance from one room to another
        """
        
        base_time = self.room_travel_times.get((start, end), 5)
        floor_diff = abs(self.room_floors.get(start, 0) - self.room_floors.get(end, 0))
        floor_penalty = floor_diff * 2.0  # Example: 2 extra minutes per floor

        total_minutes = base_time + floor_penalty
        time_scaled = total_minutes * (75 / agent_speed)  # assuming base speed is 75
        return max(0.1, time_scaled)

    def get_position(self, room):
        return self.room_positions.get(room, (0, 0))

    def simulate_movement(self, agent, start_room, end_room):
        start = self.get_position(start_room)
        end = self.get_position(end_room)
        agent_speed = agent.speed
        travel_time = self.get_travel_time(start_room, end_room, agent_speed)
        steps = max(1, round(travel_time * 1))

        # Visualisation only
        for i in range(steps):
            t = i / steps
            interp_x = start[0] + (end[0] - start[0]) * t
            interp_y = start[1] + (end[1] - start[1]) * t
            agent.position = (interp_x, interp_y)

        # The agent's final logical position is set by the arrival event handler.
        agent.position = end_room

    def record_snapshot(self):
        positions = {}

        for agent in self.patients + self.nurses + self.robots + self.doctors + self.swing_doctors:
            if isinstance(agent.position, str):
                positions[agent.id] = self.get_position(agent.position)
            else:
                positions[agent.id] = agent.position

        snapshot = {
            "time": self.current_time,
            "positions": positions
        }

        self.snapshots.append(snapshot)

    def intake_step(self):
        """
        Processes new patient arrivals from the schedule and assigns them to free clerks
        """
        
        new_arrivals = self.pop_arrivals()
        if new_arrivals and self.verbose:
            arrived_ids = [p.id for p in new_arrivals]
            print(f"[ARRIVAL] t={self.current_time}: Patients {arrived_ids} have arrived.")

        free_clerks = [c for c in self.acceptance_staff if c.busy_until <= self.current_time]

        for clerk in free_clerks:
            if not self.entrance_queue:
                break

            patient = self.entrance_queue.pop(0)

            self._move_agent_to_room(patient, "Entrance")
            assigned_priority = clerk.decide_priority(patient)
            assigned_assistance = clerk.decide_assistance(patient)
            self.log_patient_event(patient, f"Entered intake, assigned priority '{assigned_priority}' and assistance '{assigned_assistance}'.")

            patient.pred_priority = assigned_priority
            patient.pred_impairment = assigned_assistance

            processing_time = PROCESSING_TIME_BASE + (PROCESSING_TIME_ASSIST_BONUS if patient.requires_assistance else 0)
            done_time = self.current_time + processing_time
            clerk.busy_until = done_time

            old_status = patient.status
            patient.status = "being_accepted"
            self._apply_shaped_reward(patient, old_status)

            self.push_event(done_time, "acceptance_done", patient, order_key=-{"high":3,"medium":2,"low":1}[assigned_priority])

    def triage_step(self):
        """
        Assigns available Triage Agents to patients waiting in the triage_queue
        """
        
        self.triage_queue.sort(key=lambda p: PRIO_TO_IDX[p.true_priority], reverse=True)

        free_triage_agents = [t for t in self.triage_agents if t.busy_until <= self.current_time]

        while self.triage_queue and free_triage_agents:
            patient = self.triage_queue.pop(0)
            agent = free_triage_agents.pop(0)

            old_status = patient.status
            patient.status = "being_triaged"
            self._apply_shaped_reward(patient, old_status)
            agent.busy_until = self.current_time + TRIAGE_DURATION

            self.log_patient_event(patient, f"Started triage by Agent {agent.id}")
            self.push_event(self.current_time + TRIAGE_DURATION, "triage_done", patient)

    def treatment_step(self):
        """
        Assigns available doctors to waiting patients in each ward
        """

        for room, agents in self.rooms.items():
            if room in ["Entrance", "Triage", "StaffArea", "Exit"]:
                continue
            
            # Identify available doctors and waiting patients in the current room
            available_doctors = [d for d in self.rooms[room] if isinstance(d, Doctor) and d.busy_until <= self.current_time]
            waiting_patients = [p for p in self.rooms[room] if isinstance(p, Patient) and p.status == "triaged"]

            # Handle misrouted patients first without blocking others
            # Create a copy to modify the list while iterating
            for patient in list(waiting_patients): 
                correct_department = self.route_patient(patient)
                if patient.pred_illness != correct_department:
                    self.log_patient_event(patient, f"Misrouted to {room}. Preparing to return to Triage.")
                    old_status = patient.status

                    if patient.requires_assistance:
                        patient.status = "waiting_for_escort"
                        self._apply_shaped_reward(patient, old_status)
                        self.request_assistance(PRIO_TRIAGE, room, patient)
                    else:
                        patient.status = "moving_to_triage"
                        self._apply_shaped_reward(patient, old_status)
                        self._initiate_patient_move(patient, room, "Triage", "patient_arrived_at_triage")
                    
                    # Remove the misrouted patient from the waiting list for this tick
                    waiting_patients.remove(patient)

            # Assign free doctors to the remaining (correctly routed) waiting patients
            # This loop continues until either doctors or patients run out.
            while available_doctors and waiting_patients:
                doctor = available_doctors.pop(0)
                patient = waiting_patients.pop(0)

                if patient.arrival_to_room_time is not None:
                    ward_wait = self.current_time - patient.arrival_to_room_time
                    self.ward_wait_times.append(ward_wait)

                self.log_patient_event(patient, f"Treatment started by Doctor {doctor.id} in '{room}'.")
                doctor.busy_until = self.current_time + TREATMENT_DURATION
                old_status = patient.status
                patient.status = "being_treated"
                self._apply_shaped_reward(patient, old_status)
                self.push_event(self.current_time + TREATMENT_DURATION, "treatment_complete", patient)

    def dispatch_helpers(self):
        """
        Checks for open assistance requests and idle helpers, triggering the RL agent if necessary
        """
        
        idle_helpers = [h for h in (self.nurses + self.robots) if h.busy_until <= self.current_time and h.role=="idle"]

        if self.rl_mode and idle_helpers and self.assistance_requests:
            self._hook = "awaiting_dispatch"
            self._payload = {
                "idle_helpers": idle_helpers,
                "assistance_requests": self.assistance_requests.copy(),
            }

    def handle_acceptance_done(self, patient):
        """
        Handles patient state after initial processing. They either wait for an escort
        or start moving to Triage on their own
        """
        
        self._move_agent_to_room(patient, "Entrance") # Patient is physically at the entrance
        old_status = patient.status

        if patient.requires_assistance:
            self.request_assistance(PRIO_ENTRANCE, "Entrance", patient=patient)
            patient.status = "waiting_for_escort"
            self.log_patient_event(patient, "Needs escort to Triage. Waiting for helper.")
        else:
            patient.status = "moving_to_triage"
            self._initiate_patient_move(patient, "Entrance", "Triage", "patient_arrived_at_triage")

        self._apply_shaped_reward(patient, old_status)

    def handle_patient_arrived_at_triage(self, patient):
        """
        Handles a patient's arrival at Triage after self-ambulating
        They are now officially in the Triage queue
        """
        
        self._move_agent_to_room(patient, "Triage")
        if patient not in self.triage_queue:
            self.triage_queue.append(patient)
            
        if not hasattr(patient, 'arrival_to_triage_queue_time'):
            patient.arrival_to_triage_queue_time = self.current_time
        
        old_status = patient.status
        patient.status = "in_triage_queue"
        self._apply_shaped_reward(patient, old_status)
        self.log_patient_event(patient, f"Arrived at Triage and entered the queue.")

    def handle_patient_arrived_at_ward(self, patient):
        """
        Handles a patient's arrival at their designated treatment ward
        They are now ready for a doctor
        """
        
        patient.arrival_to_room_time = self.current_time
        ward = patient.target_room
        self._move_agent_to_room(patient, ward)
        
        old_status = patient.status
        patient.status = "triaged" # 'triaged' means waiting for treatment in a ward
        self._apply_shaped_reward(patient, old_status)
        self.log_patient_event(patient, f"Arrived at {ward} and is waiting for a doctor.")

    def handle_patient_arrived_at_exit(self, patient):
        """
        Handles a patient's arrival at the Exit
        The simulation journey for them is complete
        """
        
        self._move_agent_to_room(patient, "Exit")
        
        old_status = patient.status
        patient.status = "exited"
        self._apply_shaped_reward(patient, old_status)
        self.log_patient_event(patient, f"Reached the Exit and has now left the hospital.")

    def handle_helper_arrived(self, helper):
        #print(f"DEBUG: handle_helper_arrived called at time {self.current_time} for helper {helper.id}") # <-- ADD THIS
        patient = helper.assigned_patient
        # Explicitly update helper's physical position to their target destination for this event.
        helper.position = helper.target_room

        # Stage 1: Helper arrives at the patient's location for pickup
        if helper.role == "moving":
            self.log_patient_event(patient, f"Helper {helper.id} ({helper.name}) arrived for escort pickup at {helper.position}.")
            
            if not patient or patient.status not in ["waiting_for_escort", "waiting_for_exit"]:
                self.log_patient_event(patient, f"Patient state is '{patient.status}', no longer needs this escort. Helper returning to base.")
                helper.role = "moving"
                helper.assigned_patient = None
                travel = self.get_travel_time(helper.position, "StaffArea", helper.speed)
                helper.busy_until = self.current_time + travel
                self.push_event(helper.busy_until, "helper_return_to_base", helper)
                return

            # Determine the final destination of the escort.
            destination = ""
            current_loc = helper.position
            if patient.status == "waiting_for_escort":
                if current_loc == "Entrance": destination = "Triage"
                elif current_loc == "Triage": destination = patient.target_room
                else: destination = "Triage"
            elif patient.status == "waiting_for_exit":
                destination = "Exit"
            
            # Start the joint journey
            travel_time = self.get_travel_time(current_loc, destination, min(helper.speed, patient.speed))
            arrival_time = self.current_time + travel_time
            
            old_status = patient.status
            patient.status = "being_escorted"
            self._apply_shaped_reward(patient, old_status)
            helper.role = "escorting"
            
            self._move_agent_to_room(patient, None)
            self._move_agent_to_room(helper, None)
            
            patient.busy_until = arrival_time
            helper.busy_until = arrival_time
            helper.target_room = destination
            
            self.push_event(arrival_time, "helper_arrived", helper)
            self.log_patient_event(patient, f"Escort from {current_loc} to {destination} begins.")
            
            wait_time = self.current_time - patient.assistance_request_time
            self.escort_wait_times.append(wait_time)
            #print(f"DEBUG: Appended to escort_wait_times. New length: {len(self.escort_wait_times)}")

        # Stage 2: Helper and patient arrive together at the final destination
        elif helper.role == "escorting":
            destination = helper.target_room
            self.log_patient_event(patient, f"Escort complete. Arrived at {destination}.")
            
            # Update positions and place patient in the room.
            helper.position = destination
            patient.position = destination
            self._move_agent_to_room(patient, destination)
            
            # Update patient status based on where they arrived.
            old_status = patient.status
            if destination == "Triage":
                patient.status = "in_triage_queue"
                if patient not in self.triage_queue: self.triage_queue.append(patient)
            elif destination == "Exit":
                patient.status = "exited"
            else:
                patient.status = "triaged"
            self._apply_shaped_reward(patient, old_status)

            # Send the helper back to the staff area.
            helper.role = "moving"
            helper.assigned_patient = None
            travel = self.get_travel_time(helper.position, "StaffArea", helper.speed)
            helper.busy_until = self.current_time + travel
            self.push_event(helper.busy_until, "helper_return_to_base", helper)
                                    
    def handle_doctor_arrived(self, doc):
        room = doc.target_room
        doc.busy_until = self.current_time + 1 # cooldown
        self._move_agent_to_room(doc, room)

    def handle_helper_return_to_base(self, helper):
        """
        Handles a helper arriving back at the StaffArea, making them idle
        """
        
        helper.role = "idle"
        self._move_agent_to_room(helper, "StaffArea")

    def handle_triage_done(self, patient):
        if self.rl_mode:
            self._hook = "awaiting_routing"
            self.router_choices = ["PromptCare", "AcuteCare", "Psychiatry", "Pediatrics", "Resuscitation", "Imaging"]
            self.log_patient_event(patient, "Triage assessment complete. Awaiting routing decision.")
            self._payload = {"patient": patient}
            return
        
        patient.target_room = self.route_patient(patient)
        self._finish_triage(patient)


    def _finish_triage(self, patient):
        """
        Handles patient state after triage and calculates a tiered, time-based reward
        for the triage_router's decision based on the ILLNESS_WARD_MAP.
        """
        
        chosen_ward = patient.target_room
        if chosen_ward is None:
            if self.verbose: print(f"[INFO] Target room not set for Patient {patient.id}, deferring.")
            return 0.0 # Return zero reward if no decision was made

        # Get the valid wards and their reward multipliers for the patient's true illness
        valid_wards_map = ILLNESS_WARD_MAP.get(patient.true_illness, {})

        # Log the decision for metrics by checking if the choice was optimal
        is_optimal_choice = valid_wards_map.get(chosen_ward) == 1.0
        self.routing_decisions.append(is_optimal_choice)

        reward = 0
        if chosen_ward in valid_wards_map:
            # optimal/acceptable choice    
            reward_multiplier = valid_wards_map[chosen_ward]
            base_positive_reward = 100
            reward = base_positive_reward * reward_multiplier
            self.log_patient_event(patient, f"Routed to '{chosen_ward}'. Quality: {reward_multiplier*100}%.")
            
            if reward_multiplier == 1.0:
                self.routing_outcomes.append("perfect")
            else:
                self.routing_outcomes.append("acceptable")

        else:
            # The penalty is the time wasted on the detour to a wrong ward and back.
            time_to_wrong_ward = self.get_travel_time('Triage', chosen_ward, patient.speed) #
            time_back_to_triage = self.get_travel_time(chosen_ward, 'Triage', patient.speed) #
            wasted_time_penalty = -1 * (time_to_wrong_ward + time_back_to_triage) #
            reward = wasted_time_penalty #
            self.log_patient_event(patient, f"MISROUTED to '{chosen_ward}'. Incurred penalty.")
            
            self.routing_outcomes.append("wrong")

        self._move_agent_to_room(patient, "Triage") 
        old_status = patient.status 
        if patient.requires_assistance: 
            patient.status = "waiting_for_escort" 
            self.request_assistance(PRIO_TRIAGE, "Triage", patient) 
            self.log_patient_event(patient, "Needs escort to ward. Waiting for helper.") 
        else:
            patient.status = "moving_to_ward" 
            self._initiate_patient_move(patient, "Triage", chosen_ward, "patient_arrived_at_ward") 

        self._apply_shaped_reward(patient, old_status)

        return reward
        
    def handle_treatment_complete(self, patient):
        if patient.status == "done":
            if self.verbose: print(f"[WARNING] Duplicate treatment_complete event for Patient {patient.id}.")
            return

        old_status = patient.status
        patient.status = "done"
        self._apply_shaped_reward(patient, old_status)
        self.completed_patients.append(patient)
        if self.verbose: print(f"Patient {patient.id} completed treatment.")

        self.push_event(self.current_time, "ready_to_exit", patient)
        self.log_patient_event(patient, "Treatment complete, ready to exit.")
        
        completion_reward = 200
        stay_duration = self.current_time - patient.arrival_time
        if stay_duration <= 30: 
            completion_reward += 50
        self.credit_patient_completion(patient, completion_reward)

    def handle_ready_to_exit(self, patient):
        """
        Handles a patient who is ready to leave
        They either wait for an escort or start walking out
        """
        
        old_status = patient.status

        if patient.requires_assistance:
            patient.status = "waiting_for_exit"
            # The patient is physically in their last treatment room
            self._move_agent_to_room(patient, patient.position)
            self.request_assistance(PRIO_DEPT, patient.position, patient)
            self.log_patient_event(patient, "Requested assistance to exit.")
        else:
            patient.status = "moving_to_exit"
            self._initiate_patient_move(patient, patient.position, "Exit", "patient_arrived_at_exit")
        
        self._apply_shaped_reward(patient, old_status)

    def route_patient(self, patient):
        return {
            "general": "PromptCare", "cardio": "AcuteCare", "psychiatric": "Psychiatry",
            "pediatric": "Pediatrics", "emergency": "Resuscitation", "xray": "Imaging"
        }.get(patient.true_illness, "PromptCare")

    def pop_arrivals(self):
        newly_arrived = []
        while self.arrival_schedule and self.arrival_schedule[0][0] <= self.current_time:
            _, _, p = heapq.heappop(self.arrival_schedule)
            self.entrance_queue.append(p)
            newly_arrived.append(p)
        return newly_arrived

    def pop_reward(self):
        r = self._reward_buffer
        self._reward_buffer = 0.0
        return r

    def schedule_patient(self, patient):
        heapq.heappush(self.arrival_schedule, (patient.arrival_time, patient.id, patient))
        self.patients.append(patient)

    def request_assistance(self, priority, room_name, patient):
        request_time = self.current_time
        patient.assistance_request_time = request_time
        self._request_counter += 1
        self.assistance_requests.append((request_time, self._request_counter, patient, room_name))

    def export_patient_logs(self, filename="patient_logs.csv", max_patients=100):
        import csv
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["PatientID", "Time", "Event"])
            for patient in sorted(self.patients, key=lambda p: p.id)[:max_patients]:
                for time, event in getattr(patient, "event_log", []):
                    writer.writerow([patient.id, time, event])

    def finalise_episode(self, save_logs=True, make_gif=True):
        if save_logs:
            self.export_patient_logs("first10_patient_log.csv", max_patients=10)
        # if make_gif and self.visualise and self.snapshots:
        #     render_simulation(self)
    
    def log_patient_event(self, patient, event_description):
        log_entry = (self.current_time, event_description)
        if not hasattr(patient, 'event_log'):
            patient.event_log = []
        patient.event_log.append(log_entry)