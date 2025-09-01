import random

SYMPTOM_LIST = [
    'fever', 'cough', 'minor_pain', 'chest_pain', 'shortness_of_breath',
    'high_blood_pressure', 'agitation', 'confusion', 'severe_trauma',
    'unconsciousness', 'suspected_fracture', 'is_child'
]


def generate_patient_features(true_illness: str):
    """
    Generates a dictionary of deterministic symptom features based on a patient's true illness
    """
    
    symptoms = {s: 0 for s in SYMPTOM_LIST}

    if true_illness == 'pediatric':
        symptoms['is_child'] = 1
        symptoms['fever'] = 1
        symptoms['cough'] = 1

    elif true_illness == 'general':
        symptoms['fever'] = 1
        symptoms['minor_pain'] = 1

    elif true_illness == 'cardio':
        symptoms['chest_pain'] = 1
        symptoms['shortness_of_breath'] = 1
        symptoms['high_blood_pressure'] = 1

    elif true_illness == 'xray':
        symptoms['suspected_fracture'] = 1
        symptoms['minor_pain'] = 1

    elif true_illness == 'psychiatric':
        symptoms['confusion'] = 1
        symptoms['high_blood_pressure'] = 1

    elif true_illness == 'emergency':
        symptoms['unconsciousness'] = 1
        symptoms['chest_pain'] = 1
        symptoms['shortness_of_breath'] = 1
        
    return symptoms

class Agent:
    def __init__(self, agent_id, name, position):
        
        self.id = agent_id
        self.name = name
        self.position = position
        self.speed = 90 # default meters per minute
        self.busy_until = 0
    
    def move(self, destination):
        self.position = destination
    
class Patient(Agent):
    def __init__(self, agent_id, priority, position, illness, impairment_level, arrival_time):
        super().__init__(agent_id, "Patient", position)
        
        # Ground truth attributes
        self.true_priority = priority 
        self.true_impairment = impairment_level 
        self.true_illness = illness 
        
        # Predicted attributes
        self.pred_priority   = None
        self.pred_impairment = None
        self.pred_illness    = None
        self.symptoms = generate_patient_features(self.true_illness) # symptoms
                
        # Simulation attributes
        self.requires_assistance = impairment_level != "none"       
        self.routing_attempts = 0               
        self.status = "waiting"
        self.arrival_time = arrival_time
        self.start_treatment_time = None
        self.target_room = None
        self.arrival_to_room_time = None
        
        # Derived attributes
        speed_map  = {"none": 75, "low": 60, "high": 45}
        self.speed = speed_map[impairment_level]
        
        # Store the memory index of the routing decision (avoid troublesome assignment of the rewards)      
        self.triage_memory_idx = None 
        
        self.event_log = [] # log the event for patients
        
        self.assistance_request_time = 0 # track when an assistance request is made
        
class AcceptanceAgents(Agent):
    """
    Agent assigning a priority level and nurse/robot assistance
    """
    
    def __init__(self, agent_id, position="Entrance"):
        super().__init__(agent_id, "AcceptanceStaff", position)
        self.busy_until = 0
        self.priority_choices = ["low", "medium", "high"]
        self.assistance_choices = ["none", "low", "high"]
    
    def decide_priority(self, patient):
        """ 
        A %-based accurate heuristic for assigning priority
        """
        
        if random.random() < 1:
            return patient.true_priority # Be correct x% of the time
        else:
            return random.choice(self.priority_choices) # Make a random mistake

    def decide_assistance(self, patient):
        """ 
        An 95% accurate heuristic for assigning assistance
        """
        
        if random.random() < 0.95:
            return patient.true_impairment
        else:
            return random.choice(self.assistance_choices)

class TriageAgent(Agent):
    def __init__(self, agent_id, position="Triage"):
        super().__init__(agent_id, "TriageAgent", position)
        self.busy_until = 0
        
    def decide_room(self, patient):
        return patient.illness
        
class Nurse(Agent):
    def __init__(self, agent_id, position='StaffArea'):
        super().__init__(agent_id, "Nurse", position)
        self.role = "idle"
        self.busy_until = 0
        self.target_room = None
        self.speed = 90

class Doctor(Agent):
    def __init__(self, agent_id, specialty, position, swing=False):
        super().__init__(agent_id, "Doctor", position)
        self.busy_until = 0
        self.specialty = specialty
        self.swing = swing
        
class Robot(Agent):
    def __init__(self, agent_id, position='StaffArea'):
        super().__init__(agent_id, "Robot", position)
        self.role = "idle"
        self.busy_until = 0
        self.target_room = None
        self.speed = 100
        
    def assist(self, patient, destination, rooms):
        print(f"Robot assisting patient {patient.id} to {destination}.")
        rooms[destination].append(patient)