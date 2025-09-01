class Memory:
    def clear_memory(self):
        self.states        = []
        self.extra_obs     = []
        self.actions       = []
        self.logprobs      = []
        self.state_values  = []
        self.rewards       = []
        self.stream_ids    = []
        self.player_ids = []

    def __init__(self):
        self.clear_memory()
