# policy_algorithms/FEN/Memory.py
class MemoryFEN:
    __slots__ = ("states","actions","logprobs","rewards",
                 "z_indices","z_logprobs","macro_flags")

    def __init__(self) -> None:
        self.clear()

    def clear(self):
        self.states      = []
        self.actions     = []
        self.logprobs    = []
        self.rewards     = []
        self.z_indices   = []
        self.z_logprobs  = []
        self.macro_flags = []
