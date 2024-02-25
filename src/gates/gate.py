from ..qubits.register import Register

class Gate:
    """Abstract Gate class, representing a generic quantum gate."""
    
    targets = []
    
    def __init__(self, targets) -> None:
        self.targets = targets
    
    def matrix_rep(self, register: Register, N: int):
        raise NotImplementedError
    
    def expand_gate(self, register: Register, N: int):
        raise NotImplementedError
    
    
        