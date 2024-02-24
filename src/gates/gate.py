class Gate:
    """Abstract Gate class, representing a generic quantum gate."""
    
    targets = []
    
    def __init__(self, targets) -> None:
        self.targets = targets
    
    def matrix_rep(self):
        raise NotImplementedError
    
    def expand_gate(self, N: int):
        raise NotImplementedError
    
    
        