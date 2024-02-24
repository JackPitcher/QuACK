import numpy as np

from gate import Gate

class SingleGate(Gate):
    
    def __init__(self, targets) -> None:
        super().__init__(targets)
        
    def expand_gate(self, N: int):
        identity = np.eye(2)
        gate = identity if 0 not in self.targets else self.matrix_rep()
        for i in range(1, N):
            if i in self.targets:
                gate = np.kron(gate, self.matrix_rep())
            else:
                gate = np.kron(gate, identity)
                