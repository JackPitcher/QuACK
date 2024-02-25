import numpy as np

from single_gate import SingleGate
from ..qubits.register import Register

class X(SingleGate):
    
    def __init__(self, targets) -> None:
        super().__init__(targets)
        
    def matrix_rep(self, register: Register = None,  N: int = 1):
        if N > 1:
            gate = self.expand_gate(register=register, N=N)
        else:
            gate = np.array([[0, 1], [1, 0]])
        return gate