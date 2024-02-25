import numpy as np

from controlled_gate import ControlledGate
from ..qubits.register import Register
from x import X

class CNOT(ControlledGate):
    
    def __init__(self, targets, controls) -> None:
        super().__init__(targets, controls)
        self.single_gate = X(targets)
        
    def matrix_rep(self, register: Register = None, N: int = 2):
        if N > 2:
            gate = self.expand_gate(register, N)
        else:
            gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        return gate