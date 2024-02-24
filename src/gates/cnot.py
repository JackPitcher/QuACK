import numpy as np

from controlled_gate import ControlledGate
from x import X

class CNOT(ControlledGate):
    
    def __init__(self, targets, controls) -> None:
        super().__init__(targets, controls)
        self.single_gate = X(targets)
        
    def matrix_rep(self):
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])