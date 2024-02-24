import numpy as np

from single_gate import SingleGate

class X(SingleGate):
    
    def __init__(self, targets) -> None:
        super().__init__(targets)
        
    def matrix_rep(self):
        return np.array([[0, 1], [1, 0]])