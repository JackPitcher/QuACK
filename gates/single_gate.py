import numpy as np

from gate import Gate
from qubits import Register, StateVector, DensityMatrix

class SingleGate(Gate):
    """Abstract class to represent a single qubit gate."""
    
    _matrix_representation = None
    
    def __init__(self, register: Register, targets: list[int]) -> None:
        super().__init__(register, targets)
        
    def get_state(self):
        """Simply returns the state that the gate should act on, which is target qubit."""
        return self.register[self.targets[0]]
    
    def to_qubit(self, evolved_state):
        """Since evolved state will just be in 2 dimensions, we just put it in a list."""
        return [evolved_state]
        
    def matrix_rep(self):
        """Returns the matrix representation of a given gate."""
        return self._matrix_representation

class X(SingleGate):
    
    _matrix_representation = np.array([[0, 1], [1, 0]])
        
class Y(SingleGate):
    
    _matrix_representation = np.array([[0, -1j], [1j, 0]])
    
class Z(SingleGate):
    
    _matrix_representation = np.array([[1, 0], [0, -1]])
    
class H(SingleGate):
    
    _matrix_representation = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
    
    
                