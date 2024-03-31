import numpy as np

from gates.gate import Gate
from qubits import Register, StateVector, DensityMatrix

class SingleGate(Gate):
    """Abstract class to represent a single qubit gate."""
    
    _matrix_representation = None
    
    def __init__(self, register: Register, targets: list[int], theta: float = 0.0) -> None:
        super().__init__(register, targets, theta)
        
    def get_state(self):
        """Simply returns the state that the gate should act on, which is target qubit."""
        return self.register.state
        
    def matrix_rep(self):
        """Returns the matrix representation of a given gate."""
        matrix_terms = self.get_all_terms()
        
        if self.targets[0] == 0:
            terms = matrix_terms
        else:
            terms = [np.eye(2) for _ in range(len(matrix_terms))]
        for i in range(self.register.num_qubits - 1):
            if i + 1 == self.targets[0]:
                terms = [np.kron(terms[j], matrix_terms[j]) for j in range(len(terms))]
            else:
                terms = [np.kron(terms[j], np.eye(2)) for j in range(len(terms))]
        return sum(terms)

class X(SingleGate):
    
    _matrix_representation = np.array([[0, 1], [1, 0]])
    
    def get_all_terms(self):
        return [self.ZO, self.OZ]
        
class Y(SingleGate):
    
    _matrix_representation = np.array([[0, -1j], [1j, 0]])
    
    def get_all_terms(self):
        return [-1j * self.ZO, 1j * self.OZ]
    
class Z(SingleGate):
    
    _matrix_representation = np.array([[1, 0], [0, -1]])
    
    def get_all_terms(self):
        return [self.ZZ, -1 * self.OO]
    
class H(SingleGate):
    
    _matrix_representation = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
    
    def get_all_terms(self):
        return [1 / np.sqrt(2) * self.ZZ, 1 / np.sqrt(2) * self.ZO, 1 / np.sqrt(2) * self.OZ, -1 / np.sqrt(2) * self.OO]
    
class Phase(SingleGate):
    
    _matrix_representation = np.array([[1, 0], [0, -1j]])
    
    def get_all_terms(self):
        return [self.ZZ, -1j * self.OO]
    
class RX(SingleGate):
    
    def __init__(self, register: Register, targets: list[int], theta: float) -> None:
        super().__init__(register, targets, theta)
        self._matrix_representation = np.array([[np.cos(self.theta / 2), -1j * np.sin(self.theta / 2)],
                                                [-1j * np.sin(self.theta / 2), np.cos(self.theta / 2)]])
        
    def get_all_terms(self):
        return [np.cos(self.theta / 2) * self.ZZ, -1j * np.sin(self.theta / 2) * self.ZO, -1j * np.sin(self.theta / 2) * self.OZ, np.cos(self.theta / 2) * self.OO]

class RY(SingleGate):
    
    def __init__(self, register: Register, targets: list[int], theta: float) -> None:
        super().__init__(register, targets, theta)
        self._matrix_representation = np.array([[np.cos(self.theta / 2), -np.sin(self.theta / 2)],
                                                [np.sin(self.theta / 2), np.cos(self.theta / 2)]])
        
    def get_all_terms(self):
        return [np.cos(self.theta / 2) * self.ZZ, -1 * np.sin(self.theta / 2) * self.ZO, 1 * np.sin(self.theta / 2) * self.OZ, np.cos(self.theta / 2) * self.OO]
        
class RZ(SingleGate):
    
    def __init__(self, register: Register, targets: list[int], theta: float) -> None:
        super().__init__(register, targets, theta)
        self._matrix_representation = np.array([[np.exp(-1j * self.theta / 2), 0],
                                                0, np.exp(1j * self.theta / 2)])
        
    def get_all_terms(self):
        return [np.exp(-1j * self.theta / 2) * self.ZZ, np.exp(1j * self.theta / 2) * self.OO]
    
                