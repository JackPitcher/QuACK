import numpy as np
from numpy.core.multiarray import array as array

from gates import Gate
from qubits import StateVector, DensityMatrix, Qubit, Register

class MultiQubitGate(Gate):
    
    _matrix_representation = None
    
    def __init__(self, register: Register, targets: list[int], theta: float = 0) -> None:
        super().__init__(register, targets, theta)
        
    def get_state(self) -> Qubit:
        """Gets the state to act on. Since there are multiple qubits to act on,
        we need to tensor them together.

        Returns:
            Qubit: Either a density matrix or state vector with all target states tensored together.
        """
        return self.register.state
        
    def matrix_rep(self):
        matrix_terms = self.get_all_terms()
        qubit_list = range(len(self.targets))
        qubit_counter = 0
        
        if 0 in self.targets:
            terms = matrix_terms[qubit_counter]
            qubit_counter += 1
        else:
            terms = [np.eye(2) for _ in range(len(matrix_terms[0]))]
        
        for i in range(self.register.num_qubits - 1):
            if i + 1 in self.targets:
                qubit = qubit_list[qubit_counter]
                qubit_counter += 1
                terms = [np.kron(terms[j], matrix_terms[qubit][j]) for j in range(len(terms))]
            else:
                terms = [np.kron(terms[j], np.eye(2)) for j in range(len(terms))]
        return sum(terms)

class SWAP(MultiQubitGate):
    
    _matrix_representation = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    
    def get_all_terms(self) -> list:
        return [[self.ZZ, self.OZ, self.ZO, self.OO], [self.ZZ, self.OZ, self.ZO, self.OO]]
    
    def __init__(self, register, targets, theta = 0) -> None:
        super().__init__(register, targets, theta)