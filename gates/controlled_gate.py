import numpy as np

from gates.gate import Gate
from gates.single_gate import X, Z, SingleGate
from qubits import Register, StateVector, DensityMatrix, Qubit
from gates.multi_qubit_gates import MultiQubitGate, SWAP
from scipy.linalg import block_diag

class ControlledGate(Gate):
    """A controlled gate that controls a particular unitary gate."""
    
    controls: int = []
    gate: SingleGate | MultiQubitGate = None
    x_names: list[str] = ['x', 'not', 'pauli_x', 'sigma_x']
    swap_names: list[str] = ['swap']
    
    def __init__(self, register: Register, targets: list[int], 
                 controls: list[int], theta: float = 0) -> None:
        super().__init__(register, targets, theta)
        self.controls = controls
        
    def get_state(self) -> Qubit:
        """Gets the state by tensoring together the control qubits with the target qubits.

        Returns:
            Qubit: The combined state.
        """
        return self.register.state
    
    def matrix_rep(self):
        matrix_terms = self.get_all_terms()
        qubit_list = range(len(self.targets + self.controls))
        qubit_counter = 0
        if 0 in self.controls + self.targets:
            terms = matrix_terms[qubit_counter]
            qubit_counter += 1
        else:
            terms = [np.eye(2) for _ in range(len(matrix_terms[0]))]
        
        for i in range(self.register.num_qubits - 1):
            if i + 1 in self.controls + self.targets:
                qubit = qubit_list[qubit_counter]
                qubit_counter += 1
                terms = [np.kron(terms[j], matrix_terms[qubit][j]) for j in range(len(terms))]
            else:
                terms = [np.kron(terms[j], np.eye(2)) for j in range(len(terms))]
        return sum(terms)

class CNOT(ControlledGate):
    
    def __init__(self, register: Register, targets: list[int], controls: list[int], theta: float = 0) -> None:
        super().__init__(register, targets, controls, theta)
        self.gate = X(self.register, self.targets)
        
    def get_all_terms(self) -> list:
        return [[self.OO, self.OO, self.ZZ, self.ZZ], [self.OZ, self.ZO, self.OO, self.ZZ]]

class CZ(ControlledGate):
    
    def __init__(self, register: Register, targets: list[int], controls: list[int], theta: float = 0) -> None:
        super().__init__(register, targets, controls, theta)
        self.gate = Z(self.register, self.targets)
        
    def get_all_terms(self) -> list:
        return [[self.OO, self.OO, self.ZZ, self.ZZ], [self.OO, self.ZZ, self.OO, -1 * self.ZZ]]
    
class CSWAP(ControlledGate):
    
    def __init__(self, register: Register, targets: list[int], controls: list[int], theta: float = 0) -> None:
        super().__init__(register, targets, controls, theta)
        self.gate = SWAP(self.register, self.targets)
        
    def get_all_terms(self) -> list:
        return [[self.OO, self.OO, self.OO, self.OO, self.ZZ, self.ZZ, self.ZZ, self.ZZ], 
                [self.OO, self.OO, self.OO, self.OO, self.ZZ, self.ZO, self.OZ, self.ZZ], 
                [self.OO, self.OO, self.OO, self.OO, self.ZZ, self.OZ, self.ZO, self.ZZ]]
        