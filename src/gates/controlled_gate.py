import numpy as np

from gate import Gate
from ..qubits.register import Register
from single_gate import SingleGate
from scipy.linalg import block_diag

class ControlledGate(Gate):
    
    controls = []
    single_gate: SingleGate = None
    
    def __init__(self, targets, controls) -> None:
        super().__init__(targets)
        self.controls = controls
        
    def expand_gate(self, register: Register):
        """Expands the gate to work with N qubits.
        Only works for 1 control qubit for now."""
        N = len(register)
        num_targets = len(self.targets)
        U = self.single_gate.expand_gate(num_targets)
        I = np.eye(2)
        
        block_matrices = [I] * num_targets + [U]
        gate = block_diag(*block_matrices)
        for _ in range(N - num_targets - 1):
            gate = np.kron(gate, I)
        
        other_qubits = [i for i in range(N) if i not in self.targets + self.controls]
        register.reorder(self.controls + self.targets + other_qubits)
        return gate
        