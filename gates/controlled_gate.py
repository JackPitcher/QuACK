import numpy as np

from gate import Gate
from single_gate import X, Z, SingleGate
from qubits import Register, StateVector, DensityMatrix, Qubit
from multi_qubit_gates import MultiQubitGate, SWAP
from scipy.linalg import block_diag

class ControlledGate(Gate):
    """A controlled gate that controls a particular unitary gate."""
    
    controls: int = []
    gate: SingleGate | MultiQubitGate = None
    x_names: list[str] = ['x', 'not', 'pauli_x', 'sigma_x']
    swap_names: list[str] = ['swap']
    
    def __init__(self, register: Register, targets: list[int], controls: list[int], gate_name: str = '') -> None:
        super().__init__(register, targets)
        self.controls = controls
        if gate_name:
            self.gate = self.get_gate(gate_name)
        
    def to_qubit(self, evolved_state: Qubit) -> list[Qubit]:
        """Turns the evolved state into a list of qubits.

        Args:
            evolved_state (Qubit): The evolved state.

        Returns:
            list[Qubit]: A list of qubits corresponding that evolved.
        """
        if not isinstance(evolved_state, DensityMatrix):
            evolved_state = DensityMatrix(evolved_state.to_density_matrix())
        for _ in range(len(self.controls)):
            # trace out control bits since control bits do not evolve.
            evolved_state = DensityMatrix(evolved_state.partial_trace(system=1))
        if not self.register.is_density_matrix and np.trace(evolved_state.get_state() @ evolved_state.get_state()) == 1:
            evolved_state = StateVector(evolved_state.to_state_vector())
        else:
            # controlled gates enable entanglement, so if the state is now entangled, we should
            # update the register so that everything is represented as a density matrix.
            self.register.set_to_dm()
        return self.gate.to_qubit(evolved_state)
        
    def get_state(self) -> Qubit:
        """Gets the state by tensoring together the control qubits with the target qubits.

        Returns:
            Qubit: The combined state.
        """
        control_qubit = self.register[self.controls[0]]
        if len(self.controls) > 1:
            control_qubits = [self.register[control] for control in self.controls[1:]]
            if self.register.is_density_matrix:
                control_qubit = DensityMatrix(control_qubit.tensor(control_qubits))
            else:
                control_qubit = StateVector(control_qubit.tensor(control_qubits))
        target = self.register[self.targets[0]]
        if len(self.targets) > 1:
            target_qubits = [self.register[target] for target in self.targets[1:]]
            if self.register.is_density_matrix:
                target = DensityMatrix(target.tensor(target_qubits))
            else:
                target = StateVector(target.tensor(target_qubits))
        if self.register.is_density_matrix:
            return DensityMatrix(control_qubit.tensor(target))
        else:
            return StateVector(control_qubit.tensor(target))
        
    def matrix_rep(self):
        """Expands the gate to work with N control bits"""
        I = np.eye(2 ** len(self.targets))
        num_controls = len(self.controls)
        block_matrices = [I] * (2 ** num_controls - 1) + [self.gate.matrix_rep()]
        gate = block_diag(*block_matrices) 
        return gate
    
    def get_gate(self, gate_name: str):
        if gate_name.lower() in self.x_names:
            return X(self.register, self.targets)
        if gate_name.lower() in self.swap_names:
            return SWAP(self.register, self.targets)
        raise ValueError("Not an acceptable name!")

class CNOT(ControlledGate):
    
    def __init__(self, register: Register, targets: list[int], controls: list[int]) -> None:
        super().__init__(register, targets, controls)
        self.gate = X(self.register, self.targets)

class CZ(ControlledGate):
    def __init__(self, register: Register, targets: list[int], controls: list[int]) -> None:
        super().__init__(register, targets, controls)
        self.gate = Z(self.register, self.targets)
    
        
class CSWAP(ControlledGate):
    
    def __init__(self, register: Register, targets: list[int], controls: list[int], gate_name: str = '') -> None:
        super().__init__(register, targets, controls, gate_name)
        self.gate = SWAP(self.register, self.targets)
        