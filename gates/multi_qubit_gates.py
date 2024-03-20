import numpy as np

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
        state = self.register[self.targets[0]]
        target_qubits = [self.register[target] for target in self.targets[1:]]
        # Depending on whether or not we have entanglement, we should act with state vectors or density matrices
        if self.register.is_density_matrix:
            return DensityMatrix(state.tensor(target_qubits))
        return StateVector(state.tensor(target_qubits))
    
    def to_qubit(self, evolved_state: Qubit) -> list[Qubit]:
        """Converts the evolved state into a list of two qubits by tracing out both subsystems.

        Args:
            evolved_state (Qubit): The state after having been acted on by the gate.

        Returns:
            list[Qubit]: A list of the target qubits that have been evolved.
        """
        if not isinstance(evolved_state, DensityMatrix):
            evolved_state = DensityMatrix(evolved_state.to_density_matrix())
        rho_1 = DensityMatrix(evolved_state.partial_trace(system=1))
        rho_2 = DensityMatrix(evolved_state.partial_trace(system=2))
        if not self.register.is_density_matrix:
            final_state_1 = StateVector(rho_2.to_state_vector())
            final_state_2 = StateVector(rho_1.to_state_vector())
            return [final_state_1, final_state_2]
        return [rho_1, rho_2]
        
    def matrix_rep(self):
        return self._matrix_representation

class SWAP(MultiQubitGate):
    
    _matrix_representation = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    
    def __init__(self, register, targets, theta = 0) -> None:
        super().__init__(register, targets, theta)