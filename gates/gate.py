from qubits import Register, DensityMatrix, StateVector, Qubit
import numpy as np

class Gate:
    """Abstract Gate class, representing a generic quantum gate."""
    
    targets: list[int] = []
    register: Register = None
    theta: float = 0.0
    
    def __init__(self, register: Register, targets: list[int], theta: float = 0) -> None:
        """Initializes the gate with a register that the gate acts on and a list of targets in the register.

        Args:
            register (Register): The register that the gate acts on.
            targets (list[int]): The target qubits that the gate should act on.
        """
        self.register = register
        self.targets = targets
        self.theta = theta
        
    def evolve(self) -> Register:
        """Evolves the register by making the gate act on the qubit.

        Returns:
            Register: The evolved register.
        """
        gate = self.matrix_rep()
        state = self.get_state()
        if isinstance(state, DensityMatrix):
            # measure as a density matrix
            evolved_state = DensityMatrix(gate @ state.get_state() @ np.matrix(gate).getH())
        else:
            # measure as a state vector
            evolved_state = StateVector(gate @ state.ket())
        # turn the evolved state into a list of qubits and place them in the register.
        qubits = self.to_qubit(evolved_state)
        for target, qubit in zip(self.targets, qubits):
            self.register[target] = qubit
        return self.register
    
    def matrix_rep(self) -> np.array:
        raise NotImplementedError
    
    def get_state(self) -> Qubit:
        raise NotImplementedError
    
    def to_qubit(self, evolved_state: Qubit) -> list[Qubit]:
        raise NotImplementedError
        