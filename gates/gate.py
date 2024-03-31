from qubits import Register, DensityMatrix, StateVector, Qubit
import numpy as np

class Gate:
    """Abstract Gate class, representing a generic quantum gate."""
    
    ZZ = np.array([[1, 0], [0, 0]])
    OO = np.array([[0, 0], [0, 1]])
    OZ = np.array([[0, 0], [1, 0]])
    ZO = np.array([[0, 1], [0, 0]])
    
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
        evolved_state = DensityMatrix(gate @ state.get_state() @ gate.conj().T)
        self.register.update_state(evolved_state)
        return self.register
    
    def matrix_rep(self) -> np.array:
        raise NotImplementedError
    
    def get_state(self) -> Qubit:
        raise NotImplementedError
    
    def get_all_terms(self) -> list[np.array]:
        raise NotImplementedError
        