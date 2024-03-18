import numpy as np
from qubit import Qubit

class DensityMatrix(Qubit):
    
    _state: np.array
    dim: int
    
    def __init__(self, state: np.array) -> None:
        """Initializes a density matrix."""
        super().__init__(state)
        self.dim = len(self._state)
        
    def compute_normalization(self, state):
        """Computes the normalization of the density matrix by taking its trace."""
        return np.trace(state)
        
    def to_state_vector(self):
        """Converts the density matrix to a state vector when it is in a pure state.
        If it is not a pure state, it will immediately return the denisty matrix.
        It does this by computing the eigenvalues and eigenvectors of the density matrix,
        and then summing them into the state.
        TODO: Make this return a qubit instead of a numpy array."""
        if np.trace(self._state @ self._state) != 1:
            print("Not a pure state, remaining as a density matrix...")
            return self._state
        values, vectors = np.linalg.eig(self._state)
        state = np.zeros(self.dim, dtype=complex)
        for value, vector in zip(values, vectors.T):
            state += value * vector
        return state
    
    def partial_trace(self, system):
        """Computes the partial trace of a system with respect to a subsystem.
        TODO: Make this return a density matrix instead of a numpy array.

        Args:
            system (int): An integer representing which subsystem to trace out.

        Returns:
            np array: The matrix after tracing it out. 
        """
        if system == 1:
            return np.trace(self._state.reshape(2, self.dim // 2, 2, self.dim // 2), axis1=0, axis2=2)
        return np.trace(self._state.reshape(*[2] * 4), axis1=1, axis2=3)
        
    def _get_measurement_stats(self):
        """
        Measures the qubit in the computational basis.
        Returns a list of collapsed states and the probability associated with them.
        """
        zero_state = np.array([[1, 0], [0, 0]])
        one_state = np.array([[0, 0], [0, 1]])
        zero_prob = np.trace(zero_state @ self._state)
        one_prob = np.trace(one_state @ self._state)
        return ([DensityMatrix(zero_state), DensityMatrix(one_state)], [zero_prob.item(), one_prob.item()]) 