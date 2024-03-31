import numpy as np
from typing import Tuple
from qubits.qubit import Qubit

class StateVector(Qubit):
    
    _state: np.array
    
    def __init__(self, state: np.array) -> None:
        """Initializes the state vector. The state vector must have length two, since we only
        currently support two level systems. If the state is not normalized, we normalize the state."""
        super().__init__(state)
        self._state = self._state.flatten()
        
    def compute_normalization(self, state):
        """Computes the normalization of the state by summing the absolute value squared.
        TODO: Make this access self._state instead of passing state parameter."""
        return np.sqrt(np.sum(np.abs(state) ** 2))
    
    def get_state(self):
        return self._state
    
    def _as_qubit(self, state):
        return StateVector(state)
        
    def bra(self) -> np.array:
        """Returns a bra (i.e. a row vector, conjugate transpose) representation of the qubit.
        """
        return self._state.conj()
    
    def ket(self) -> np.array:
        """Returns a ket (i.e. a column vector) representation of the qubit.
        """
        return self._state[..., None]
        
    def to_density_matrix(self):
        """Converts the state vector into a density matrix by taking the outer product."""
        return self.ket() * self.bra()
        
    def _get_measurement_stats(self) -> Tuple[list[int], list['StateVector']]:
        """
        Measures the qubit in the computational basis.
        Returns a list of collapsed states and the probability associated with them.
        """
        zero_state = StateVector([1, 0])
        one_state = StateVector([0, 1])
        zero_proj = zero_state.ket() * zero_state.bra()
        one_proj = one_state.ket() * one_state.bra()
        zero_prob = self.bra() @ zero_proj @ self.ket()
        one_prob = self.bra() @ one_proj @ self.ket()
        return ([zero_state, one_state], [zero_prob.item(), one_prob.item()]) 
    
    def _get_measurement_stats(self, projectors: list[np.array] = []):
        """
        Measures the qubit in the computational basis.
        Returns a list of collapsed states and the probability associated with them.
        """
        if projectors == []:
            zero_state = np.array([[1, 0], [0, 0]])
            one_state = np.array([[0, 0], [0, 1]])
            projectors = [zero_state, one_state]
        probs = []
        for projector in projectors:
            probs.append(self.bra() @ projector @ self.ket())
        return (projectors, probs)
        