import numpy as np
from typing import Union, Tuple

class Qubit:
    """Abstract class representing a generic qubit in the computational basis."""
    
    _state: np.array
    fp_tol: float = 1e-10
    
    def __init__(self, state: np.array) -> None:
        """Initializes the qubit, and normalize it if it's not already normalized."""
        state = np.array(state)
        normalization = self.compute_normalization(state)
        if normalization != 1:
            # State should always be normalized!
            state = state / normalization
        self._state = state
        
    def get_state(self):
        """Returns the numpy array representation of the state."""
        return self._state
    
    def tensor(self, others: list['Qubit']):
        """Returns a tensor product of this qubit with all the qubits in others.
        To consider: Return the qubit representation instead of numpy.

        Args:
            others (list[Qubit]): A list of qubits.

        Returns:
            np.array: the numpy array of the tensor product
        """
        ret_val = self._state
        if not isinstance(others, list):
            others = [others]
        for other in others:
            ret_val = np.kron(ret_val, other.get_state())
        return self._as_qubit(ret_val)
    
    def _as_qubit(self) -> 'Qubit':
        raise NotImplementedError
        
    def __eq__(self, other):
        """Enables equality checking between two different qubits."""
        if isinstance(other, self.__class__):
            return (np.abs(self._state - other._state) < self.fp_tol).all()
        return False
    
    def __ne__(self, other):
        """Enables inequality checking between two different qubits."""
        return not self.__eq__(other)
    
    def measure(self, projectors: list[np.array] = [], return_stats: bool = False) -> Union['Qubit', 
                                                           Tuple['Qubit',  
                                                                 list['Qubit'],
                                                                 list[float]]]:
        """
        Measures the qubit in the computational basis.
        Returns a random collapsed state according to the measurement statistics.
        """
        states, probs = self._get_measurement_stats(projectors)
        probs = [float(p.real) for p in probs]
        collapsed_state_ind = np.random.choice(len(states), p=probs)
        self._state = states[collapsed_state_ind]
        if return_stats:
            return (collapsed_state_ind, states, probs)
        return collapsed_state_ind
        
    def compute_normalization(self, state):
        raise NotImplementedError
    
    def _get_measurement_stats(self, projectors: list[np.array] = []):
        raise NotImplementedError
    