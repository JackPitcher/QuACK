import numpy as np

class StateVector:
    
    state: np.array
    
    def __init__(self, state: list[2]) -> None:
        if len(state) != 2:
            raise ValueError("Currently only supporting 2 level qubits")
        state = np.array(state)
        normalization = np.sum(state ** 2)
        if normalization != 1:
            # State should always be normalized!
            state /= np.sqrt(normalization)
        self.state = np.array(state)
        
    def bra(self):
        """Returns a bra (i.e. a row vector, conjugate) representation of the qubit.
        """
        return self.state.conj()
    
    def ket(self):
        """Returns a ket (i.e. a column vector) representation of the qubit.

        Returns:
            np.array: Ket representation of the qubit.
        """
        return self.state[..., None]
        
    def measure(self, return_stats: bool = False):
        """
        Measures the qubit in the computational basis.
        Returns a random collapsed state according to the measurement statistics.
        """
        states, probs = self._get_measurement_stats()
        collapsed_state_ind = np.random.choice(len(states), p=probs)
        if return_stats:
            return states[collapsed_state_ind], states, probs
        return states[collapsed_state_ind]
        
    def _get_measurement_stats(self):
        """
        Measures the qubit in the computational basis.
        Returns a list of collapsed states and the probability associated with them.
        """
        zero_state = [0, 1]
        one_state = [1, 0]
        zero_proj = np.outer(zero_state, zero_state)
        one_proj = np.outer(one_state, one_state)
        zero_prob = self.bra() @ zero_proj @ self.ket()
        one_prob = self.bra() @ one_proj @ self.ket()
        return ([zero_state, one_state], [zero_prob.item(), one_prob.item()])
        