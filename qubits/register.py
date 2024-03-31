from collections.abc import Sequence
from qubits.state_vector import StateVector
from qubits.density_matrix import DensityMatrix
from qubits.qubit import Qubit

import numpy as np

class Register:
    """A Register class that acts as a container for several qubits."""
    
    order: list[int]
    state: DensityMatrix
    num_qubits: int
    
    def __init__(self, qubits: list[Qubit] = [], N: int = 0) -> None:
        """Initializes a register, either with a list of qubits or N qubits.
        If N is 0 and the qubit list is empty, just creates an empty register.
        If N > len(qubits), fill the list with ground state qubits until len(qubits) == N"""
        for i, qubit in enumerate(qubits):
            if not isinstance(qubit, DensityMatrix):
                qubits[i] = DensityMatrix(qubit.to_density_matrix())     
        if N > len(qubits):
            # set all extra qubits to ground state
            qubits = qubits + [DensityMatrix([[1, 0], [0, 0]]) for _ in range(N - len(qubits))]
        self.num_qubits = len(qubits)
        self.order = range(len(qubits))
        self.state = self.compute_state(qubits)
        
    def compute_state(self, qubits) -> Qubit:
        if len(qubits) > 1:
            state = qubits[0].tensor(qubits[1:])
        else:
            state = qubits[0]
        return state
        
    def measure(self, index: int, return_stats: bool = False) -> Qubit | tuple:
        """Measures a qubit in the register.

        Args:
            index (int): The index at which to measure
            return_stats (bool, optional): Whether to return the detailed statistics of the measurement;
            that is, the probabilities the state had to collapse to the 0 or 1 state. Defaults to False.

        Returns:
            Either the collapsed state as a StateVector, or the collapsed state along with a list of
            possible states and a list of probabilities of collapsing to those states. 
        """
        index = self.order[index]
        zero_state = np.array([[1, 0], [0, 0]])
        one_state = np.array([[0, 0], [0, 1]])
        zero_proj_lst = [np.eye(2) if i != index else zero_state for i in range(self.num_qubits)]
        one_proj_lst = [np.eye(2) if i != index else one_state for i in range(self.num_qubits)]
        if self.num_qubits > 1:
            zero_proj = zero_proj_lst[0]
            one_proj = one_proj_lst[0]
            for zp, op in zip(zero_proj_lst[1:], one_proj_lst[1:]):
                zero_proj = np.kron(zero_proj, zp)
                one_proj = np.kron(one_proj, op)
        zero_prob = max(np.trace(self.state.get_state() @ zero_proj), 0.0)
        one_prob = max(np.trace(self.state.get_state() @ one_proj), 0.0)
        probs = [float(zero_prob.real), float(one_prob.real)]
        collapsed_state_ind = np.random.choice(len(probs), p=probs)
        if return_stats:
            return (collapsed_state_ind, [zero_state, one_state], probs)
        return collapsed_state_ind
        
    def update_state(self, new_state: DensityMatrix):
        self.state = new_state
        
    def __len__(self):
        """How many qubits are in the register."""
        return self.num_qubits
    