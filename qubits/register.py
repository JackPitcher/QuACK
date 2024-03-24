from collections.abc import Sequence
from qubits.state_vector import StateVector
from qubits.density_matrix import DensityMatrix
from qubits.qubit import Qubit

import numpy as np

class Register(Sequence):
    """A Register class that acts as a container for several qubits. Currently only supports State Vectors."""
    
    qubits: list[Qubit] = []
    is_density_matrix: bool = False
    
    def __init__(self, qubits: list[Qubit] = [], N: int = 0) -> None:
        """Initializes a register, either with a list of qubits or N qubits.
        If N is 0 and the qubit list is empty, just creates an empty register.
        If N > len(qubits), fill the list with ground state qubits until len(qubits) == N"""  
        self.qubits = qubits      
        if N > len(qubits):
            # set all extra qubits to ground state
            self.qubits = qubits + [StateVector([1, 0]) for _ in range(N - len(qubits))]
        
    def add_qubit(self, qubit: Qubit):
        """Adds a qubit in place to the register."""
        self.qubits.append(qubit)
        
    def remove_qubit(self, target: Qubit | int) -> Qubit | None:
        """Removes a qubit from the register.
        Target can be either a StateVector or an integer.
        If target is a StateVector, removes that state vector in place from the exist if it exists.
        If target is an int, removes the state vector at that index in the register, and returns
        the removed state vector."""
        if isinstance(target, int):
            return self.qubits.pop(target)
        self.qubits.remove(target)
        
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
        return self.qubits[index].measure(return_stats)
            
    def reorder(self, new_order: list[int]):
        """Reorders the register according to the new order.

        Args:
            new_order (list[int]): A list of indices to reorder the register.
        """
        self.qubits = [self.qubits[i] for i in new_order]
    
    def array_representation(self, return_ket: bool = True) -> np.array:
        if return_ket:
            return np.array([state.ket() for state in self]).flatten()[..., None]
        return np.array([state.bra() for state in self]).flatten()
    
    def update_register(self, evolved_state: np.array):
        new_bits = evolved_state.reshape(-1, 2)
        for qubit, new_bit in zip(self.qubits, new_bits):
            qubit.update(new_bit)
        
    def __len__(self):
        """How many qubits are in the register."""
        return len(self.qubits)
    
    def __getitem__(self, index: int):
        """Enables foreach loops and indexing into the register."""
        return self.qubits[index]
    
    def __setitem__(self, index: int, qubit: Qubit):
        if isinstance(qubit, DensityMatrix) and not self.is_density_matrix:
            self.set_to_dm()
        elif isinstance(qubit, StateVector) and self.is_density_matrix:
            qubit = DensityMatrix(qubit.to_density_matrix())
        self.qubits[index] = qubit
        
    def set_to_dm(self):
        self.is_density_matrix = True
        for i in range(len(self.qubits)):
            if not isinstance(self.qubits[i], DensityMatrix):
                self.qubits[i] = DensityMatrix(self.qubits[i].to_density_matrix())
    