from collections import Sequence
from state_vector import StateVector

class Register(Sequence):
    
    qubits: list[StateVector] = []
    
    def __init__(self, qubits: list[StateVector] = [], N: int = 0) -> None:        
        if N > 0 and qubits == []:
            # set all qubits to ground state
            qubits = [StateVector([0, 1]) for i in range(N)]
        self.qubits = qubits
        
    def add_qubit(self, qubit: StateVector):
        self.qubits.append(qubit)
        
    def remove_qubit(self, target):
        self.qubits.remove(target)
        
    def measure(self, target, return_stats: bool = False):
        return self.qubits[target].measure(return_stats)
    
    def reorder(self, new_order: list[int]):
        self.qubits = [self.qubits[i] for i in new_order]
        
    def __len__(self):
        return len(self.qubits)
    
    def __getitem__(self, index):
        return self.qubits[index]
    