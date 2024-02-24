from state_vector import StateVector

class Register:
    
    qubits: list[StateVector] = []
    target = 0  # which qubit should be acted on
    
    def __init__(self, qubits: list[StateVector]) -> None:
        self.qubits = qubits
    
    def set_target(self, i: int):
        self.target = i
        
    def add_qubit(self, qubit: StateVector):
        self.qubits.append(qubit)
        
    def remove_qubit(self):
        self.qubits.remove(self.target)
        
    def measure(self, return_stats: bool = False):
        return self.qubits[self.target].measure(return_stats)
        