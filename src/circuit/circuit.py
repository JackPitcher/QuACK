from ..qubits.register import Register
from ..gates.gate import Gate
from ..gates.cnot import CNOT
from ..gates.x import X

class QuantumCircuit:
    
    reg: Register = None
    operations: dict = {}
    classical_storage: list[int]
    
    def __init__(self, reg: Register, operations: dict) -> None:
        """Initlalizes the quantum circuit, with a register of qubits to act on, and a dictionary of operations (e.g. measurements or gates).

        Args:
            reg (Register): This should be a register of qubits with all the qubits necessary for the circuit to work.
            operations (dict): This is a dictionary in the form
                {"op_name": {"targets": []}}
            For GATES, the target dictionary will just be targets, but for measurements, there is also a classical store bit as well, which indicates where
            the output of the measurement should be stored.
            For example, I could have an operations dict like this:
            {
                "x_0": {"targets": [0]},
                "measurement_0": {"targets: [0]", classical_store: 0},
                "x_1": {"targets": [0, 1]},
                "measurement_1": {"targets: [0]", classical_store: 1},
                "measurement_2": {"targets: [1]", classical_store: 2}
            }
            This means the circuit should apply the X gate to Qubit 0, then measure it and store the outcome in classical register 0. It then applies the X gate on
            qubits 0 and 1, and then measures both of these and storing them in different classical registers.
        """
        self.reg = reg
        self.operations = operations
    
    def run(self):
        for op in self.operations.keys():
            if "measure" in op:
                target_dict = self.operations[op]
                targets = target_dict["targets"]
                classical_store = target_dict["classical_store"]
                result = self.reg.measure(targets)
                self.classical_storage[classical_store] = result
            else:
                target_dict = self.operations[op]
                targets = target_dict["targets"]
                gate = self._create_gate(gate_name=op, targets=targets)
                self.reg.evolve(gate)
                
    def _create_gate(self, gate_name: str, targets) -> Gate:
        if "x_" in gate_name:
            return X(targets)
        if "cnot_" in gate_name:
            return CNOT(targets)
        raise ValueError("Gate name is not in the currently supported gates.")