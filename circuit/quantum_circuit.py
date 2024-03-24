from qubits import Register
from gates import CNOT, X, Y, Z, H, Gate, SWAP, CSWAP, CZ
from circuit.operations import MeasurementOp, GateOp
from typing import Union

class QuantumCircuit:
    
    reg: Register = None
    operations: dict = {}
    classical_storage: list[int]
    
    def __init__(self, reg: Register, operations: dict, num_classical_stores: int, 
                 ops: list[Union[MeasurementOp, GateOp]] = []) -> None:
        """Initlalizes the quantum circuit, with a register of qubits to act on, and a dictionary of operations (e.g. measurements or gates).
        TODO: We should make a way to draw the circuit out, similar to how other QI libraries allow you
        to draw the circuit. That will make it easier to debug.

        Args:
            reg (Register): This should be a register of qubits with all the qubits necessary for the circuit to work.
            operations (dict): This is a dictionary in the form
                {"op_name": {"targets": []}}
            For GATES, the target dictionary will just be targets, but for measurements, there is also a classical store bit as well, which indicates where
            the output of the measurement should be stored.
            For example, I could have an operations dict like this:
            {
                "x_0": {"targets": [0]},
                "measurement_0": {"target: 0", classical_store: 0},
                "x_1": {"targets": [0, 1]},
                "measurement_1": {"target: 0", classical_store: 1},
                "measurement_2": {"target: 1", classical_store: 2}
            }
            This means the circuit should apply the X gate to Qubit 0, then measure it and store the outcome in classical register 0. It then applies the X gate on
            qubits 0 and 1, and then measures both of these and storing them in different classical registers.
        """
        self.reg = reg
        self.operations = operations
        self.classical_storage = [0 for _ in range(num_classical_stores)]
        self.ops = ops
    
    def run(self):
        for op in self.operations.keys():
            if "measure" in op:
                target_dict = self.operations[op]
                target = target_dict["target"]
                classical_store = target_dict["classical_store"]
                result = self.reg.measure(target)
                self.classical_storage[classical_store] = result
            else:
                target_dict = self.operations[op]
                targets = target_dict["targets"]
                controls = target_dict["controls"]
                gate = self._create_gate(gate_name=op, targets=targets, controls=controls)
                self.reg = gate.evolve()
                
    def run_with_ops(self):
        for op in self.ops:
            if isinstance(op, MeasurementOp):
                target = op.target
                classical_store = op.classical_store
                result = self.reg.measure(target)
                self.classical_storage[classical_store] = result
            else:
                gate = op.gate
                self.reg = gate.evolve()
                
    def _create_gate(self, gate_name: str, targets: list[int], controls: list[int]) -> Gate:
        if "x_" in gate_name:
            return X(register=self.reg, targets=targets)
        if "h_" in gate_name:
            return H(register=self.reg, targets=targets)
        if "y_" in gate_name:
            return Y(register=self.reg, targets=targets)
        if "cz_" in gate_name:
            return CZ(register=self.reg, targets=targets, controls=controls)
        if "z_" in gate_name:
            return Z(register=self.reg, targets=targets)
        if "cnot_" in gate_name:
            return CNOT(register=self.reg, targets=targets, controls=controls)
        if "swap_" in gate_name:
            return SWAP(register=self.reg, targets=targets)
        if "cswap_" in gate_name:
            return CSWAP(register=self.reg, targets=targets, controls=controls)
        raise ValueError("Gate name is not in the currently supported gates.")
    