from qubits import Register
from gates import CNOT, X, Y, Z, H, Gate, SWAP, CSWAP, CZ
from circuit.operations import MeasurementOp, GateOp
from typing import Union
from numba import njit

class QuantumCircuit:
    
    reg: Register = None
    classical_storage: list[int]
    ops: list[Union[MeasurementOp, GateOp]]
    
    def __init__(self, reg: Register, num_classical_stores: int, ops: list[Union[MeasurementOp, GateOp]]) -> None:
        """Initlalizes the quantum circuit, with a register of qubits to act on, and a dictionary of operations (e.g. measurements or gates).
        TODO: We should make a way to draw the circuit out, similar to how other QI libraries allow you
        to draw the circuit. That will make it easier to debug.

        Args:
            reg (Register): This should be a register of qubits with all the qubits necessary for the circuit to work.
            operations (list): A list of operations of either Gates or Measurements
            This means the circuit should apply the X gate to Qubit 0, then measure it and store the outcome in classical register 0. It then applies the X gate on
            qubits 0 and 1, and then measures both of these and storing them in different classical registers.
        """
        self.reg = reg
        self.classical_storage = [0 for _ in range(num_classical_stores)]
        self.ops = ops
                
    def run(self):
        # TODO: With the CircuitSimulator, this function is unnecessary, so it can be deleted. Leaving it here for now for testing purposes.
        for op in self.ops:
            if isinstance(op, MeasurementOp):
                target = op.target
                classical_store = op.classical_store
                result = self.reg.measure(target)
                self.classical_storage[classical_store] = result
            else:
                gate = op.gate
                self.reg = gate.evolve()
                    