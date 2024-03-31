from typing import Any
from circuit import QuantumCircuit
from circuit import GateOp, MeasurementOp
from qubits import Register

class Hamiltonian:
    """
    Abstract class for representing Hamiltonians and their measurements.
    
    === Attributes ===
    module: a string representing which module this Hamiltonian uses. Can be a 'qiskit' or 'qutip'
    ops: the measurement operations needed to evaluate this Hamiltonian.
    """
    module: str
    ops: list

    def __init__(self, module: str, ops: list) -> None:
        self.module = module
        self.ops = ops

    def get_ops(self) -> list:
        return self.ops.copy()
    
    def construct_ansatz(self, op: str|None=None, theta: float|None=None):
        raise NotImplementedError

    def get_energy(self, values: dict) -> float:
        raise NotImplementedError
    

class QuackHamiltonian(Hamiltonian):
    """
    An abstract class for Hamiltonians using the QuACK framework.
    """
    def __init__(self, ops: list) -> None:
        self.module = "quack"
        self.ops = ops


class SimpleQuackHamiltonian(QuackHamiltonian):
    """
    A simple Hamiltonian implemented with QuACK.
    """
    def __init__(self) -> None:
        super().__init__(["XX", "YY", "ZZ"])
    
    def construct_ansatz(self, theta: list|float, op: str | None = None):
        if not isinstance(theta, float):
            theta = theta[0]

        register = Register([], N=2)
        ops = []
        ops.append(GateOp(name="h", register=register, targets=[0]))
        ops.append(GateOp(name="cnot", register=register, targets=[1], controls=[0]))
        ops.append(GateOp(name="rx", register=register, targets=[0], theta=theta))

        if op == "XX":
            ops.append(GateOp(name="h", register=register, targets=[0]))
            ops.append(GateOp(name="h", register=register, targets=[1]))
            ops.append(GateOp(name="cnot", register=register, targets=[1], controls=[0]))
            ops.append(MeasurementOp(target=1, classical_storage=0))
        elif op == "YY":
            ops.append(GateOp(name="sdg", register=register, targets=[0]))
            ops.append(GateOp(name="sdg", register=register, targets=[1]))
            ops.append(GateOp(name="h", register=register, targets=[0]))
            ops.append(GateOp(name="h", register=register, targets=[1]))
            ops.append(GateOp(name="cnot", register=register, targets=[1], controls=[0]))
            ops.append(MeasurementOp(target=1, classical_storage=0))
        elif op == "ZZ":
            ops.append(GateOp(name="cnot", register=register, targets=[1], controls=[0]))
            ops.append(MeasurementOp(target=1, classical_storage=0))
        elif op is not None:
            raise ValueError(f"Warning: Measurement on the {op} basis is not supported.")
        
        circuit = QuantumCircuit(reg=register, num_classical_stores=1, ops=ops)
        return circuit
    
    def get_energy(self, values: dict) -> float:
        energy = (1 + values["ZZ"] - values["XX"] - values["YY"]) / 2
        return energy