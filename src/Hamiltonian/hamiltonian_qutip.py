import numpy as np
from qutip.qip.circuit import QubitCircuit, Gate
from qutip import Qobj

from src.Hamiltonian.hamiltonian import Hamiltonian


class QutipHamiltonian(Hamiltonian):
    """
    Abstract class for Hamiltonians using Qiskit.
    """
    def __init__(self, ops: list) -> None:
        super().__init__('qutip', ops)

    def get_measurer(self, qc: QubitCircuit,
                           op: str) -> dict:
        raise NotImplementedError


class SimpleQutipHamiltonian(QutipHamiltonian):
    """
    A simple Hamiltonian implemented in Qiskit.
    """
    def __init__(self):
        super().__init__(["XX", "YY", "ZZ"])

    def get_measurer(self, qc: QubitCircuit,
                           op: str) -> dict:
        if op == "XX":
            # Change basis
            qc.add_gate("SNOT", targets=[0])
            qc.add_gate("SNOT", targets=[1])
            
            # CNOT used to measure the ZZ
            qc.add_gate("CNOT", controls=0, targets=1)

            # Measure
            qc.add_measurement("M", targets=[1], classical_store=0)
        elif op == "YY":
            def sdg():
                mat = np.array([[1., 0.], [0., -1.j]])
                return Qobj(mat, dims=[[2], [2]])
            qc.user_gates = {"SDG": sdg}
            # Change basis
            qc.add_gate("SDG", targets=0)
            qc.add_gate("SDG", targets=1)
            qc.add_gate("SNOT", targets=0)
            qc.add_gate("SNOT", targets=1)

            # CNOT used to measure ZZ
            qc.add_gate("CNOT", controls=0, targets=1)

            # Measure
            qc.add_measurement("M", targets=1, classical_store=0)
        elif op == "ZZ":
            # CNOT used to measure ZZ
            qc.add_gate("CNOT", controls=0, targets=1)

            # Measure
            qc.add_measurement("M", targets=1, classical_store=0)
        else:
            print(f"Warning: Measurement on the {op} basis is not supported.")
            return None
        
        return qc
    
    def get_energy(self, values: dict) -> float:
        energy = (1 + values["ZZ"] - values["XX"] - values["YY"]) / 2
        return energy