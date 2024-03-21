import numpy as np
from hamiltonian.hamiltonian import Hamiltonian


###########################
### Qiskit Hamiltonians ###
###########################
import qiskit as qk
class QiskitHamiltonian(Hamiltonian):
    """
    Abstract class for Hamiltonians using Qiskit.
    """
    def __init__(self, ops: list) -> None:
        super().__init__('qiskit', ops)

    def get_measurer(self, qc: qk.QuantumCircuit,
                           qr: qk.QuantumRegister,
                           cr: qk.ClassicalRegister,
                           op: str) -> dict:
        raise NotImplementedError


class SimpleQiskitHamiltonian(QiskitHamiltonian):
    """
    A simple Hamiltonian implemented in Qiskit.
    """
    def __init__(self):
        super().__init__(["XX", "YY", "ZZ"])

    def get_measurer(self, qc: qk.QuantumCircuit,
                           qr: qk.QuantumRegister,
                           cr: qk.ClassicalRegister,
                           op: str) -> dict:
        if op == "XX":
            # Change basis
            qc.h(qr[0])
            qc.h(qr[1])
            
            # CNOT used to measure the ZZ
            qc.cx(qr[0], qr[1])

            # Measure
            qc.measure(qr[1], cr[0])
        elif op == "YY":
            # Change basis
            qc.sdg(qr[0])
            qc.sdg(qr[1])
            qc.h(qr[0])
            qc.h(qr[1])

            # CNOT used to measure ZZ
            qc.cx(qr[0], qr[1])

            # Measure
            qc.measure(qr[1], cr[0])
        elif op == "ZZ":
            # CNOT used to measure ZZ
            qc.cx(qr[0], qr[1])

            # Measure
            qc.measure(qr[1], cr[0])
        else:
            print(f"Warning: Measurement on the {op} basis is not supported.")
            return None
        
        return qc
    
    def get_energy(self, values: dict) -> float:
        energy = (1 + values["ZZ"] - values["XX"] - values["YY"]) / 2
        return energy


##########################
### QuTiP Hamiltonians ###
##########################
from qutip.qip.circuit import QubitCircuit, Gate
from qutip import Qobj


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