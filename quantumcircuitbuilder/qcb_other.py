from quantumcircuitbuilder.qcb import QuantumCircuitBuilder


##################
### Qiskit QCB ###
##################
import qiskit as qk


class QiskitQuantumCircuitBuilder(QuantumCircuitBuilder):
    """
    Abstract class for QCBs using the Qiskit module.
    """
    def __init__(self):
        super().__init__("qiskit")


class SimpleQiskitCircuit(QiskitQuantumCircuitBuilder):
    def ansatz(self, theta: list) -> tuple:
        """
        Builds an ansatz based on this circuit.
        """
        theta = theta[0]
        qr = qk.QuantumRegister(2, "qr")
        cr = qk.ClassicalRegister(1, "cr")
        qc = qk.QuantumCircuit(qr, cr)

        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.rx(theta, qr[0])

        return qc, qr, cr


#################
### QuTiP QCB ###
#################
from qutip.qip.circuit import QubitCircuit, Gate


class QutipQuantumCircuitBuilder(QuantumCircuitBuilder):
    """
    An abstract class for QCBs using the QuTiP module.
    """
    def __init__(self):
        super().__init__("qutip")


class SimpleQutipCircuit(QutipQuantumCircuitBuilder):
    def ansatz(self, theta: list) -> tuple:
        """
        Builds an ansatz based on this circuit.
        """
        theta = theta[0]

        qc = QubitCircuit(N=2, num_cbits=1)
        qc.add_gate("SNOT", targets=[0])
        qc.add_gate("CNOT", controls=0, targets=1)
        qc.add_gate("RX", targets=[0], arg_value=theta)

        return qc