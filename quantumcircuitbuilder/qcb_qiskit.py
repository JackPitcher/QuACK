import qiskit as qk

from quantumcircuitbuilder.qcb import QuantumCircuitBuilder


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