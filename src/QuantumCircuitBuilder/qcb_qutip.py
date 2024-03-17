from qutip.qip.circuit import QubitCircuit, Gate

from src.QuantumCircuitBuilder.qcb import QuantumCircuitBuilder


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