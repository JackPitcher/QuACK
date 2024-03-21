from typing import Any
import numpy as np
# Import QuTiP
from qutip.qip.circuit import QubitCircuit, Gate, CircuitSimulator
from qutip import tensor, basis
# Import Qiskit
import qiskit as qk
from qiskit_aer import Aer
# Import QuACK
from quantumcircuitbuilder.qcb import QuantumCircuitBuilder
from hamiltonian.hamiltonian import Hamiltonian
from optimizer.optimizer import Optimizer

class Experiment:
    """
    Contains all information needed to find the lowest eigenvalue of a Hamiltonian.

    === Attributes ===
    - hamiltonian: the function that returns the energy value of the Hamiltonian
    - ops: the operations needed to measure the hamiltonian's energy
    - qcb: the QuantumCircuitBuilder that builds the circuit we will use to optimize the Hamiltonian
    - optimizer: the optimization method
    - module: the module being used. Can be 'qiskit' or 'qutip'

    === Representation Invariants ===
    - self.module == self.qcb.module
    - self.module == self.hamiltonian.module
    """
    hamiltonian: callable
    qcb: QuantumCircuitBuilder
    optimizer: callable
    module: str
    params: dict

    def __init__(self, hamiltonian: callable,
                       qcb: QuantumCircuitBuilder,
                       optimizer: Optimizer,
                       module: str) -> None:
        if module != hamiltonian.module \
            or module != qcb.module:
            raise TypeError("Please provide consistent module usage.")

        self.hamiltonian = hamiltonian
        self.qcb = qcb
        optimizer.set_func(self._step)
        self.optimizer = optimizer
        self.module = module
        self.params = {
            "shots": 64
        }

    def set_param(self, param: str, val: Any) -> None:
        self.params[param] = val

    def _get_counts(self, qc) -> dict:
        raise NotImplementedError

    def _step(self, theta: list, verbose=False) -> float:
        vqe_res = {}
        for op in self.hamiltonian.get_ops():
            if self.module == "qutip":
                qc = self.qcb.ansatz(theta)
                qc = self.hamiltonian.get_measurer(qc, op)
            elif self.module == "qiskit":
                qc, qr, cr = self.qcb.ansatz(theta)
                qc = self.hamiltonian.get_measurer(qc, qr, cr, op)
            
            counts = self._get_counts(qc)
            vqe_res[op] = (counts[0] - counts[1])/self.params["shots"]
    
        energy = self.hamiltonian.get_energy(vqe_res)

        return energy

    def run(self, guess: np.array, verbose=False) -> list:
        self.optimizer.set_theta(guess)
        result = self.optimizer.run(verbose=verbose)
        
        return result


class QiskitExperiment(Experiment):
    def __init__(self, hamiltonian: Hamiltonian,
                       qcb: QuantumCircuitBuilder,
                       optimizer: callable) -> None:
        super().__init__(hamiltonian, qcb, optimizer, "qiskit")
    
    def _get_counts(self, qc) -> dict:
        counts = {0: 0, 1: 0}
        sim_bknd = Aer.get_backend('qasm_simulator')
        temp = sim_bknd.run(qc, shots=self.params["shots"]).result().get_counts()
        if '0' in temp:
            counts[0] = temp['0']
        if '1' in temp:
            counts[1] = temp['1']
        
        return counts


class QutipExperiment(Experiment):
    def __init__(self, hamiltonian: Hamiltonian,
                       qcb: QuantumCircuitBuilder,
                       optimizer: callable) -> None:
        super().__init__(hamiltonian, qcb, optimizer, "qutip")
    
    def _get_counts(self, qc) -> dict:
        counts = {0: 0, 1: 0}
        zero_state = tensor(basis(2, 0), basis(2, 0))
        sim = CircuitSimulator(qc)
        for i in range(self.params["shots"]):
            result = sim.run(state=zero_state)
            counts[result.get_cbits(0)[0]] += 1
        
        return counts