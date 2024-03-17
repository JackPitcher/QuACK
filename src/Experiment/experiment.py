from typing import Any
import numpy as np
from qutip.qip.circuit import QubitCircuit, Gate, CircuitSimulator
from qutip import tensor, basis
import qiskit as qk
from qiskit_aer import Aer

from src.QuantumCircuitBuilder.qcb import QuantumCircuitBuilder
from src.Hamiltonian.hamiltonian import Hamiltonian

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
                       optimizer: callable,
                       module: str) -> None:
        if module != hamiltonian.module \
            or module != qcb.module:
            raise TypeError("Please provide consistent module usage.")

        self.hamiltonian = hamiltonian
        self.qcb = qcb
        self.optimizer = optimizer
        self.module = module
        self.params = {
            "shots": 64,
            "solver_iterations": 64
        }

    def set_param(self, param: str, val: Any) -> None:
        self.params[param] = val

    def _get_counts(self, qc) -> dict:
        counts = {0: 0, 1: 0}
        if self.module == "qutip":
            zero_state = tensor(basis(2, 0), basis(2, 0))
            sim = CircuitSimulator(qc)
            for i in range(self.params["shots"]):
                result = sim.run(state=zero_state)
                counts[result.get_cbits(0)[0]] += 1
            return counts
        elif self.module == "qiskit":
            sim_bknd = Aer.get_backend('qasm_simulator')
            temp = sim_bknd.run(qc, shots=self.params["shots"]).result().get_counts()
            if '0' in temp:
                counts[0] = temp['0']
            if '1' in temp:
                counts[1] = temp['1']
        else:
            raise NotImplementedError(f"Module {self.module} not implemented")
        
        return counts

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

        if verbose: 
            print("Mean values from measurement results:\n", vqe_res)
            print(f"{theta[0]:<10f} {energy:<10f} {vqe_res['XX']:<10f} {vqe_res['YY']:<10f} {vqe_res['ZZ']:<10f}")

        return energy

    def run(self, guess: list, verbose=False) -> list:
        theta = guess
        for i in range(self.params["solver_iterations"]):
            theta = self.optimizer(self._step, theta)
            if verbose:
                print(f"Iteration {i}, theta={theta}")
        if verbose:
            print(theta, self._step(theta))
        return theta