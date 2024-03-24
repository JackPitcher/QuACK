from typing import Any
import numpy as np
# Import QuTiP
from qutip.qip.circuit import QubitCircuit, Gate, CircuitSimulator
from qutip import tensor, basis
# Import Qiskit
import qiskit as qk
from qiskit_aer import Aer
# Import QuACK
from hamiltonian.hamiltonian import Hamiltonian
from optimizer.optimizer import Optimizer
from circuit.quantum_circuit import QuantumCircuit

class Experiment:
    """
    Contains all information needed to find the lowest eigenvalue of a Hamiltonian.

    === Attributes ===
    - hamiltonian: the function that returns the energy value of the Hamiltonian
    - ops: the operations needed to measure the hamiltonian's energy
    - optimizer: the optimization method
    - module: the module being used. Can be 'qiskit' or 'qutip'

    === Representation Invariants ===
    - self.module == self.qcb.module
    - self.module == self.hamiltonian.module
    """
    hamiltonian: Hamiltonian
    optimizer: Optimizer
    module: str
    params: dict

    def __init__(self, hamiltonian: Hamiltonian,
                       optimizer: Optimizer,
                       module: str) -> None:
        if module != hamiltonian.module:
            raise TypeError("Please provide consistent module usage.")

        self.hamiltonian = hamiltonian
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
            qc = self.hamiltonian.construct_ansatz(theta=theta, op=op)            
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
                       optimizer: Optimizer) -> None:
        super().__init__(hamiltonian, optimizer, "qiskit")
    
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
                       optimizer: Optimizer) -> None:
        super().__init__(hamiltonian, optimizer, "qutip")
    
    def _get_counts(self, qc) -> dict:
        counts = {0: 0, 1: 0}
        zero_state = tensor(basis(2, 0), basis(2, 0))
        sim = CircuitSimulator(qc)
        for i in range(self.params["shots"]):
            result = sim.run(state=zero_state)
            counts[result.get_cbits(0)[0]] += 1
        
        return counts
    

class QuackExperiment(Experiment):
    def __init__(self, hamiltonian: Hamiltonian,
                 optimizer: Optimizer) -> None:
        super().__init__(hamiltonian, optimizer, "quack")
    
    def _get_counts(self, qc: QuantumCircuit, theta: list, op: str) -> dict:
        counts = {0: 0, 1: 0}
        for _ in range(self.params["shots"]):
            qc = self.hamiltonian.construct_ansatz(theta=theta, op=op)
            qc.run_with_ops()
            result = qc.classical_storage[0]
            if np.array_equal(result, [[1, 0], [0, 0]]):
                counts[0] += 1
            else:
                counts[1] += 1
        return counts
    
    def _step(self, theta: list, verbose=False) -> float:
        vqe_res = {}
        for op in self.hamiltonian.get_ops():
            qc = self.hamiltonian.construct_ansatz(theta=theta, op=op)            
            counts = self._get_counts(qc, theta=theta, op=op)
            vqe_res[op] = (counts[0] - counts[1])/self.params["shots"]
    
        energy = self.hamiltonian.get_energy(vqe_res)

        return energy