from typing import Any
import numpy as np
# Import QuTiP
from qutip.qip.circuit import QubitCircuit, Gate, CircuitSimulator
from qutip import tensor, basis
# Import Qiskit
import qiskit as qk
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer
# Import QuACK
from hamiltonian.hamiltonian import Hamiltonian
from optimizer.optimizer import Optimizer
from circuit.quantum_circuit import QuantumCircuit
from circuit_simulator import NumbaSimulator

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
                       module: str,
                       num_qubits: int) -> None:
        if module != hamiltonian.module:
            raise TypeError("Please provide consistent module usage.")

        self.hamiltonian = hamiltonian
        optimizer.set_func(self._step)
        self.optimizer = optimizer
        self.module = module
        self.params = {
            "shots": 64
        }
        self.num_qubits = num_qubits

    def set_param(self, param: str, val: Any) -> None:
        self.params[param] = val

    def _get_counts(self, qc) -> dict:
        raise NotImplementedError

    def _step(self, theta: list, verbose=False) -> float:
        vqe_res = {}
        for op in self.hamiltonian.get_ops():
            qc = self.hamiltonian.construct_ansatz(theta=theta, op=op, num_qubits=self.num_qubits)            
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
                       optimizer: Optimizer,
                       num_qubits: int=2) -> None:
        super().__init__(hamiltonian, optimizer, "qiskit", num_qubits)
    
    def _get_counts(self, qc) -> dict:
        counts = {0: 0, 1: 0}
        sim_bknd = Aer.get_backend('qasm_simulator')
        temp = sim_bknd.run(qc, shots=self.params["shots"]).result().get_counts()
        if '0' in temp:
            counts[0] = temp['0']
        if '1' in temp:
            counts[1] = temp['1']
        
        return counts
    
class ProbabilityQiskitExperiment(Experiment):
    
    num_qubits: int
    
    def __init__(self, hamiltonian: Hamiltonian, optimizer: Optimizer, num_qubits=2) -> None:
        super().__init__(hamiltonian, optimizer, "qiskit", num_qubits=num_qubits)
        
    def _step(self, theta: list, verbose=False) -> float:
        vqe_res = {}
        for op in self.hamiltonian.get_ops():
            qc = self.hamiltonian.construct_ansatz(theta=theta, op=op, num_qubits=self.num_qubits, measure=False)
            backend = Aer.get_backend('statevector_simulator')

            outputstate = backend.run(qc, shots=1).result().get_statevector()

            probs = Statevector(outputstate).probabilities([1])           
            vqe_res[op] = probs[0] - probs[1]
    
        energy = self.hamiltonian.get_energy(vqe_res)

        return energy

    def run(self, guess: np.array, verbose=False) -> list:
        import time
        start = time.perf_counter()
        self.optimizer.set_theta(guess)
        result = self.optimizer.run(verbose=verbose)
        #print(f"With {self.optimizer.params['Max Iterations']} iterations, {self.num_qubits} qubits, took {time.perf_counter() - start}s")
        return result


class QutipExperiment(Experiment):
    def __init__(self, hamiltonian: Hamiltonian,
                       optimizer: Optimizer,
                       num_qubits: int=2) -> None:
        super().__init__(hamiltonian, optimizer, "qutip", num_qubits=num_qubits)
    
    def _get_counts(self, qc) -> dict:
        counts = {0: 0, 1: 0}
        zero_state = tensor(basis(2, 0), basis(2, 0))
        sim = CircuitSimulator(qc)
        for i in range(self.params["shots"]):
            result = sim.run(state=zero_state)
            counts[result.get_cbits(0)[0]] += 1
        
        return counts
    
class ProbabilityQutipExperiment(Experiment):
    
    num_qubits: int
    
    def __init__(self, hamiltonian: Hamiltonian, optimizer: Optimizer, num_qubits=2) -> None:
        super().__init__(hamiltonian, optimizer, "qutip", num_qubits=num_qubits)
        
    def _step(self, theta: list, verbose=False) -> float:
        vqe_res = {}
        zero_state = tensor([basis(2, 0)] * self.num_qubits)
        for op in self.hamiltonian.get_ops():
            qc = self.hamiltonian.construct_ansatz(theta=theta, op=op, num_qubits=self.num_qubits)
            result = qc.run_statistics(zero_state)
            probs = result.get_probabilities()
            if len(probs) > 1:  
                vqe_res[op] = probs[0] - probs[1]
            else:
                vqe_res[op] = probs[0]
    
        energy = self.hamiltonian.get_energy(vqe_res)

        return energy
    
    def run(self, guess: np.array, verbose=False) -> list:
        import time
        start = time.perf_counter()
        self.optimizer.set_theta(guess)
        result = self.optimizer.run(verbose=verbose)
        #print(f"With {self.optimizer.params['Max Iterations']} iterations, {self.num_qubits} qubits, took {time.perf_counter() - start}s")
        return result
    

class QuackExperiment(Experiment):
    """
    A class for running an experiment in QuACK.

    === Attributes ===
    - simulator: the constructor for which simulator to use.
    """
    simulator: callable

    def __init__(self, hamiltonian: Hamiltonian,
                 optimizer: Optimizer,
                 simulator: callable,
                 num_qubits: int=2) -> None:
        super().__init__(hamiltonian, optimizer, "quack", num_qubits=num_qubits)
        self.simulator = simulator
        
    @staticmethod    
    def get_classical_counts(classical_storage: np.array, index: int):
        result = {0: 0, 1: 0}
        for shot in classical_storage:
            result[shot[index]] += 1
        return result
    
    def _get_counts(self, qc: QuantumCircuit) -> dict:
        sim = self.simulator(qc, self.params["shots"], "")
        sim.run()
        counts = self.get_classical_counts(sim.cs_result, 0)
        return counts
    
    def _step(self, theta: list, verbose=False) -> float:
        vqe_res = {}
        for op in self.hamiltonian.get_ops():
            qc = self.hamiltonian.construct_ansatz(theta=theta, op=op, num_qubits=self.num_qubits)            
            counts = self._get_counts(qc)
            vqe_res[op] = (counts[0] - counts[1])/self.params["shots"]
    
        energy = self.hamiltonian.get_energy(vqe_res)

        return energy
    

class ProbabilityQuackExperiment(Experiment):
    def __init__(self, hamiltonian: Hamiltonian,
                 optimizer: Optimizer,
                 simulator: callable,
                 num_qubits: int = 2,
                 matmul_method: str = "numpy") -> None:
        super().__init__(hamiltonian, optimizer, "quack", num_qubits=num_qubits)
        self.simulator = simulator
        self.matmul_method = matmul_method

    def _step(self, theta: list, verbose=False) -> float:
        vqe_res = {}
        for op in self.hamiltonian.get_ops():
            qc = self.hamiltonian.construct_ansatz(theta=theta, op=op, num_qubits=self.num_qubits)
            sim = self.simulator(qc, 1, "", self.matmul_method)
            sim.run()
            zero_prob = sim.cs_result[0]            
            vqe_res[op] = 2.*zero_prob - 1
    
        energy = self.hamiltonian.get_energy(vqe_res)

        return energy

    def run(self, guess: np.array, verbose=False) -> list:
        import time
        start = time.perf_counter()
        self.optimizer.set_theta(guess)
        result = self.optimizer.run(verbose=verbose)
        #print(f"With {self.optimizer.params['Max Iterations']} iterations, {self.num_qubits} qubits, took {time.perf_counter() - start}s")
        return result