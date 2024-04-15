import numpy as np
from experiment.experiment import QuackExperiment, QiskitExperiment, QutipExperiment, ProbabilityQuackExperiment, ProbabilityQiskitExperiment, ProbabilityQutipExperiment
from optimizer.optimizer import Adam
from hamiltonian.hamiltonian import SimpleQuackHamiltonian
from hamiltonian.hamiltonian_other import SimpleQiskitHamiltonian, SimpleQutipHamiltonian
from circuit_simulator.CircuitSimulator import NumbaSimulator, CUDASimulator, ProbabilitySimulator
from benchmark.benchmark_utils import benchmark_single_func


def quack_benchmark_func(shots: int, qubits: int, steps: int, simulator: str):
    hamiltonian = SimpleQuackHamiltonian()
    optimizer = Adam()
    optimizer.set_param("Max Iterations", steps)
    if simulator == 'cuda':
        experiment = QuackExperiment(hamiltonian, optimizer, CUDASimulator, num_qubits=qubits)
    elif simulator == 'numba':
        experiment = QuackExperiment(hamiltonian, optimizer, NumbaSimulator, num_qubits=qubits)
    experiment.set_param("shots", shots)
    experiment.run([3.1], verbose=False)


def qutip_benchmark_func(shots: int, qubits: int, steps: int):
    hamiltonian = SimpleQutipHamiltonian()
    optimizer = Adam()
    optimizer.set_param("Max Iterations", steps)
    experiment = QutipExperiment(hamiltonian, optimizer, num_qubits=qubits)
    experiment.set_param("shots", shots)
    experiment.run([3.1], verbose=False)


def qiskit_benchmark_func(shots: int, qubits: int, steps: int):
    hamiltonian = SimpleQiskitHamiltonian()
    optimizer = Adam()
    optimizer.set_param("Max Iterations", steps)
    experiment = QiskitExperiment(hamiltonian, optimizer, num_qubits=qubits)
    experiment.set_param("shots", shots)
    experiment.run([3.1], verbose=False)


def shots_single(func: str, num_runs: int, shots: int, qubits: int, steps: int, measure: str):
    if func == "quack_numba":
        f = lambda: quack_benchmark_func(shots, qubits, steps, "numba")
        func_name = "QuACK Numba"
    elif func == "quack_cuda":
        f = lambda: quack_benchmark_func(shots, qubits, steps, "cuda")
        func_name = "QuACK Cuda"
    elif func == "qutip":
        f = lambda: qutip_benchmark_func(shots, qubits, steps)
        func_name = "QuTiP"
    elif func == "qiskit":
        f = lambda: qiskit_benchmark_func(shots, qubits, steps)
        func_name = "Qiskit"
    else:
        raise ValueError("Please provide a valid function string!")
    return benchmark_single_func(f, func_name, num_runs, measure=measure)


###############################
### PROBABILITY EXPERIMENTS ###
###############################
def probs_quack_benchmark_func(qubits: int, steps: int, matmul_method: str):
    hamiltonian = SimpleQuackHamiltonian()
    optimizer = Adam()
    optimizer.set_param("Max Iterations", steps)
    experiment = ProbabilityQuackExperiment(hamiltonian, optimizer, ProbabilitySimulator, num_qubits=qubits, matmul_method=matmul_method)
    experiment.run([3.1], verbose=False)


def probs_qutip_benchmark_func(qubits: int, steps: int):
    hamiltonian = SimpleQutipHamiltonian()
    optimizer = Adam()
    optimizer.set_param("Max Iterations", steps)
    experiment = ProbabilityQutipExperiment(hamiltonian, optimizer, num_qubits=qubits)
    experiment.run([3.1], verbose=False)


def probs_qiskit_benchmark_func(qubits: int, steps: int):
    hamiltonian = SimpleQiskitHamiltonian()
    optimizer = Adam()
    optimizer.set_param("Max Iterations", steps)
    experiment = ProbabilityQiskitExperiment(hamiltonian, optimizer, num_qubits=qubits)
    experiment.run([3.1], verbose=False)


def probs_single(func: str, num_runs: int, qubits: int, steps: int, measure: str, matmul_method: str=None):
    if func == "qiskit":
        f = lambda: probs_qiskit_benchmark_func(qubits=qubits, steps=steps)
        func_name = "Qiskit"
    elif func == "qutip":
        f = lambda: probs_qutip_benchmark_func(qubits=qubits, steps=steps)
        func_name = "QuTiP"
    elif func == "quack":
        f = lambda: probs_quack_benchmark_func(qubits=qubits, steps=steps, matmul_method=matmul_method)
        func_name = "QuACK"
    return benchmark_single_func(f, func_name, num_runs, measure=measure)