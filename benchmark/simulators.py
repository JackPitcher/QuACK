import numpy as np
from experiment.experiment import QuackExperiment, QiskitExperiment, QutipExperiment
from optimizer.optimizer import Adam
from hamiltonian.hamiltonian import SimpleQuackHamiltonian
from hamiltonian.hamiltonian_other import SimpleQiskitHamiltonian, SimpleQutipHamiltonian
from circuit_simulator.CircuitSimulator import NumbaSimulator, CUDASimulator
from benchmark.benchmark_utils import benchmark_single_func, benchmark_multi_func, plot_total_times


def quack_benchmark_func(shots: int, simulator: str):
    hamiltonian = SimpleQuackHamiltonian()
    optimizer = Adam()
    if simulator == 'cuda':
        experiment = QuackExperiment(hamiltonian, optimizer, CUDASimulator)
    elif simulator == 'numba':
        experiment = QuackExperiment(hamiltonian, optimizer, NumbaSimulator)
    experiment.set_param("shots", shots)
    qc = hamiltonian.construct_ansatz([3.1], "XX")
    experiment._get_counts(qc)


def qiskit_benchmark_func(shots: int):
    hamiltonian = SimpleQiskitHamiltonian()
    optimizer = Adam()
    experiment = QiskitExperiment(hamiltonian, optimizer)
    experiment.set_param("shots", shots)
    qc = hamiltonian.construct_ansatz([3.1], "XX")
    experiment._get_counts(qc)


def benchmark_single(func: str, num_runs: int, shots: int):
    if func == "quack_numba":
        f = lambda: quack_benchmark_func(shots, "numba")
        func_name = "QuACK Numba"
    elif func == "quack_cuda":
        f = lambda: quack_benchmark_func(shots, "cuda")
        func_name = "QuACK Cuda "
    elif func == "qiskit":
        f = lambda: qiskit_benchmark_func(shots)
        func_name = "Qiskit     "
    else:
        raise ValueError("Please provide a valid function string!")
    return benchmark_single_func(f, func_name, num_runs)