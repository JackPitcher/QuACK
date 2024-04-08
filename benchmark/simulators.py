import numpy as np
from experiment.experiment import QuackExperiment, QiskitExperiment, QutipExperiment
from optimizer.optimizer import GradientDescent
from hamiltonian.hamiltonian import SimpleQuackHamiltonian
from hamiltonian.hamiltonian_other import SimpleQiskitHamiltonian, SimpleQutipHamiltonian
from circuit_simulator.CircuitSimulator import NumbaSimulator, CUDASimulator
from benchmark.benchmark_utils import benchmark_single_func, benchmark_multi_func, plot_total_times


def quack_benchmark_func(shots: int, steps: int, simulator: str):
    hamiltonian = SimpleQuackHamiltonian()
    schedule = [[steps], [0.1]]
    step_size = np.pi/1e2
    optimizer = GradientDescent(schedule=schedule, step_size=step_size)
    if simulator == 'cuda':
        experiment = QuackExperiment(hamiltonian, optimizer, CUDASimulator)
    elif simulator == 'numba':
        experiment = QuackExperiment(hamiltonian, optimizer, NumbaSimulator)
    experiment.set_param("shots", shots)
    experiment.run([3.1], verbose=False)


def qutip_benchmark_func(shots: int, steps: int):
    hamiltonian = SimpleQutipHamiltonian()
    schedule = [[steps], [0.1]]
    step_size = np.pi/1e2
    optimizer = GradientDescent(schedule=schedule, step_size=step_size)
    experiment = QutipExperiment(hamiltonian, optimizer)
    experiment.set_param("shots", shots)
    experiment.run([3.1], verbose=False)


def qiskit_benchmark_func(shots: int, steps: int):
    hamiltonian = SimpleQiskitHamiltonian()
    schedule = [[steps], [0.1]]
    step_size = np.pi/1e2
    optimizer = GradientDescent(schedule=schedule, step_size=step_size)
    experiment = QiskitExperiment(hamiltonian, optimizer)
    experiment.set_param("shots", shots)
    experiment.run([3.1], verbose=False)


def benchmark_single(func: str, num_runs: int, shots: int, steps: int):
    if func == "quack_numba":
        f = lambda: quack_benchmark_func(shots, steps, "numba")
        func_name = "QuACK Numba"
    elif func == "quack_cuda":
        f = lambda: quack_benchmark_func(shots, steps, "cuda")
        func_name = "QuACK Cuda "
    elif func == "qutip":
        f = lambda: qutip_benchmark_func(shots, steps)
        func_name = "QuTiP      "
    elif func == "qiskit":
        f = lambda: qiskit_benchmark_func(shots, steps)
        func_name = "Qiskit     "
    else:
        raise ValueError("Please provide a valid function string!")
    benchmark_single_func(f, func_name, num_runs)


def compare_quack(num_runs: int, shots: int, steps: int):
    funcs = []
    func_names = []
    funcs.append(lambda: quack_benchmark_func(shots, steps, "numba"))
    func_names.append("QuACK Numba")

    funcs.append(lambda: quack_benchmark_func(shots, steps, "cuda"))
    func_names.append("QuACK Cuda ")

    print(f"Benchmarking QuACK with:\nNum Runs = {num_runs}\nNum Shots = {shots}\nNum Steps = {steps}")
    benchmark_multi_func(funcs, func_names, num_runs)


def plot_compare_quack(num_runs: int, shots: list[int]):
    funcs = []
    func_names = []
    funcs.append(lambda x: quack_benchmark_func(x, 1, "numba"))
    func_names.append("QuACK Numba")

    funcs.append(lambda x: quack_benchmark_func(x, 1, "cuda"))
    func_names.append("QuACK Cuda")

    plot_total_times(funcs, func_names, num_runs, shots)
    

def benchmark_all(num_runs: int, shots: int, steps: int):
    funcs = []
    func_names = []
    funcs.append(lambda: quack_benchmark_func(shots, steps, "numba"))
    func_names.append("QuACK Numba")

    funcs.append(lambda: quack_benchmark_func(shots, steps, "cuda"))
    func_names.append("QuACK Cuda ")

    funcs.append(lambda: qutip_benchmark_func(shots, steps))
    func_names.append("QuTiP      ")

    funcs.append(lambda: qiskit_benchmark_func(shots, steps))
    func_names.append("Qiskit     ")

    print(f"Benchmarking all with:\nNum Runs = {num_runs}\nNum Shots = {shots}\nNum Steps = {steps}")
    benchmark_multi_func(funcs, func_names, num_runs)


if __name__ == "__main__":
    #test_single("quack_cuda", 2, 64, 64)
    #test_all(4, 64, 64)
    compare_quack(4, 1e5, 1)
    #plot_compare_quack(4, [1e2, 1e3, 1e4, 1e5])