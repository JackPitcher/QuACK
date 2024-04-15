"""
Run this file to collect all benchmark results for this code.
"""
from benchmark.experiments import shots_single, probs_single
import matplotlib.pyplot as plt
import json


def benchmark_fixed_qubits_variable_shots(qubits: int, shots: list[int], num_runs: int) -> None:
    """
    Benchmark by fixing the number of qubits and varying the number of shots.

    === Parameters ===
    - qubits: the fixed number of qubits
    - shots: the different number of shots to run for.
    - num_runs: the number of runs to experiment with
    """
    qiskit_results = []
    #qutip_results = []
    numba_results = []
    cuda_results = []
    measure = "average"

    for shot in shots:
        print(f"NUMBER OF SHOTS: {shot}")
        shot = 2**shot
        qiskit_results.append(shots_single("qiskit", num_runs, shot, qubits, 10, measure))
        #qutip_results.append(shots_single("qutip", num_runs, shot, qubits, 1, measure))
        numba_results.append(shots_single("quack_numba", num_runs, shot, qubits, 10, measure))
        cuda_results.append(shots_single("quack_cuda", num_runs, shot, qubits, 10, measure))
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(shots, qiskit_results, '.', label="Qiskit")
    #ax.plot(shots, qutip_results, '.', label="Qutip")
    ax.plot(shots, numba_results, '.', label="QuACK CPU")
    ax.plot(shots, cuda_results, '.', label="QuACK GPU")
    ax.set_xlabel("$\\log_2(\\text{shots})$")
    ax.set_ylabel("Average (s)")
    ax.set_title(f"Varying Shots, Fixed Qubits = {qubits}")
    ax.legend()
    plt.savefig("output/plot_fixed_qubits_varying_shots.jpg")

def benchmark_fixed_shots_variable_qubtis(shots: int, qubits: list[int], num_runs: int) -> None:
    qiskit_results = []
    cpu_results = []
    gpu_results = []
    measure = "average"

    for qubit in qubits:
        print(f"QUBITS: {qubit}")
        qiskit_results.append(shots_single("qiskit", num_runs, shots, qubit, 1, measure))
        cpu_results.append(shots_single("quack_numba", num_runs, shots, qubit, 1, measure))
        gpu_results.append(shots_single("quack_cuda", num_runs, shots, qubit, 1, measure))
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(qubits, qiskit_results, '.', label="Qiskit")
    #ax.plot(qubits, qutip_results, '.', label="Qutip")
    ax.plot(qubits, cpu_results, '.', label="QuACK CPU")
    ax.plot(qubits, gpu_results, '.', label="QuACK GPU")
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Average (s)")
    ax.set_title(f"Fixed Shots={shots}, Varying Qubits")
    ax.legend()
    plt.savefig("output/plot_fixed_shots_varying_qubits.jpg")

def benchmark_probs_variable_qubits(qubits: list[int], num_runs: int) -> None:
    qiskit_results = []
    qutip_results = []
    numpy_results = []
    cuda_results = []
    measure = "average"

    for qubit in qubits:
        print(f"QUBITS: {qubit}")
        qiskit_results.append(probs_single("qiskit", num_runs, qubit, 100, measure))
        qutip_results.append(probs_single("qutip", num_runs, qubit, 100, measure))
        numpy_results.append(probs_single("quack", num_runs, qubit, 100, measure, "numpy"))
        cuda_results.append(probs_single("quack", num_runs, qubit, 100, measure, "cuda"))
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(qubits, qiskit_results, '.', label="Qiskit")
    ax.plot(qubits, qutip_results, '.', label="QuTiP")
    ax.plot(qubits, numpy_results, '.', label="QuACK Numpy")
    ax.plot(qubits, cuda_results, '.', label="QuACK CUDA")
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Average (s)")
    ax.set_title(f"Varying Qubits, Different MatMul Methods")
    ax.legend()
    plt.savefig("output/plot_probs_varying_qubits.jpg")


def benchmark_quack_vary_matmuls_vary_qubits(qubits: list[int], num_runs: int) -> None:
    numpy_results = []
    #numba_results = []
    preco_results = []
    cuda_results = []
    measure = "average"

    for qubit in qubits:
        print(f"QUBITS: {qubit}")
        numpy_results.append(probs_single("quack", num_runs, qubit, 1, measure, "numpy"))
        #numba_results.append(probs_single("quack", num_runs, qubit, 1, measure, "numba"))
        preco_results.append(probs_single("quack", num_runs, qubit, 1, measure, "parallel"))
        cuda_results.append(probs_single("quack", num_runs, qubit, 1, measure, "cuda"))

    fig, ax = plt.subplots(1, 1)
    ax.plot(qubits, numpy_results, '.', label="Numpy")
    #ax.plot(qubits, numba_results, '.', label="Numba")
    ax.plot(qubits, preco_results, '.', label="Pre-compiled")
    ax.plot(qubits, cuda_results, '.', label="CUDA")
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Average (s)")
    ax.set_title(f"Varying Qubits, Different MatMul Methods")
    ax.legend()
    plt.savefig("output/plot_varying_matmul.jpg")


if __name__ == "__main__":
    benchmark_fixed_qubits_variable_shots(4, [12, 13, 14, 15], 1)
    #benchmark_fixed_shots_variable_qubtis(256, [2, 3, 4, 5, 6, 7], 10)
    #benchmark_probs_variable_qubits([i for i in range(2, 9)], 1)
    #benchmark_quack_vary_matmuls_vary_qubits([2, 3, 4, 5, 6, 7, 8, 9, 10], 10)

"""
results = {}

results["Quack Numba"] = {}
results["Quack Numba"]["1e3"] = benchmark_single("quack_numba", 10, 1e3)
results["Quack Numba"]["1e4"] = benchmark_single("quack_numba", 10, 1e4)
results["Quack Numba"]["1e5"] = benchmark_single("quack_numba", 10, 1e5)

results["Quack CUDA"] = {}
results["Quack CUDA"]["1e3"] = benchmark_single("quack_cuda", 10, 1e3)
results["Quack CUDA"]["1e4"] = benchmark_single("quack_cuda", 10, 1e4)
results["Quack CUDA"]["1e5"] = benchmark_single("quack_cuda", 10, 1e5)

results["Qiskit"] = {}
results["Qiskit"]["1e3"] = benchmark_single("qiskit", 10, 1e3)
results["Qiskit"]["1e4"] = benchmark_single("qiskit", 10, 1e4)
results["Qiskit"]["1e5"] = benchmark_single("qiskit", 10, 1e5)


with open("benchmark/results.json", "w") as fp:
    json.dump(results, fp, indent=4, sort_keys=True)
"""