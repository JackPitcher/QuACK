"""
Run this file to collect all benchmark results for this code.
"""
from benchmark.simulators import benchmark_single
import json


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