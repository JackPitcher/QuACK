import utils
from benchmark.benchmark_utils import benchmark_single_func
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math


def evaluate_matrix(func, A, B):
    func(A, B)


def stats_single_matmul(func: callable, size: int, num_runs: int) -> dict[str, float]:
    start = time.perf_counter()
    max_time = 0
    min_time = -1
    for i in tqdm(range(num_runs)):
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        run_start = time.perf_counter()
        evaluate_matrix(func, A, B)
        run_end = time.perf_counter()
        diff = run_end - run_start
        if diff > max_time:
            max_time = diff
        if min_time == -1 or diff < min_time:
            min_time = diff
    end = time.perf_counter()
    total = end - start
    return {
        "total": total,
        "average": total/num_runs,
        "max": max_time,
        "min": min_time
    }


def benchmark_single_matmul(func: callable, func_name: str, size: int, num_runs: int):
    stats = stats_single_matmul(func, size, num_runs)
    print(f"Function: {func_name.strip()}, Num Runs: {num_runs}\n" + 
          f"Total time: {'{:.3e}'.format(stats["total"])}s\n" +
          f"Average time: {'{:.3e}'.format(stats["average"])}s\n" +
          f"Max time: {'{:.3e}'.format(stats["max"])}s\n" + 
          f"Min time: {'{:.3e}'.format(stats["min"])}s")
    return stats["average"]


def benchmark_plot(sizes: list[int], num_runs: int, measure: str="average", cuda: bool=False):
    numpy_results = []
    numba_results = []
    preco_results = []
    if cuda:
        cuda_results = []
    for size in sizes:
        print(f"Benchmarking size 2^{size}")
        size = 2**size
        numpy = lambda x, y: x @ y
        numpy_results.append(stats_single_matmul(numpy, size, num_runs)[measure])
        #numba_results.append(stats_single_matmul(utils.numba_matmul, size, num_runs)[measure])
        if size < 12:
            preco = utils.generate_matmul(size)
            preco_results.append(stats_single_matmul(preco, size, num_runs)[measure])
        if cuda:
            threadsperblock = (16, 16)
            bpg_x = math.ceil(size / threadsperblock[0])
            bpg_y = math.ceil(size / threadsperblock[1])
            blockspergrid = (bpg_x, bpg_y)
            c = np.empty((size, size), dtype=np.complex64)
            cuda_func = lambda x, y: utils.cu_matmul[blockspergrid, threadsperblock](x, y, c)
            cuda_results.append(stats_single_matmul(cuda_func, size, num_runs)[measure])
    
    fig, ax = plt.subplots(1,1)
    ax.plot(sizes, numpy_results, '.', label="Numpy")
    #ax.plot(sizes, numba_results, '.', label="Numba")
    #ax.plot(sizes, preco_results, '.', label="Precompiled")
    if cuda:
        ax.plot(sizes, cuda_results, '.', label="CUDA")
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Average (s)")
    ax.legend()
    plt.savefig("output/matmul_benchmarks.pdf")



if __name__ == "__main__":
    sizes = np.arange(1, 14)
    num_runs = 10
    benchmark_plot(sizes, num_runs, cuda=True)