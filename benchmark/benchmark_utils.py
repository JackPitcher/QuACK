import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_single_func_stats(func: callable, num_runs: int) -> dict:
    if not isinstance(num_runs, int):
        raise ValueError("Please give an integer!")
    if num_runs < 1:
        raise ValueError("Please give a positive integer!")
    start = time.perf_counter()
    max_time = 0
    min_time = -1
    for i in tqdm(range(num_runs)):
        run_start = time.perf_counter()
        func()
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


def benchmark_single_func(func: callable, func_name: str, num_runs: int):
    stats = get_single_func_stats(func, num_runs)
    print(f"Function: {func_name.strip()}, Num Runs: {num_runs}\n" + 
          f"Total time: {'{:.3e}'.format(stats["total"])}s\n" +
          f"Average time: {'{:.3e}'.format(stats["average"])}s\n" +
          f"Max time: {'{:.3e}'.format(stats["max"])}s\n" + 
          f"Min time: {'{:.3e}'.format(stats["min"])}s")
    return stats["average"]


def benchmark_multi_func(funcs: list[callable], func_names: list[str], num_runs: int):
    s = f"Benchmarking with number of runs: {num_runs}\n"
    s += "Function    | " + "Total (s)  | " + "Average (s) | " + "Max (s)    | " + "Min (s)\n"
    for i in range(len(funcs)):
        stats = get_single_func_stats(funcs[i], num_runs)
        s += f"{func_names[i]}" + " | "
        s += f"{'{:.3e}'.format(stats["total"])}" + "  | "
        s += f"{'{:.3e}'.format(stats["average"])}" + "   | "
        s += f"{'{:.3e}'.format(stats["max"])}" + "  | "
        s += f"{'{:.3e}'.format(stats["min"])}" + "\n"
    print(s)


def plot_total_times(funcs: list[callable], func_names: list[str], num_runs: int, num_shots: list[int]):
    total_times = [[] for f in funcs]
    for i in tqdm(range(num_runs)):
        for j, func in enumerate(funcs):
            start = time.perf_counter()
            func(num_shots[i])
            end = time.perf_counter()
            total_times[j].append(end - start)
    
    fig, ax = plt.subplots(1,1)
    for i in range(len(funcs)):
        ax.plot(num_shots, total_times[i], label=func_names[i])
    ax.set_xlabel("Number of Shots")
    ax.set_ylabel("Total Time")
    legend = ax.legend()
    plt.savefig("output/numba_vs_cuda.pdf", bbox_inches="tight")
    
    