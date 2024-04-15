import time
from tqdm import tqdm

def get_single_func_stats(func: callable, num_runs: int) -> dict:
    if not isinstance(num_runs, int):
        raise ValueError("Please give an integer!")
    if num_runs < 1:
        raise ValueError("Please give a positive integer!")
    func()  # warmup the cache
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


def benchmark_single_func(func: callable, func_name: str, num_runs: int, measure: str):
    stats = get_single_func_stats(func, num_runs)
    print(f"Function: {func_name.strip()}, Num Runs: {num_runs}\n" + 
          f"Total time: {'{:.3e}'.format(stats["total"])}s\n" +
          f"Average time: {'{:.3e}'.format(stats["average"])}s\n" +
          f"Max time: {'{:.3e}'.format(stats["max"])}s\n" + 
          f"Min time: {'{:.3e}'.format(stats["min"])}s")
    return stats[measure]
    
    