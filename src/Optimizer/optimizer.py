import numpy as np


def scalar_minimizer(func: callable, theta: list, step_size: int=0.01, bs: tuple|None=None):
    E = func(theta)

    n_theta = [theta[0] - step_size, theta[0] + step_size]
    if bs is not None:
        if bs[0] >= bs[1]:
            raise ValueError("Please provide bounds with bs[0] < bs[1]")
        n_theta = np.clip(n_theta, bs[0], bs[1])
    n_E = [func([n_theta[0]]), func([n_theta[1]])]
    diffs = [n_E[0] - E, n_E[1] - E]

    index = np.argmin(diffs)
    
    if E < n_E[index]:
        return theta
    
    return [n_theta[index]]