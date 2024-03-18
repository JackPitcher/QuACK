import numpy as np


def scalar_minimizer(func: callable, theta: list, step_size: float=0.01, bs: tuple|None=None):
    """
    Very basic minimizer step for a function func that only takes in one parameter.
    Allows for the use of bounds

    === Parameters ===
    func: the function to minimize.
    theta: the parameter to minimize over.
    step_size: the size of the step to take at each iteration.
    bs: optional bounds to keep theta within.

    === Prerequisites ===
    len(theta) == 1
    bs[0] < bs[1]
    """

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


def diff(dH: np.array, func: callable, theta: np.array, step_size) -> np.array:
    """
    Computes the derivative of a Hamiltonian.
    === Parameters ===
    dH: the numpy array to store the results in
    func: the Hamiltonian being differentiated.
    theta: the parameters we are differentiating on.
    step_size: size of the finite difference step
    """
    for i in range(len(theta)):
        theta[i] += step_size
        a = func(theta)
        theta[i] -= 2*step_size
        dH[i] = a - func(theta)
        theta[i] += step_size
    return dH

def gradient_descent(func: callable, theta: np.array, step_size: float=np.pi/2.0, lr: float=0.001):
    """
    Computes a gradient descent step on func.
    === Parameters ===
    func: the function to take the derivative of.
    theta: the parameters to take the derivative from.
    step_size: size of the finite difference step.
    lr: the learning rate.
    """
    dH = np.empty_like(theta)
    diff(dH, func, theta, step_size)
    return theta - lr * dH