import numpy as np
from typing import Optional


class Optimizer:
    """
    Abstract class for defining an optimization routine.
    
    === Attributes ===
    func: the function to be optimized.
    theta: the current parameter value.
    """
    func: Optional[callable]
    theta: Optional[np.array]

    def __init__(self, func: Optional[callable]=None, guess: Optional[np.array]=None):
        self.func = func
        self.theta = guess

    def set_func(self, func: callable) -> None:
        self.func = func

    def set_theta(self, theta: np.array) -> None:
        self.theta = theta

    def step(self):
        raise NotImplementedError
    
    def run(self, verbose: bool=False):
        raise NotImplementedError


class GradientDescent(Optimizer):
    """
    Class for a gradient descent routine.
    
    === Attributes ===
    schedule: the learning rate schedule for this routine.
        - example: [[10, 20], [1, 0.1]]
            - First list is end range of the solver iterations for that step
            - Second list is the learning rate in that range
    step_size: the step size to take when finding the derivative.
    d: the current derivative value for each parameter.

    === Representation Invariants ===
    theta.size == d.size
    """
    schedule: dict
    step_size: float
    d: Optional[np.array]

    def __init__(self, schedule: list, step_size: float, 
                 func: Optional[callable]=None, guess: Optional[np.array]=None):
        super().__init__(func, guess)
        self.schedule = schedule
        self.step_size = step_size
        if guess is None:
            self.d = None
        else:
            self.d = np.empty_like(guess)
    
    def set_theta(self, theta: np.array):
        self.theta = theta
        if self.d is None:
            self.d = np.empty_like(theta)

    def diff(self) -> np.array:
        """
        Computes the derivative of self.func, stored in self.d.
        """
        for i in range(len(self.theta)):
            self.theta[i] += self.step_size
            a = self.func(self.theta)
            self.theta[i] -= 2*self.step_size
            self.d[i] = a - self.func(self.theta)
            self.theta[i] += self.step_size

    def step(self, lr: float):
        """
        Computes a gradient descent step on func.
        === Parameters ===
        lr: the learning rate.
        """
        self.diff()
        self.theta -= lr * self.d
    
    def run(self, verbose: bool=False) -> np.array:
        """
        Runs gradient descent on func, with initial guess guess.
        === Parameters ===
        verbose: boolean value determining whether to show run details.
        """
        if self.func is None:
            raise AttributeError("Please provide a function to optimize!")
        if self.theta is None:
            raise AttributeError("Please provide an initial guess!")

        for i in range(len(self.schedule[0])):
            a = 0 if i == 0 else self.schedule[0][i-1]
            for _ in range(a, self.schedule[0][i]):
                self.step(self.schedule[1][i])
            if verbose:
                print(f"Iteration {self.schedule[0][i]}, theta={self.theta}")
        
        if verbose:
            print(f"Final values: theta={self.theta}, f(theta)={self.func(self.theta)}")
        
        return self.theta.copy()


ADAM_DEFAULT_PARAMS = {
    "Step Size": 0.01,
    "Beta 1": 0.9,
    "Beta 2": 0.999,
    "Epsilon": 10e-8,
    "Diff Step": 1e-1,
    "Max Iterations": 1e3
}

class Adam(Optimizer):
    def __init__(self, func: Optional[callable]=None, guess: Optional[np.array]=None,
                 params: Optional[dict[str, float]]=None):
        super().__init__(func, guess)
        if params is None:
            self.params = ADAM_DEFAULT_PARAMS.copy()
        else:
            self.params = params

    def set_param(self, name: str, val: float) -> None:
        self.params[name] = val

    def diff(self) -> np.array:
        """
        Computes the derivative of self.func, stored in self.d.
        """
        for i in range(len(self.theta)):
            self.theta[i] += self.params["Diff Step"]
            a = self.func(self.theta)
            self.theta[i] -= 2*self.params["Diff Step"]
            self.d[i] = (a - self.func(self.theta)) * 0.5 / self.params["Diff Step"]
            self.theta[i] += self.params["Diff Step"]

    def run(self, verbose: bool=False):
        self.d = np.zeros_like(self.theta)
        m_old = np.zeros_like(self.theta)
        v_old = np.zeros_like(self.theta)
        t = 0
        a = self.params["Step Size"]
        b1 = self.params["Beta 1"]
        b2 = self.params["Beta 2"]
        eps = self.params["Epsilon"]
        max_iter = self.params["Max Iterations"]
        while t < max_iter:
            t += 1
            self.diff()
            mt = b1 * m_old + (1 - b1) * self.d
            vt = b2 * v_old + (1 - b2) * self.d**2
            mt_hat = mt / (1 - b1**t)
            vt_hat = vt / (1 - b2**t)
            self.theta = self.theta - a * mt_hat / (np.sqrt(vt_hat) + eps)
            m_old = mt
            v_old = vt
            if verbose and t % 500 == 0:
                print(f"t = {t}, theta={self.theta}, mth/svth={mt_hat/(np.sqrt(vt_hat)+eps)}")
                print(self.d, self.d**2)
        return self.theta.copy()

