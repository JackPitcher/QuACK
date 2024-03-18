from typing import Any

class Hamiltonian:
    """
    Abstract class for representing Hamiltonians and their measurements.
    
    === Attributes ===
    module: a string representing which module this Hamiltonian uses. Can be a 'qiskit' or 'qutip'
    ops: the measurement operations needed to evaluate this Hamiltonian.
    """
    module: str
    ops: list

    def __init__(self, module: str, ops: list) -> None:
        self.module = module
        self.ops = ops

    def get_ops(self) -> list:
        return self.ops.copy()

    def get_energy(self, values: dict) -> float:
        raise NotImplementedError