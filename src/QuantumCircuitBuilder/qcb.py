class QuantumCircuitBuilder:
    """
    An abstract class for constructing ansatz.

    === Attributes ===
    module: the module being used. Can be either qiskit or qutip.
    """
    module: str
    
    def __init__(self, module: str) -> None:
        self.module = module

    def module() -> str:
        """
        Returns the module attribute.
        """
        return self.module

    def ansatz(theta: list) -> tuple:
        raise NotImplementedError