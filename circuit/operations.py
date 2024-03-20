from gates import Gate, X, Y, Z, H, RX, RY, RZ, CNOT, CZ, SWAP, CSWAP
from qubits import Register

class MeasurementOp:
    
    target: int
    classical_store: int
    
    def __init__(self, target: int, classical_storage: int) -> None:
        self.target = target
        self.classical_store = classical_storage
        
class GateOp:
    
    gate: Gate
    targets: list[int]
    controls: list[int]
    reg: Register
    theta: float
    x_names = ["x", "not"]
    y_names = ["y"]
    z_names = ['z']
    h_names = ["h", "hadamard"]
    rx_names = ["rx"]
    ry_names = ["ry"]
    rz_names = ['rz']
    cnot_names = ["cnot", "cx"]
    cz_names = ['cz']
    cswap_names = ['cswap']
    swap_names = ['swap']
    
    def __init__(self, name, register, targets, controls = [], theta = 0.0) -> None:
        self.reg = register
        self.targets = targets
        self.controls = controls
        self.theta = theta
        self.gate = self._create_gate(name)
        
    def _create_gate(self, gate_name: str) -> Gate:
        if gate_name.lower() in self.x_names:
            return X(register=self.reg, targets=self.targets)
        if gate_name.lower() in self.h_names:
            return H(register=self.reg, targets=self.targets)
        if gate_name.lower() in self.y_names:
            return Y(register=self.reg, targets=self.targets)
        if gate_name.lower() in self.cz_names:
            return CZ(register=self.reg, targets=self.targets, controls=self.controls)
        if gate_name.lower() in self.z_names:
            return Z(register=self.reg, targets=self.targets)
        if gate_name.lower() in self.cnot_names:
            return CNOT(register=self.reg, targets=self.targets, controls=self.controls)
        if gate_name.lower() in self.swap_names:
            return SWAP(register=self.reg, targets=self.targets)
        if gate_name.lower() in self.cswap_names:
            return CSWAP(register=self.reg, targets=self.targets, controls=self.controls)
        if gate_name.lower() in self.rx_names:
            return RX(register=self.reg, targets=self.targets, theta=self.theta)
        if gate_name.lower() in self.rz_names:
            return RZ(register=self.reg, targets=self.targets, theta=self.theta)
        if gate_name.lower() in self.ry_names:
            return RY(register=self.reg, targets=self.targets, theta=self.theta)
        raise ValueError("Gate name is not in the currently supported gates.")