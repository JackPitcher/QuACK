import sys
sys.path.append(r'c:/Users/npmgs/github/QuACK')

from circuit import QuantumCircuit, MeasurementOp, GateOp
from qubits import DensityMatrix, Register
from numba import njit, cuda
import numpy as np
import math


### Register Interactions
def get_projectors(register: Register, index: int) -> tuple[np.array, np.array]:
    index = register.order[index]
    zero_state = np.array([[1, 0], [0, 0]])
    one_state = np.array([[0, 0], [0, 1]])
    zero_proj_lst = [np.eye(2) if i != index else zero_state for i in range(register.num_qubits)]
    zero_proj = zero_proj_lst[0]
    for i in range(1, register.num_qubits):
        zero_proj = np.kron(zero_proj, zero_proj_lst[i])
    return zero_proj


### Matrix Multiplications

@njit
def matmul(A: np.array, B: np.array) -> np.array:
    C = np.zeros(shape=(A.shape[0], B.shape[1]), dtype=np.complex64)
    for row in range(A.shape[0]):
        for col in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[row, col] += A[row, k] * B[k, col]
    return C


@cuda.jit
def cuda_matmul(A: np.array, B: np.array, C: np.array) -> np.array:
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0j
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


class CircuitSimulator:
    circuit: QuantumCircuit
    shots: int
    device: str
    state_result: np.array
    cs_result: np.array

    def __init__(self, circuit: QuantumCircuit, shots: int, device: str) -> None:
        self.circuit = circuit
        self.shots = int(shots)
        self.device = device
        self.state_result = None
        self.cs_result = None

    def circuit_to_numpy(self):
        ops = []
        classical_store = []
        state = self.circuit.reg.state.get_state()

        for op in self.circuit.ops:
            if isinstance(op, GateOp):
                ops.append(op.gate.matrix_rep())
                classical_store.append(-1)
            elif isinstance(op, MeasurementOp):
                ops.append(get_projectors(self.circuit.reg, op.target))
                classical_store.append(op.classical_store)
        
        return np.array(ops, dtype=np.complex64), state, np.array(classical_store, dtype=np.int8)

    def compute_measurements(self, probs):
        result = np.zeros((self.shots,) + (len(self.circuit.classical_storage),), dtype=np.int8)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = np.random.choice([0, 1], p=probs[i, j])
        self.cs_result = result

    def run(self):
        raise NotImplementedError
    

class NumbaSimulator(CircuitSimulator):
    def run(self):
        ops, state, cs = self.circuit_to_numpy()
        cs_size = len(self.circuit.classical_storage)
        state_out = np.zeros((self.shots,) + state.shape, dtype=np.complex64)
        cs_out = np.zeros((self.shots,) + (cs_size,) + (2,), dtype=np.float32)
        self._run(ops, cs, state, self.shots, state_out, cs_out)
        self.state_result = state_out
        self.compute_measurements(cs_out)
        return state_out, cs_out
        

    @staticmethod
    @njit
    def _run(ops: np.array, cs: np.array, state: np.array, shots: int, state_out: np.array, cs_out: np.array):
        for shot in range(shots):
            state_out[shot] = state
            for i in range(len(ops)):
                if cs[i] == -1:  # this is a gate
                    #out[shot] = ops[i] @ out[shot] @ ops[i].conj().T
                    state_out[shot] = matmul(ops[i], state_out[shot])
                    state_out[shot] = matmul(state_out[shot], ops[i].conj().T)
                else:  # this is a measurement
                    zero_prob = float(max(np.trace(matmul(state_out[shot], ops[i])).real, 0.0))
                    one_prob = 1 - zero_prob
                    cs_out[shot, cs[i], 0] = zero_prob
                    cs_out[shot, cs[i], 1] = one_prob
        



class CUDASimulator(CircuitSimulator):
    def run(self):
        ops, state = self.circuit_to_numpy()
        ops = np.array(ops, dtype=np.complex64)
        out = np.zeros((self.shots,) + state.shape, dtype=np.complex64)
        d_out = cuda.to_device(out)
        threadsperblock = 256
        blockspergrid = math.ceil(out.shape[0] / threadsperblock)
        self._run[blockspergrid, threadsperblock](ops, state, self.shots, d_out)
        out = d_out.copy_to_host()
        self.result = out
        

    @staticmethod
    @cuda.jit
    def _run(ops: np.array, state: np.array, shots: int, out: np.array):
        shot = cuda.grid(1)
        out[shot] = state
        for i in range(len(ops)):
            cuda_matmul(ops[i], out[shot], out[shot])
            cuda_matmul(out[shot], ops[i].conj().T, out[shot])


if __name__ == "__main__":
    from qubits.register import Register
    import time
    num_shots = int(1)
    theta = np.pi
    register = Register([], N=2)
    ops = []
    ops.append(GateOp(name="h", register=register, targets=[0]))
    ops.append(GateOp(name="cnot", register=register, targets=[1], controls=[0]))
    ops.append(GateOp(name="rx", register=register, targets=[0], theta=theta))
    op = "XX"
    if op == "XX":
        ops.append(GateOp(name="h", register=register, targets=[0]))
        ops.append(GateOp(name="h", register=register, targets=[1]))
        ops.append(GateOp(name="cnot", register=register, targets=[1], controls=[0]))
    elif op == "ZZ":
        ops.append(GateOp(name="cnot", register=register, targets=[1], controls=[0]))
    ops.append(MeasurementOp(target=1, classical_storage=0))
    gate_circuit = QuantumCircuit(register, 1, ops)
    sim = NumbaSimulator(gate_circuit, num_shots, "")
    #sim = CUDASimulator(gate_circuit, num_shots, "")
    start = time.time()
    sim.run()
    print(f"{num_shots} shots took {time.time() - start}s")
    state_out, cs_out = sim.state_result, sim.cs_result
    """
    def get_counts(res: np.array, target: int) -> float:
        reg = Register([], 2)
        counts = {0: 0, 1: 0}
        for v in res:
            reg.update_state(DensityMatrix(v))
            counts[reg.measure(target)] += 1
        return counts

    target = 1
    counts = get_counts(out, target)
    result = (counts[0] - counts[1]) / num_shots
    print(result)
    theta = np.pi
    register = Register([], N=2)
    start = time.time()
    for _ in range(num_shots):
        ops = []
        ops.append(GateOp(name="h", register=register, targets=[0]))
        ops.append(GateOp(name="cnot", register=register, targets=[1], controls=[0]))
        ops.append(GateOp(name="rx", register=register, targets=[0], theta=theta))
        ops.append(GateOp(name="h", register=register, targets=[0]))
        ops.append(GateOp(name="h", register=register, targets=[1]))
        ops.append(GateOp(name="cnot", register=register, targets=[1], controls=[0]))
        ops.append(MeasurementOp(target=1, classical_storage=0))
        circuit = QuantumCircuit(register, 1, ops)
        circuit.run()
    print(f"Regular took {time.time() - start}s")
    """
