from circuit import QuantumCircuit, MeasurementOp, GateOp
from numba import njit, cuda, prange
import numpy as np
import math
from utils import numba_matmul, cu_left_mul, cu_right_mul, make_copy, cu_right_mul_trace, p_matmul, cu_matmul, cu_conj_T, cu_T


def combine_gates(gates: list) -> np.array:
    """
    Combines each gate in gates by repeatedly computing gates[i+1] @ gates[i].
    For example, with two consecutive gates G_1 and G_2, it should return G_2 G_1.

    === Prerequisites ===
    - gates[i].shape[1] == gates[i+1].shape[0] for all i
    - len(gates) >= 1
    """
    if len(gates) < 1:
        raise ValueError("Please provide a list with at least one gate!")
    result = np.eye(gates[0].shape[0])
    for gate in gates:
        result = gate @ result
    return result


class CircuitSimulator:
    """
    A circuit simulator that gives parallelization functionality for multiple shots.

    === Attributes ===
    - circuit: the circuit to be simulated
    - shots: the number of shots to run for
    - device: the device to run on
    - state_result: the resulting state after applying the circuit
    - cs_result: the resulting classical storage after applying the circuit
    """
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

    def circuit_to_numpy(self, dtype=np.complex64):
        """
        Converts self.circuit to a list of numpy arrays
        to be used in Numba and Numba CUDA kernels.

        === Returns ===
        - ops: a numpy array containing the gates
        - state: a numpy array containing the circuit's register state
        - classical_store: a numpy array matching measurements to their classical storages
        """
        ops = []
        classical_store = []
        state = self.circuit.reg.state.get_state()
        temp_gate_ops = []
        for op in self.circuit.ops:
            if isinstance(op, GateOp):
                temp_gate_ops.append(op.gate.matrix_rep())
            elif isinstance(op, MeasurementOp):
                if temp_gate_ops != []:
                    ops.append(combine_gates(temp_gate_ops))
                    temp_gate_ops = []
                    classical_store.append(-1)
                ops.append(self.circuit.reg.get_zero_projector(op.target, self.circuit.reg.num_qubits))
                classical_store.append(op.classical_store)
        
        return np.array(ops, dtype=dtype), state, np.array(classical_store, dtype=np.int8)
    
    def compute_measurements(self, probs: np.array):
        """
        Computes the classical measurements using the probs array.
        Assigns the results to self.cs_result

        === Parameters ===
        - probs: a numpy array of dimension (shots, num_classical_storage, 2)

        === Prerequisites ===
        - 0 <= probs[i, j, k] <= 1 for all i, j, k
        - type(probs[i, j, k]) == float
        """
        result = np.zeros((self.shots,) + (len(self.circuit.classical_storage),), dtype=np.int8)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = np.random.choice([0, 1], p=probs[i, j])
        self.cs_result = result

    def run(self) -> None:
        """
        Abstract method running the circuit self.shots times.
        
        === Output ===
        This method should not return anything. Instead, it should store
        - resultant states in self.state_result
        - resultant classical storage in self.cs_result
        """
        raise NotImplementedError
    

class NumbaSimulator(CircuitSimulator):
    """
    A circuit simulator using Numba for CPU optimized performance.
    """
    def run(self):
        ops, state, cs = self.circuit_to_numpy()
        
        state_out = np.zeros((self.shots,) + state.shape, dtype=np.complex64)

        cs_size = len(self.circuit.classical_storage)
        cs_out = np.zeros((self.shots,) + (cs_size,) + (2,), dtype=np.float32)

        self._run(ops, cs, state, self.shots, state_out, cs_out)
        self.state_result = state_out
        #self.compute_measurements(cs_out)
        self.cs_result = compute_measurements(cs_out)
        return state_out, cs_out
        

    @staticmethod
    @njit
    def _run(ops: np.array, cs: np.array, state: np.array, shots: int, state_out: np.array, cs_out: np.array):
        """
        Numba kernel for running the circuits.

        === Parameters ===
        - ops: the np.array of gates/measurements in np.array format
        - cs: the np.array mapping measurements to their classical storages
        - state: the register state in np.array format
        - shots: the number of shots to run for
        - state_out: the np.array to store the resultant states in
        - cs_out: the np.array to store the resultant classical storage probabilities in

        === Prerequisites ===
        - state_out.shape == (shots,) + state.shape
        - cs_out.shape == (shots,) + (max(cs),) + (2,)
        """
        for shot in prange(shots):
            state_out[shot] = state
            for i in range(len(ops)):
                if cs[i] == -1:  # this is a gate
                    state_out[shot] = numba_matmul(state_out[shot], ops[i].conj().T)
                    state_out[shot] = numba_matmul(ops[i], state_out[shot])
                else:  # this is a measurement
                    X = numba_matmul(state_out[shot], ops[i])
                    zero_prob = float(max(np.trace(X).real, 0.0))
                    # TODO: Need to check whether this really works - I think we need to set the state measured to be zero/one, 
                    # but it seems to work for now
                    one_prob = 1 - zero_prob
                    cs_out[shot, cs[i], 0] = zero_prob
                    cs_out[shot, cs[i], 1] = one_prob
        
@njit
def sample_shots(trace: np.array, out_idx: int, out: np.array) -> None:
    for shot in prange(trace.shape[0]):
        zero_prob = float(max(trace[shot].real, 0.0))
        out[shot, out_idx, 0] = zero_prob
        out[shot, out_idx, 1] = 1 - zero_prob


@njit
def rand_choice_nb(arr, prob):
    """
    Computes np.random.choice with a specified p=
    Numba doesn't allow the use of the p= argument to random.choice.
    Workaround: https://github.com/numba/numba/issues/2539#issuecomment-507306369

    === Parameters ===
    - arr: array to sample values from
    - prob: probabilities of each sample value
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


@njit
def compute_measurements(probs: np.array) -> np.array:
    out = np.zeros((probs.shape[0], probs.shape[1]), dtype=np.int8)
    for i in prange(out.shape[0]):
        for j in prange(out.shape[1]):
            out[i, j] = rand_choice_nb([0, 1], probs[i, j])
    return out


class CUDASimulator(CircuitSimulator):
    def run(self):
        # Setup variables
        ops, state, cs = self.circuit_to_numpy()
        state_out = np.zeros((self.shots,) + state.shape, dtype=np.complex64)
        cs_size = len(self.circuit.classical_storage)
        cs_out = np.zeros((self.shots,) + (cs_size,) + (2,), dtype=np.float32)

        # Setup CUDA kernel information for matmul
        output_tensor = np.zeros_like(state_out)
        threadsperblock = (8,8,8)
        bpg_x = math.ceil(state_out.shape[0] / threadsperblock[0])
        bpg_y = math.ceil(state_out.shape[1] / threadsperblock[1])
        bpg_z = math.ceil(state_out.shape[2] / threadsperblock[2])
        blockspergrid = (bpg_x, bpg_y, bpg_z)

        # Setup CUDA kernel information for trace
        trace_output = np.zeros(self.shots, dtype=np.complex64)
        trace_threadsperblock = 64
        trace_bpg_x = math.ceil(state_out.shape[0] / trace_threadsperblock)
        trace_blockspergrid = trace_bpg_x

        # Set State
        for i in range(self.shots):
            state_out[i] = state
        
        # Move things to device
        state_out_d = cuda.to_device(state_out)
        output_tensor_d = cuda.to_device(output_tensor)
        trace_output_d = cuda.to_device(trace_output)
        ops = cuda.to_device(ops)
        op_T = cuda.to_device(np.zeros_like(ops[0]))
        # Run circuit
        for i in range(len(ops)):
            if cs[i] == -1:  # this is a gate
                op = ops[i]
                cu_left_mul[blockspergrid, threadsperblock](op, state_out_d, output_tensor_d)
                cu_conj_T[blockspergrid, threadsperblock](op, op_T)
                cu_right_mul[blockspergrid, threadsperblock](op_T, output_tensor_d, state_out_d)
            else:  # this is a measurement
                cu_T[blockspergrid, threadsperblock](ops[i], op_T)
                cu_right_mul_trace[trace_blockspergrid, trace_threadsperblock](op_T, state_out_d, trace_output_d)
                X = trace_output_d.copy_to_host()
                sample_shots(X, cs[i], cs_out)
        self.state_result = state_out
        self.cs_result = compute_measurements(cs_out)
        return state_out, cs_out


class ProbabilitySimulator(CircuitSimulator):
    """
    Simulator that outputs probabilities.

    === Attributes ===
    - method: str indicating which matmul method to use. Can be:
        - numpy: standard @ matmul
        - numba: naive njit matmul
        - parallel: naive njit parallel
        - cuda: gpu enabled
    """
    method: str

    def __init__(self, circuit: QuantumCircuit, shots: int, device: str, method: str) -> None:
        super().__init__(circuit, shots, device)
        self.method = method

    def run(self):
        ops, state, cs = self.circuit_to_numpy(dtype=np.complex128)

        cs_size = len(self.circuit.classical_storage)
        if self.method == "numpy":
            state_out, probs = self._run_numpy(ops, cs, state, cs_size)
        elif self.method == "numba":
            state_out, probs = self._run_numba(ops, cs, state, cs_size)
        elif self.method == "parallel":
            state_out, probs = self._run_numba_parallel(ops, cs, state, cs_size)
        elif self.method == "cuda":
            state_out, probs = self._run_cuda(ops, cs, state, cs_size)
        self.cs_result = probs
        return state_out, probs
    
    @staticmethod
    def _run_numpy(ops: np.array, cs: np.array, state: np.array, cs_size: int):
        probs = np.zeros(cs_size, dtype=np.float64)
        state_out = state.astype(np.complex64)
        for i in range(len(ops)):
            if cs[i] == -1:  # this is a gate
                state_out = ops[i] @ state_out @ ops[i].conj().T
            else:  # this is a measurement
                X = state_out @ ops[i]
                zero_prob = float(max(np.trace(X).real, 0.0))
                # TODO: Need to check whether this really works - I think we need to set the state measured to be zero/one, 
                # but it seems to work for now
                probs[cs[i]] = zero_prob
                
        return state_out, probs
    
    @staticmethod
    @njit
    def _run_numba(ops: np.array, cs: np.array, state: np.array, cs_size: int):
        probs = np.zeros(cs_size, dtype=np.float64)
        state_out = state.astype(np.complex64)
        for i in range(len(ops)):
            if cs[i] == -1:  # this is a gate
                # state_out = ops[i] @ state_out @ ops[i].conj().T
                state_out = numba_matmul(state_out, ops[i].conj().T)
                state_out = numba_matmul(ops[i], state_out)
            else:  # this is a measurement
                X = numba_matmul(state_out, ops[i])
                # X = state_out @ ops[i]
                zero_prob = float(max(np.trace(X).real, 0.0))
                # TODO: Need to check whether this really works - I think we need to set the state measured to be zero/one, 
                # but it seems to work for now
                probs[cs[i]] = zero_prob
                
        return state_out, probs

    @staticmethod
    @njit(parallel=True)
    def _run_numba_parallel(ops: np.array, cs: np.array, state: np.array, cs_size: int):
        probs = np.zeros(cs_size, dtype=np.float64)
        state_out = state.astype(np.complex64)
        for i in range(len(ops)):
            if cs[i] == -1:  # this is a gate
                # state_out = ops[i] @ state_out @ ops[i].conj().T
                state_out = p_matmul(state_out, ops[i].conj().T)
                state_out = p_matmul(ops[i], state_out)
            else:  # this is a measurement
                X = p_matmul(state_out, ops[i])
                # X = state_out @ ops[i]
                zero_prob = float(max(np.trace(X).real, 0.0))
                # TODO: Need to check whether this really works - I think we need to set the state measured to be zero/one, 
                # but it seems to work for now
                probs[cs[i]] = zero_prob
                
        return state_out, probs
    
    @staticmethod
    def _run_cuda(ops: np.array, cs: np.array, state: np.array, cs_size: int):
        probs = np.zeros(cs_size, dtype=np.float64)
        state_out = state.astype(np.complex64)
        tmp_out = np.empty_like(state_out)

        threadsperblock = (16, 16)
        bpg_x = math.ceil(state_out.shape[0] / threadsperblock[0])
        bpg_y = math.ceil(state_out.shape[1] / threadsperblock[1])
        blockspergrid = (bpg_x, bpg_y)

        state_out = cuda.to_device(state_out)
        tmp_out = cuda.to_device(tmp_out)
        ops = cuda.to_device(ops)
        op_conj_T = cuda.to_device(np.zeros_like(ops[0]))

        for i in range(len(ops)):
            if cs[i] == -1:  # this is a gate
                # state_out = ops[i] @ state_out @ ops[i].conj().T
                cu_conj_T[blockspergrid, threadsperblock](ops[i], op_conj_T)
                cu_matmul[blockspergrid, threadsperblock](ops[i], state_out, tmp_out)
                cu_matmul[blockspergrid, threadsperblock](tmp_out, op_conj_T, state_out)
            else:  # this is a measurement
                cu_matmul[blockspergrid, threadsperblock](state_out, ops[i], tmp_out)
                X = tmp_out.copy_to_host()
                # X = state_out @ ops[i]
                zero_prob = float(max(np.trace(X).real, 0.0))
                # TODO: Need to check whether this really works - I think we need to set the state measured to be zero/one, 
                # but it seems to work for now
                probs[cs[i]] = zero_prob
                
        return state_out, probs