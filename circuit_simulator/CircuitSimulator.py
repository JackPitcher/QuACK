import sys
sys.path.append(r'c:/Users/npmgs/github/QuACK')

from circuit import QuantumCircuit, MeasurementOp, GateOp
from numba import njit, cuda
import numpy as np
import math
from utils import numba_matmul, cuda_matmul

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

    def circuit_to_numpy(self):
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

        for op in self.circuit.ops:
            if isinstance(op, GateOp):
                ops.append(op.gate.matrix_rep())
                classical_store.append(-1)
            elif isinstance(op, MeasurementOp):
                ops.append(self.circuit.reg.get_zero_projector(op.target, self.circuit.reg.num_qubits))
                classical_store.append(op.classical_store)
        
        return np.array(ops, dtype=np.complex64), state, np.array(classical_store, dtype=np.int8)

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
        self.compute_measurements(cs_out)
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
        for shot in range(shots):
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
