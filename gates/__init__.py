import sys
sys.path.append(r'c:/Users/jackp/QuACK/gates')

from gate import Gate
from multi_qubit_gates import MultiQubitGate, SWAP
from controlled_gate import ControlledGate, CNOT, CZ, CSWAP
from single_gate import SingleGate, X, Y, Z, H
