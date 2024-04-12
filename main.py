import numpy as np
from experiment.experiment import QuackExperiment, QiskitExperiment, QutipExperiment
from optimizer.optimizer import GradientDescent, Adam
from hamiltonian.hamiltonian import SimpleQuackHamiltonian
from hamiltonian.hamiltonian_other import SimpleQiskitHamiltonian, SimpleQutipHamiltonian
from circuit_simulator.CircuitSimulator import NumbaSimulator, CUDASimulator
import time
from tqdm import tqdm

module = 'quack'
if module == "quack":
    opt_method = "adam"
    sim_method = "cpu"
    hamiltonian = SimpleQuackHamiltonian()
    if opt_method == "gd":
        schedule = [[128, 256, 512], [0.1, 0.05, 0.01]]
        step_size = np.pi/1e2
        optimizer = GradientDescent(schedule=schedule, step_size=step_size)
    elif opt_method == "adam":
        optimizer = Adam()
    if sim_method == 'gpu':
        experiment = QuackExperiment(hamiltonian, optimizer, CUDASimulator)
    elif sim_method == 'cpu':
        experiment = QuackExperiment(hamiltonian, optimizer, NumbaSimulator)
    experiment.set_param("shots", 64)
    res = experiment.run([3.1], verbose=False)
    print(f"Result: {res}")
elif module == 'qutip':
    hamiltonian = SimpleQutipHamiltonian()
    schedule = [[128, 256, 512], [0.1, 0.05, 0.01]]
    step_size = np.pi/1e2
    optimizer = GradientDescent(schedule=schedule, step_size=step_size)
    experiment = QutipExperiment(hamiltonian, optimizer)
    experiment.set_param("shots", 128)
    experiment.run([3.1], verbose=False)
elif module == 'qiskit':
    hamiltonian = SimpleQiskitHamiltonian()
    schedule = [[128, 256, 512], [0.1, 0.05, 0.01]]
    step_size = np.pi/1e2
    optimizer = GradientDescent(schedule=schedule, step_size=step_size)
    
    experiment = QiskitExperiment(hamiltonian, optimizer)
    experiment.set_param("shots", 128)
    
    experiment.run([3.1], verbose=False)