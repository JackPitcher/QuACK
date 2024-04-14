import numpy as np
from experiment.experiment import QuackExperiment, ProbabilityQuackExperiment, QiskitExperiment, QutipExperiment, ProbabilityQutipExperiment, ProbabilityQiskitExperiment
from optimizer.optimizer import GradientDescent, Adam
from hamiltonian.hamiltonian import SimpleQuackHamiltonian
from hamiltonian.hamiltonian_other import SimpleQiskitHamiltonian, SimpleQutipHamiltonian
from circuit_simulator.CircuitSimulator import NumbaSimulator, CUDASimulator, ProbabilitySimulator
import time
from tqdm import tqdm

module = 'prob_qutip'

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
    experiment.set_param("shots", 1e6)
    res = experiment.run([3.1], verbose=True)
    print(f"Result: {res}")
elif module == "prob":
    hamiltonian = SimpleQuackHamiltonian()
    optimizer = Adam()
    optimizer.set_param("Max Iterations", 1e3)
    simulator = ProbabilitySimulator
    experiment = ProbabilityQuackExperiment(hamiltonian, optimizer, simulator, num_qubits=2)
    res = experiment.run([3.1], verbose=False)
    print(f"Result: {res[0]}, Diff={abs(res[0] - np.pi)}")
elif module == "prob_qiskit":
    hamiltonian = SimpleQiskitHamiltonian()
    optimizer = Adam()
    optimizer.set_param("Max Iterations", 1e3)
    experiment = ProbabilityQiskitExperiment(hamiltonian=hamiltonian, optimizer=optimizer, num_qubits=2)
    res = experiment.run([3.1], verbose=False)
    print(f"Result: {res[0]}, Diff={abs(res[0] - np.pi)}")
elif module == "prob_qutip":
    hamiltonian = SimpleQutipHamiltonian()
    optimizer = Adam()
    optimizer.set_param("Max Iterations", 1e3)
    experiment = ProbabilityQutipExperiment(hamiltonian=hamiltonian, optimizer=optimizer, num_qubits=2)
    res = experiment.run([3.1], verbose=False)
    print(f"Result: {res[0]}, Diff={abs(res[0] - np.pi)}")
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
    #optimizer = GradientDescent(schedule=schedule, step_size=step_size)
    optimizer = Adam()
    optimizer.set_param("Max Iterations", 1e4)
    experiment = QiskitExperiment(hamiltonian, optimizer)
    experiment.set_param("shots", 1e5)
    
    res = experiment.run([3.1], verbose=True)
    print(f"Result: {res}")