import numpy as np
from experiment.experiment import QiskitExperiment, QutipExperiment
from optimizer.optimizer import GradientDescent
from hamiltonian.hamiltonian_other import SimpleQiskitHamiltonian, SimpleQutipHamiltonian
from quantumcircuitbuilder.qcb_other import SimpleQiskitCircuit, SimpleQutipCircuit

module = 'qutip'

if module == 'qutip':
    hamiltonian = SimpleQutipHamiltonian()
    qcb = SimpleQutipCircuit()
    schedule = [[16, 64, 128], [0.1, 0.05, 0.01]]
    step_size = np.pi/1e2
    optimizer = GradientDescent(schedule=schedule, step_size=step_size)
    experiment = QutipExperiment(hamiltonian, qcb, optimizer)
    experiment.set_param("shots", 16)
    experiment.run([3.1], verbose=True)

elif module == 'qiskit':
    hamiltonian = SimpleQiskitHamiltonian()
    qcb = SimpleQiskitCircuit()
    schedule = [[512, 1024, 2048], [0.1, 0.05, 0.01]]
    step_size = np.pi/1e2
    optimizer = GradientDescent(schedule=schedule, step_size=step_size)
    
    experiment = QiskitExperiment(hamiltonian, qcb, optimizer)
    experiment.set_param("shots", 1024)
    
    experiment.run([3.1], verbose=True)