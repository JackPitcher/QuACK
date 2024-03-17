import numpy as np
from Experiment.experiment import Experiment
from Optimizer.optimizer import scalar_minimizer
from Hamiltonian.hamiltonian_qiskit import SimpleQiskitHamiltonian
from QuantumCircuitBuilder.QCBQiskit import SimpleQiskitCircuit
from Hamiltonian.hamiltonian_qutip import SimpleQutipHamiltonian
from QuantumCircuitBuilder.QCBQutip import SimpleQutipCircuit

module = 'qutip'

if module == 'qutip':
    hamiltonian = SimpleQutipHamiltonian()
    qcb = SimpleQutipCircuit()
    optimizer = lambda x, y: scalar_minimizer(x, y, step_size=0.01, bs=(0, np.pi))
    experiment = Experiment(hamiltonian, qcb, optimizer, 'qutip')
    experiment.set_param("shots", 16)
    experiment.set_param("solver_iterations", 64)
    
    experiment.run([3.0], verbose=True)

elif module == 'qiskit':
    hamiltonian = SimpleQiskitHamiltonian()
    qcb = SimpleQiskitCircuit()
    #optimizer = lambda x, y: scalar_minimizer(x, y, step_size=0.005)
    optimizer = lambda x, y: scalar_minimizer(x, y, step_size=0.01, bs=(0, np.pi))
    experiment = Experiment(hamiltonian, qcb, optimizer, 'qiskit')
    experiment.set_param("shots", 1024)
    experiment.set_param("solver_iterations", 1024)
    
    experiment.run([0.2], verbose=True)