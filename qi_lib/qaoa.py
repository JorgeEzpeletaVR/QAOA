import numpy as np
from qiskit import transpile
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives import StatevectorSampler
from qiskit_ibm_runtime import EstimatorOptions
from qiskit_ibm_runtime import Session
from qiskit_ibm_runtime.estimator import EstimatorV2 as Estimator
from qiskit_ibm_runtime.sampler import SamplerV2 as Sampler
from qiskit_quantuminspire.qi_provider import QIProvider
from scipy.optimize import minimize


def get_random_parameters(num_params):
    """
    Get random initial circuit parameters for optimization
    """
    return 2 * np.pi * np.random.rand(num_params)

def get_backend_and_transpile(qc, backend_name, qubit_priority_list):
    """
    Given a quantum circuit, Quantum Inpsire backend name, and a qubit priority list
    - Transpiles the quantum circuit for the QIT backend, respecting the qubit priority
    - Returns the relevant QI backend, and the transpiled circuit
    """
    provider = QIProvider()
    backend = provider.get_backend(backend_name)

    qc_transpiled = transpile(qc, backend, initial_layout=qubit_priority_list[0:qc.num_qubits])
    return backend, qc_transpiled

def _do_circ_param_minimising(cost_func, x0, max_ansatz, max_hamiltonian, estimator):
    """
    Helper function for repeated code in circuit parameter optimisation function
    """
    return minimize(
        cost_func, x0, args=(max_ansatz, max_hamiltonian, estimator), method="COBYLA"
    )

def minimise_circuit_parameters(cost_func, x0, max_ansatz, max_hamiltonian, *, local=True, backend_name, qubit_priority_list, num_shots):
    """
    Minimise cost function (cost_func) for a given random start parameter (x0) for a given ansatz (max_ansatz) and hamiltonian (max_hamiltonian)
    """
    if local:
        estimator = StatevectorEstimator()
        result = _do_circ_param_minimising(cost_func, x0, max_ansatz, max_hamiltonian, estimator)
    else:
        # Get the backend and transpiled circuit
        backend, max_ansatz_transpiled = get_backend_and_transpile(max_ansatz, backend_name, qubit_priority_list)
        # Make the Hamiltonian matches the transpiled circuit
        #   (some of the QI backends force you to have a specific number of Qubits, so the Hamiltonian has to be
        #    scaled to match)
        max_hamiltonian_mapped = max_hamiltonian.apply_layout(max_ansatz_transpiled.layout)
        with Session(backend=backend) as session:
            estimator = Estimator(mode=session, options=EstimatorOptions(resilience_level=1, default_shots=num_shots))
            result = _do_circ_param_minimising(cost_func, x0, max_ansatz_transpiled, max_hamiltonian_mapped, estimator)
    # We only care about the circuit parameters in the result
    return result.x

def _do_node_groupings(qc, sampler, *, num_shots):
    """
    Helper function for repeated code in circuit param to node grouping function:
        Repeatedly sample the ansatz circuit, with the optimised parameters
    """
    job = sampler.run([qc], shots=num_shots)
    data_pub = job.result()[0].data
    return data_pub.meas.get_counts()

def get_node_groupings_from_circuit_parameters(max_ansatz, min_circ_param, *, local=True, backend_name, qubit_priority_list, num_shots):
    # Apply the optimised circuit parameters to the ansatz circuit
    qc = max_ansatz.assign_parameters(min_circ_param)
    # Add measurements to the circuit
    qc.measure_all()

    if local:
        sampler = StatevectorSampler()
        counts = _do_node_groupings(qc, sampler, num_shots=num_shots)
    else:
        backend, qc_transpiled = get_backend_and_transpile(qc, backend_name, qubit_priority_list)
        with Session(backend=backend) as session:
            sampler = Sampler(mode=session)
            counts = _do_node_groupings(qc_transpiled, sampler, num_shots=num_shots)

    # Convert the measurement tallies to node groupings
    binary_string = max(counts.items(), key=lambda kv: kv[1])[0]
    return [int(y) for y in reversed(list(binary_string))]
