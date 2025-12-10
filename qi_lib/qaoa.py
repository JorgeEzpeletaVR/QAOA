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
    return 2 * np.pi * np.random.rand(num_params)

def get_backend_and_transpile(qc):
    provider = QIProvider()
    backend = provider.get_backend("QX emulator")

    qubit_priority_list = [2, 0, 1, 3, 4]
    qc_transpiled = transpile(qc, backend, initial_layout=qubit_priority_list[0:qc.num_qubits])
    return backend, qc_transpiled

def _do_circ_param_minimising(cost_func, x0, max_ansatz, max_hamiltonian, estimator):
    return minimize(
        cost_func, x0, args=(max_ansatz, max_hamiltonian, estimator), method="COBYLA"
    )

def minimise_circuit_parameters(cost_func, x0, max_ansatz, max_hamiltonian, *, local=True):
    if local:
        estimator = StatevectorEstimator()
        result = _do_circ_param_minimising(cost_func, x0, max_ansatz, max_hamiltonian, estimator)
    else:
        backend, max_ansatz_transpiled = get_backend_and_transpile(max_ansatz)
        max_hamiltonian_mapped = max_hamiltonian.apply_layout(max_ansatz_transpiled.layout)
        with Session(backend=backend) as session:
            estimator = Estimator(mode=session, options=EstimatorOptions(resilience_level=1, default_shots=256))
            result = _do_circ_param_minimising(cost_func, x0, max_ansatz, max_hamiltonian_mapped, estimator)
    return result.x

def _do_node_groupings_sampling(qc, sampler):
    job = sampler.run([qc], shots=1024)
    data_pub = job.result()[0].data
    return data_pub.meas.get_counts()

def get_node_groupings_from_circuit_parameters(max_ansatz, min_circ_param, *, local=True):
    qc = max_ansatz.assign_parameters(min_circ_param)
    qc.measure_all()

    if local:
        sampler = StatevectorSampler()
        counts = _do_node_groupings_sampling(qc, sampler)
    else:
        backend, qc_transpiled = get_backend_and_transpile(qc)
        with Session(backend=backend) as session:
            sampler = Sampler(mode=session)
            counts = _do_node_groupings_sampling(qc_transpiled, sampler)

    binary_string = max(counts.items(), key=lambda kv: kv[1])[0]
    return [int(y) for y in reversed(list(binary_string))]
