import numpy as np
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

def _minimise_circuit_parameters_local(cost_func, x0, max_ansatz, max_hamiltonian):
    estimator = StatevectorEstimator()
    return minimize(
        cost_func, x0, args=(max_ansatz, max_hamiltonian, estimator), method="COBYLA"
    )

def _minimise_circuit_parameters_qi(cost_func, x0, max_ansatz, max_hamiltonian):
    provider = QIProvider()
    backend = provider.get_backend("QX emulator")
    estimator_options = EstimatorOptions(resilience_level=1, default_shots=256)

    with Session(backend=backend) as session:
        estimator = Estimator(mode=session, options=estimator_options)
        return minimize(
            cost_func, x0, args=(max_ansatz, max_hamiltonian, estimator), method="COBYLA"
        )

def minimise_circuit_parameters(cost_func, x0, max_ansatz, max_hamiltonian, *, local=True):
    if local:
        result = _minimise_circuit_parameters_local(cost_func, x0, max_ansatz, max_hamiltonian)
    else:
        result = _minimise_circuit_parameters_qi(cost_func, x0, max_ansatz, max_hamiltonian)
    return result.x

def _get_node_groupings_from_circuit_parameters_local(qc):
    sampler = StatevectorSampler()
    job = sampler.run([qc], shots=1024)
    data_pub = job.result()[0].data
    return data_pub.meas.get_counts()

def _get_node_groupings_from_circuit_parameters_qi(qc):
    provider = QIProvider()
    backend = provider.get_backend("QX emulator")
    with Session(backend=backend) as session:
        sampler = Sampler(mode=session)
        job = sampler.run([qc], shots=1024)
        data_pub = job.result()[0].data
        return data_pub.meas.get_counts()

def get_node_groupings_from_circuit_parameters(max_ansatz, min_circ_param, *, local=True):
    qc = max_ansatz.assign_parameters(min_circ_param)
    qc.measure_all()

    if local:
        counts = _get_node_groupings_from_circuit_parameters_local(qc)
    else:
        counts = _get_node_groupings_from_circuit_parameters_qi(qc)

    binary_string = max(counts.items(), key=lambda kv: kv[1])[0]
    return [int(y) for y in reversed(list(binary_string))]
