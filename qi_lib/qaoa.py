import numpy as np
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize
from qiskit.primitives import StatevectorSampler

def get_random_parameters(num_params):
    return 2 * np.pi * np.random.rand(num_params)

def minimise_circuit_parameters(cost_func, x0, max_ansatz, max_hamiltonian):
    estimator = StatevectorEstimator()

    result = minimize(
        cost_func, x0, args=(max_ansatz, max_hamiltonian, estimator), method="COBYLA"
    )
    return result.x

def get_node_groupings_from_circuit_parameters(max_ansatz, min_circ_param):
    qc = max_ansatz.assign_parameters(min_circ_param)
    # Add measurements to our circuit
    qc.measure_all()

    sampler = StatevectorSampler()
    job = sampler.run([qc], shots=1024)
    data_pub = job.result()[0].data
    counts = data_pub.meas.get_counts()

    binary_string = max(counts.items(), key=lambda kv: kv[1])[0]
    return [int(y) for y in reversed(list(binary_string))]
