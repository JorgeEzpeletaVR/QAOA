import numpy as np
import matplotlib.pyplot as plt
from mitiq.zne import RichardsonFactory
from mitiq.zne.scaling import fold_global
from qiskit import transpile
from qiskit._accelerate.circuit import DAGCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives import StatevectorSampler
from qiskit_ibm_runtime import EstimatorOptions
from qiskit_ibm_runtime import Session
from qiskit_ibm_runtime.estimator import EstimatorV2 as Estimator
from qiskit_ibm_runtime.sampler import SamplerV2 as Sampler
from scipy.optimize import minimize
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def get_random_parameters(num_params,seed=99):
    """
    Get random initial circuit parameters for optimization
    """
    rng = np.random.default_rng(seed)
    return 2 * np.pi * rng.random(num_params)


def _do_circ_param_minimising(cost_func, x0, max_ansatz, max_hamiltonian, estimator, max_iter, tol):
    """
    Helper function for repeated code in circuit parameter optimisation function
    """
    return minimize(cost_func, x0, args=(max_ansatz, max_hamiltonian, estimator), method="COBYLA",options={'maxiter': max_iter, 'tol': tol})


def minimise_circuit_parameters(cost_func, x0, max_ansatz, max_hamiltonian, *, local=True, platform, backend, qubit_priority_list, num_shots, max_iter=30,tol=0.001):
    """
    Minimise cost function (cost_func) for a given random start parameter (x0) for a given ansatz (max_ansatz) and hamiltonian (max_hamiltonian)
    Parameters:
        cost_func (callable): The objective function to minimize (returns energy estimate)
        x0 (ndarray): Initial guess for the ansatz parameters (betas and gammas)
        max_ansatz (QuantumCircuit): Parameterized QAOA ansatz circuit
        max_hamiltonian (SparsePauliOp): Operator representation of the cost Hamiltonian
        local (bool): If True, uses a local statevector simulator; otherwise, uses cloud runtime
        platform (str): Target platform name ("IBM" or "QI") for non-local execution
        backend (Backend): The specific backend instance or name to target
        qubit_priority_list (list): Qubit mapping preferences (specifically for QI backend)
        num_shots (int): Number of shots per energy estimation in the optimizer
        max_iter (int, optional): Maximum number of iterations for the COBYLA optimizer (default: 30)
        tol (float, optional): Tolerance for convergence in the COBYLA optimizer (default: 0.001)

    Returns:
        ndarray: The optimized circuit parameters (angles) that minimize the cost function
    """

    # Gobal variable to store the evaluated cost in each iteration
    global objective_func_vals
    objective_func_vals = []
    
    if local:
        estimator = StatevectorEstimator()
        result = _do_circ_param_minimising(cost_func, x0, max_ansatz, max_hamiltonian, estimator, max_iter, tol)

    else:
        # Transpilation depending on the hardware
        if platform=="QI":
            initial_layout = qubit_priority_list[0:max_ansatz.num_qubits] if qubit_priority_list else None
            max_ansatz_transpiled = transpile(max_ansatz, backend=backend, initial_layout=initial_layout)
            # Make the Hamiltonian matches the transpiled circuit(some QI backends force you to have a specific number of Qubits-->Hamiltonian has to be scaled to match)
            max_hamiltonian_mapped = max_hamiltonian.apply_layout(max_ansatz_transpiled.layout)
            # Can use Session in QI
            with Session(backend=backend) as session:
                estimator = Estimator(mode=session, options=EstimatorOptions(resilience_level=1, default_shots=num_shots, max_execution_time=3600))
                result = _do_circ_param_minimising(cost_func, x0, max_ansatz_transpiled, max_hamiltonian_mapped, estimator, max_iter, tol)

        elif platform=="IBM":
            pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
            max_ansatz_transpiled = pm.run(max_ansatz)
            max_hamiltonian_mapped = max_hamiltonian.apply_layout(max_ansatz_transpiled.layout)
            # No Session in IBM
            estimator = Estimator(mode=backend, options=EstimatorOptions(resilience_level=1, default_shots=num_shots))
            result = _do_circ_param_minimising(cost_func, x0, max_ansatz_transpiled, max_hamiltonian_mapped, estimator, max_iter, tol)
    
    # We only care about the circuit parameters in the result
    return result.x


def cost_func(params, ansatz, hamiltonian, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (Estimator): Estimator primitive instance

    Returns:
        float: Energy estimate
    """
    pub = (ansatz, hamiltonian, params)
    cost = estimator.run([pub]).result()[0].data.evs
    objective_func_vals.append(cost)
    return cost


def cost_func_ZNE(params, ansatz, hamiltonian, estimator):
    ZNE_SCALE_FACTORS = [1, 3, 5]
    ZNE_FACTORY = RichardsonFactory(ZNE_SCALE_FACTORS)

    # Mitiq's folding transpiler assigns to _global_phase on DAGCircuit. The accelerated
    # DAGCircuit exposes a read-only attribute, so we override with a writable property
    # mapped to the public global_phase attribute.
    if not isinstance(getattr(DAGCircuit, "_global_phase", None), property):
        def _get_global_phase(self):
            return self.global_phase
        def _set_global_phase(self, val):
            self.global_phase = val
        DAGCircuit._global_phase = property(_get_global_phase, _set_global_phase)

    bound_ansatz = ansatz.assign_parameters(params)

    energies = []
    for scale in ZNE_SCALE_FACTORS:
        folded = fold_global(bound_ansatz, scale_factor=scale)

        # Ensure folded circuit is compatible with the backend (e.g. strip unsupported ops).
        backend = getattr(estimator, "_backend", None)
        if backend is not None:
            folded = transpile(folded, backend=backend, optimization_level=1)

        energy = estimator.run([(folded, hamiltonian, [])]).result()[0].data.evs
        energies.append(energy)

    return ZNE_FACTORY.extrapolate(ZNE_SCALE_FACTORS, energies)


def _do_node_groupings(qc, sampler, *, num_shots):
    """
    Helper function for repeated code in circuit param to node grouping function. Repeatedly sample the ansatz circuit, with the optimised parameters

    Parameters:
        qc (QuantumCircuit): The quantum circuit to be sampled (must include measurements)
        sampler (Sampler): The Qiskit Sampler primitive instance (StatevectorSampler or SamplerV2)
        num_shots (int): Number of shots (repetitions) for the execution

    Returns:
        dict: A dictionary mapping measured bitstrings to their counts (e.g., {'010': 15, '110': 3})
    """
    job = sampler.run([qc], shots=num_shots)
    data_pub = job.result()[0].data
    return data_pub.meas.get_counts()



def get_node_groupings_from_circuit_parameters(max_ansatz, min_circ_param, *, local=True, platform, backend, qubit_priority_list, num_shots):
    """Run the optimized ansatz to measure the final solution (node partition).

    Parameters:
        max_ansatz (QuantumCircuit): Parameterized ansatz circuit
        min_circ_param (ndarray): Optimized circuit parameters found by the optimizer
        local (bool): If True, runs on a local simulator; otherwise, runs on the specified platform
        platform (str): Name of the platform ("IBM" or "QI")
        backend (Backend): Backend instance or name to execute the circuit
        qubit_priority_list (list): List of physical qubits to map the logical qubits (for QI)
        num_shots (int): Number of shots (repetitions) for the measurement

    Returns:
        list: The most frequent bitstring converted to a list of integers (the solution)
        dict: The complete dictionary of measurement counts
    """
    # Apply the optimised circuit parameters to the ansatz circuit
    qc = max_ansatz.assign_parameters(min_circ_param)
    # Add measurements to the circuit
    qc.measure_all()

    if local:
        sampler = StatevectorSampler()
        counts = _do_node_groupings(qc, sampler, num_shots=num_shots)
    else:
        if platform == "IBM":
            pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
            qc_transpiled = pm.run(qc)
            sampler = Sampler(mode=backend)
            counts = _do_node_groupings(qc_transpiled, sampler, num_shots=num_shots)

        elif platform == "QI":
            initial_layout = qubit_priority_list[0:qc.num_qubits] if qubit_priority_list else None
            qc_transpiled = transpile(qc, backend=backend, initial_layout=initial_layout)
            with Session(backend=backend) as session:
                sampler = Sampler(mode=session)
                counts = _do_node_groupings(qc_transpiled, sampler, num_shots=num_shots)

    # Convert the measurement tallies to node groupings
    binary_string = max(counts.items(), key=lambda kv: kv[1])[0]
    return [int(y) for y in reversed(list(binary_string))], counts


def plot_convergence(filename="convergence_result.jpg"):
    """Plot the evolution of the objective function values collected during optimization

    Parameters:
        filename (str): File path where the convergence plot will be saved
    """

    plt.figure(figsize=(12, 6))
    plt.plot(objective_func_vals)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.savefig(filename, dpi=300, bbox_inches='tight')

   
def plot_histogram(counts, edges, filename="histogram_result.jpg"):
    """
    Plots a histogram where the optimal solutions are green and non-optimal solutions are grey and computes best_optimal / best_non_optimal

    Parameters:
        counts (dict): Dictionary mapping bitstrings to their measurement counts (e.g. {'101': 50, '000': 10})
        edges (list): List of tuples representing the graph edges (u, v, weight) to calculate the cut value
        filename (str): File path where the histogram image will be saved
    """

    # Max-cut manual computation
    cut_values = {}
    for bitstring in counts.keys():
        nodes = [int(bit) for bit in reversed(bitstring)] 
        current_cut = 0
        for u, v, w in edges:
            if nodes[u] != nodes[v]:
                current_cut += w
        cut_values[bitstring] = current_cut
    optimal_cut = max(cut_values.values())

    # Construction of the histogram
    labels, values, colors = [], [], []
    best_optimal = 0
    best_non_optimal = 0

    for bitstring, count in counts.items():
        nodes = [int(bit) for bit in reversed(bitstring)]
        current_cut = 0
        for u, v, w in edges:
            if nodes[u] != nodes[v]:
                current_cut += w

        labels.append(bitstring)
        values.append(count)

        if current_cut == optimal_cut:
            colors.append('#2ecc71')
            best_optimal = max(best_optimal, count)
        else:
            colors.append('#95a5a6')
            best_non_optimal = max(best_non_optimal, count)


    combined_sorted = sorted(zip(labels, values, colors), key=lambda x: x[1], reverse=True)[:10]
    labels, values, colors = zip(*combined_sorted)

    # Calculation of the Optimal Dominance: the ratio of the most frequent optimal solution's count to the most frequent non-optimal solution's count.
    if best_non_optimal > 0:
        delta = best_optimal / best_non_optimal
    else:
        delta = best_optimal 
    text_str = f"opt_dominance: {delta:.2f}"

    plt.figure(figsize=(20, 15))
    plt.bar(labels, values, color=colors)
    plt.xlabel('Solutions')
    plt.ylabel('Count')
    plt.title('Max-cut histogram (Top 10)')
    plt.text(0.95, 0.95, text_str,transform=plt.gca().transAxes,fontsize=26,verticalalignment='top',horizontalalignment='right',bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.savefig(filename)
    plt.close()

    return labels[0], cut_values[labels[0]]