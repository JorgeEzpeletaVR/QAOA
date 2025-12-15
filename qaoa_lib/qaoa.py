import numpy as np
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives import StatevectorSampler
from qiskit_ibm_runtime import EstimatorOptions
from qiskit_ibm_runtime import Session
from qiskit_ibm_runtime.estimator import EstimatorV2 as Estimator
from qiskit_ibm_runtime.sampler import SamplerV2 as Sampler
from scipy.optimize import minimize
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

objective_func_vals = []


def get_random_parameters(num_params):
    """
    Get random initial circuit parameters for optimization
    """
    return 2 * np.pi * np.random.rand(num_params)


def _do_circ_param_minimising(cost_func, x0, max_ansatz, max_hamiltonian, estimator, max_iter, tol):
    """
    Helper function for repeated code in circuit parameter optimisation function
    """
    return minimize(
        cost_func, x0, args=(max_ansatz, max_hamiltonian, estimator), method="COBYLA",options={'maxiter': max_iter, 'tol': tol}
    )


def minimise_circuit_parameters(cost_func, x0, max_ansatz, max_hamiltonian, *, local=True, platform, backend, qubit_priority_list, num_shots, max_iter=30,tol=0.001):
    """
    Minimise cost function (cost_func) for a given random start parameter (x0) for a given ansatz (max_ansatz) and hamiltonian (max_hamiltonian)
    """
    if local:
        estimator = StatevectorEstimator()
        result = _do_circ_param_minimising(cost_func, x0, max_ansatz, max_hamiltonian, estimator, max_iter, tol)

    else:
        # Transpilation depending on the hardware
        if platform=="QI":
            initial_layout = qubit_priority_list[0:max_ansatz.num_qubits] if qubit_priority_list else None
            max_ansatz_transpiled = transpile(max_ansatz, backend=backend, initial_layout=initial_layout)
            # Make the Hamiltonian matches the transpiled circuit
            # (some of the QI backends force you to have a specific number of Qubits, so the Hamiltonian has to be scaled to match)
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


def _do_node_groupings(qc, sampler, *, num_shots):
    """
    Helper function for repeated code in circuit param to node grouping function:
        Repeatedly sample the ansatz circuit, with the optimised parameters
    """
    job = sampler.run([qc], shots=num_shots)
    data_pub = job.result()[0].data
    return data_pub.meas.get_counts()


def get_node_groupings_from_circuit_parameters(max_ansatz, min_circ_param, *, local=True, platform, backend, qubit_priority_list, num_shots):
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
    plt.figure(figsize=(12, 6))
    plt.plot(objective_func_vals)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.savefig(filename, dpi=300, bbox_inches='tight')



def plot_histogram(counts, edges, filename="histogram_result.jpg"):
    """
    Plots a histogram where the optimal solutions are green and non-optimal solutions are grey
    Computes the percentage of optimal solutions
    QI: the same histogram is displayed in the last project created for the run in https://compute.quantum-inspire.com/projects
    """

    # First pass: calculate cut value for each bitstring to find the optimal
    cut_values = {}
    for bitstring in counts.keys():
        nodes = [int(bit) for bit in reversed(bitstring)]
        current_cut = 0
        for u, v, w in edges:
            if nodes[u] != nodes[v]:
                current_cut += w
        cut_values[bitstring] = current_cut

    optimal_cut = max(cut_values.values())
    optimal_hits = 0
    labels, values, colors = [], [], []

    # Second pass: build histogram data with colors based on optimal cut
    for bitstring, count in counts.items():
        labels.append(bitstring)
        values.append(count)

        if cut_values[bitstring] == optimal_cut:
            colors.append('#2ecc71')
            optimal_hits += count
        else:
            colors.append('#95a5a6')

    # Calculate the percentage of optimal solutions
    percentage=optimal_hits / sum(values) * 100
    print(f"Percentage of optimal solutions: {percentage}%")
    text_str = f"Optimal solutions: {percentage:.2f}%"

    plt.figure(figsize=(20, 15))
    plt.bar(labels, values, color=colors)
    plt.xlabel('Solutions')
    plt.ylabel('Count')
    plt.title('Max-cut histogram')
    plt.text(0.95, 0.95, text_str, transform=plt.gca().transAxes, fontsize=26,verticalalignment='top', horizontalalignment='right',bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.savefig(filename)
    plt.close()