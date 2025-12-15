# OUTDATED, TRY TO NOT USE IT

import os
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

from qaoa_lib.graphs import create_graph, draw_graph, graph_to_pauli_list
from qaoa_lib.qaoa import get_random_parameters, minimise_circuit_parameters, get_node_groupings_from_circuit_parameters, plot_histogram


# Name of the test
TEST_NAME="QI_T5_20_2"
os.makedirs(os.path.join("test_results", TEST_NAME), exist_ok=True)

# First Alex's test
# N = 4
# EDGES = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]

# 5-wheel graph
N=5
EDGES = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 1, 1.0)]

# Number of evaluations made to estimate the cost function in each iteration
OPTIMIZER_NUM_SHOTS = 256
# Number of measurements made to the final circuit
NODE_GROUPING_NUM_SHOTS = 1024
# Maximum number of iterations of the optimizer (the minimum is #params + 2)
MAX_ITER=20
# Tolerance of the optimizer
TOL=0.001
# Layers of the circuit
REPS=2

LOCAL = False
QI_BACKEND = "Tuna-5"
QI_QUBIT_PRIORITY_LIST = [0, 1, 2, 3, 4]


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
    return cost


open(os.path.join("test_results", TEST_NAME, f"config_{TEST_NAME}.txt"), "w").write(f"TEST_NAME={TEST_NAME}\nN={N}\nEDGES={EDGES}\nOPTIMIZER_NUM_SHOTS={OPTIMIZER_NUM_SHOTS}\nNODE_GROUPING_NUM_SHOTS={NODE_GROUPING_NUM_SHOTS}\nMAX_ITER={MAX_ITER}\nTOL={TOL}\nREPS={REPS}\nLOCAL={LOCAL}\nQI_BACKEND={QI_BACKEND}\nQI_QUBIT_PRIORITY_LIST={QI_QUBIT_PRIORITY_LIST}")

graph = create_graph(N, EDGES)
draw_graph(graph,filename=os.path.join("test_results", TEST_NAME, f"initial_graph_{TEST_NAME}.jpg"))

# Get Hamiltonian from graph edges
max_hamiltonian = SparsePauliOp.from_list(graph_to_pauli_list(N, EDGES))
# Generate ansatz from Hamiltonain
max_ansatz = QAOAAnsatz(max_hamiltonian, reps=REPS)
# max_ansatz.draw("mpl", filename="max_ansatz.jpg")

# Get initial params
x0 = get_random_parameters(max_ansatz.num_parameters)
print("Initial parameters:", x0)

# Optimise circuit parameters
x = minimise_circuit_parameters(cost_func, x0, max_ansatz, max_hamiltonian, local=LOCAL, backend_name=QI_BACKEND, qubit_priority_list=QI_QUBIT_PRIORITY_LIST, num_shots=OPTIMIZER_NUM_SHOTS, max_iter=MAX_ITER, tol=TOL)
print("Optimised parameters:", x)
node_groupings, counts = get_node_groupings_from_circuit_parameters(max_ansatz, x, local=LOCAL, backend_name=QI_BACKEND, qubit_priority_list=QI_QUBIT_PRIORITY_LIST, num_shots=NODE_GROUPING_NUM_SHOTS)
print("Node groupings:", node_groupings)



plot_histogram(counts, EDGES, filename=os.path.join("test_results", TEST_NAME, f"histogram_{TEST_NAME}.jpg"))
draw_graph(graph, filename=os.path.join("test_results", TEST_NAME, f"coloured_graph_{TEST_NAME}.jpg"), node_color=["r" if node_groupings[i] == 0 else "c" for i in range(N)])