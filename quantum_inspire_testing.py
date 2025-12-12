from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

from qi_lib.graphs import create_graph, draw_graph, graph_to_pauli_list
from qi_lib.qaoa import get_random_parameters, minimise_circuit_parameters, get_node_groupings_from_circuit_parameters

N = 4
EDGES = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
# N = 5
# EDGES = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0), (0, 4, 1.0), (3, 4, 1.0)]
OPTIMIZER_NUM_SHOTS = 256
NODE_GROUPING_NUM_SHOTS = 1024
LOCAL = False
QI_BACKEND = "QX emulator"
QI_QUBIT_PRIORITY_LIST = [2, 0, 1, 3, 4]


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

graph = create_graph(N, EDGES)
draw_graph(graph)

# Get Hamiltonian from graph edges
max_hamiltonian = SparsePauliOp.from_list(graph_to_pauli_list(N, EDGES))
# Generate ansatz from Hamiltonain
max_ansatz = QAOAAnsatz(max_hamiltonian, reps=2)
# max_ansatz.draw("mpl", filename="max_ansatz.jpg")
# Get initial params
x0 = get_random_parameters(max_ansatz.num_parameters)
print("Initial parameters:", x0)
# Optimise circuit parameters
x = minimise_circuit_parameters(cost_func, x0, max_ansatz, max_hamiltonian, local=LOCAL, backend_name=QI_BACKEND, qubit_priority_list=QI_QUBIT_PRIORITY_LIST, num_shots=OPTIMIZER_NUM_SHOTS)
# x = [3.8471405, 0.29315694, 4.99223468, 1.14716908]
print("Optimised parameters:", x)
node_groupings,counts = get_node_groupings_from_circuit_parameters(max_ansatz, x, local=LOCAL, backend_name=QI_BACKEND, qubit_priority_list=QI_QUBIT_PRIORITY_LIST, num_shots=NODE_GROUPING_NUM_SHOTS)
print("Node groupings:", node_groupings)

draw_graph(graph, filename="graph_coloured.jpg", node_color=["r" if node_groupings[i] == 0 else "c" for i in range(N)])
