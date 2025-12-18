import os

# QAOA
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

# Graph
from qaoa_lib.graphs import create_graph, draw_graph, graph_to_pauli_list
from qaoa_lib.qaoa import get_random_parameters, minimise_circuit_parameters, sample_from_circuit_parameters, plot_histogram, get_cost_func, plot_convergence

# QI
from qiskit_quantuminspire.qi_provider import QIProvider

# IBM
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator

# Definition of the graph
# Standard graph: 5 Wheel graph
N = 5
EDGES = [(0, i, 1.0) for i in range(1, N)] + [(i, i+1, 1.0) for i in range(1, N-1)] +  [(N-1, 1, 1.0)]

# Characteristics of the algorithm
# Number of evaluations made to estimate the cost function in each iteration
OPTIMIZER_NUM_SHOTS = 256
# Number of measurements made to the final circuit (for QI max 2048)
NODE_GROUPING_NUM_SHOTS = 1000
# Maximum number of iterations of the optimizer (min num_vars+2)
MAX_ITER = 30
# Tolerance of the optimizer
TOL = 0.001
# Layers of the circuit
REPS = 2

# Name of the test
TEST_NAME = "Test"
os.makedirs(os.path.join("test_results", TEST_NAME), exist_ok=True)

# Defaults (local simulator)
LOCAL = True
PLATFORM = None
MY_TOKEN = None
MY_CRN = None
IBM_SIM = None
BACKEND_NAME = None
QUBIT_PRIORITY = None
RUN_ZNE = False

# -- QI - QX Emulator --
# LOCAL = False
# PLATFORM = "QI"
# BACKEND_NAME = "QX emulator"
# QUBIT_PRIORITY =  [0, 1, 2, 3, 4]

# -- QI - Tuna-5 --
# LOCAL = False
# PLATFORM = "QI"
# BACKEND_NAME = "Tuna-5"
# Qubit priority from: https://github.com/DiCarloLab-Delft/QuantumInspireUtilities/blob/main/getting_started_tuna5.ipynb
# QUBIT_PRIORITY =  [2, 0, 1, 3, 4]

# -- IBM - Fez --
# LOCAL = False
# PLATFORM = "IBM"   
# # IBM CREDENTIALS (use your credentials, do not use my time :) )
# MY_TOKEN = ""
# MY_CRN = ""
# IBM_SIM=True  # Set False to run on real IBM hardware

# -- Add ZNE --
# RUN_ZNE=False

# Conection
if LOCAL:
    BACKEND = None
    BACKEND_NAME="Local simulator"
else:
    if PLATFORM == "IBM":
        if not MY_TOKEN or not MY_CRN:
            raise ValueError("ERROR: To execute the algorithm in IBM hardware, MY_TOKEN and MY_CRN must be defined")
        
        if IBM_SIM:
            service = QiskitRuntimeService(channel="ibm_cloud", token=MY_TOKEN, instance=MY_CRN)
            backend = service.backend("ibm_fez")
            BACKEND = AerSimulator.from_backend(backend)
            BACKEND_NAME="ibm_fez_simulator"

        else:
            service = QiskitRuntimeService(channel="ibm_cloud", token=MY_TOKEN, instance=MY_CRN)
            BACKEND = service.least_busy(operational=True, simulator=False, min_num_qubits=127)
            BACKEND_NAME = BACKEND.name

    elif PLATFORM == "QI":
        provider = QIProvider()
        BACKEND = provider.get_backend(BACKEND_NAME)


# Create a rustworkx graph, and output it to file
graph = create_graph(N, EDGES)
draw_graph(graph, filename=os.path.join("test_results", TEST_NAME, f"initial_graph_{TEST_NAME}.jpg"))

# Get Hamiltonian from graph edges
max_hamiltonian = SparsePauliOp.from_list(graph_to_pauli_list(N, EDGES))
# Generate ansatz from Hamiltonain
max_ansatz = QAOAAnsatz(max_hamiltonian, reps=REPS)
# Get initial parameters
x0 = get_random_parameters(max_ansatz.num_parameters)
print("Initial parameters:", x0)

# Optimise circuit parameters
x, cost_func_val = minimise_circuit_parameters(get_cost_func, x0, max_ansatz, max_hamiltonian, local=LOCAL,platform=PLATFORM, backend=BACKEND, qubit_priority_list=QUBIT_PRIORITY, num_shots=OPTIMIZER_NUM_SHOTS, max_iter=MAX_ITER, tol=TOL, run_zne=RUN_ZNE)
print("Optimised parameters:", x)
print("Cost function value:", cost_func_val)

# Sample optimal circuit parameters
node_groupings, counts = sample_from_circuit_parameters(max_ansatz, x, local=LOCAL, platform=PLATFORM, backend=BACKEND, qubit_priority_list=QUBIT_PRIORITY, num_shots=NODE_GROUPING_NUM_SHOTS)
# Save raw results
print("Node groupings:", node_groupings)
open(os.path.join("test_results", TEST_NAME, f"config_{TEST_NAME}.txt"), "w").write(f"TEST_NAME={TEST_NAME}\nBACKEND={BACKEND_NAME}\nN={N}\nEDGES={EDGES}\nOPTIMIZER_NUM_SHOTS={OPTIMIZER_NUM_SHOTS}\nNODE_GROUPING_NUM_SHOTS={NODE_GROUPING_NUM_SHOTS}\nMAX_ITER={MAX_ITER}\nTOL={TOL}\nREPS={REPS}\nINITIAL_PARAMS={list(x0)}\n")
# Plot histogram
top_solution, top_cut = plot_histogram(counts, EDGES, filename=os.path.join("test_results", TEST_NAME, f"histogram_{TEST_NAME}.jpg"))
# Add optimal solution to test results
open(os.path.join("test_results", TEST_NAME, f"final_results_{TEST_NAME}.txt"), "w").write(f"Best Solution: {top_solution}\nCut Value: {top_cut}\nInitial Parameters: {x0}\nFinal Parameters: {x}\nFinal Cost Function Value: {cost_func_val}\n")
# Draw graph again, with optimal max cut drawn
draw_graph(graph, filename=os.path.join("test_results", TEST_NAME, f"coloured_graph_{TEST_NAME}.jpg"), node_color=["r" if node_groupings[i] == 0 else "c" for i in range(N)])
# Plot convergence
plot_convergence(filename=os.path.join("test_results", TEST_NAME, f"convergence_{TEST_NAME}.jpg"))
