"""
This program can run a test for Max-cut in various platforms.
Different TEST_NAME should be specified in each different run.
In each run, a folder with the name of the test in /test_results is created.
This new folder contains:
    - config_{TEST_NAME}.txt: characteristics of the test
    - initial_graph_{TEST_NAME}.jpg: the initial graph
    - histogram_{TEST_NAME}.jpg: the histogram with the measurements in the final circuit
    - coloured_graph_{TEST_NAME}.jpg: final division of the nodes  
    - convergence_{TEST_NAME}.jpg: the graphic with the cost evaluated in each iteration
The test can be run as follows:
    - To run it in local simulator (LOCAL=True) 
    - To run it in any company hardware (LOCAL=False)
        · QI hadware: PLATFORM="QI" and QI_BACKEND and QUBIT_PRIORITY (optional) have to be specified.
        · IBM hardware: PLATFORM="IBM" and IBM_SIM=True to run it in ibm_fez simulator or IBM_SIM=False to use the least busy real hardware
The parametes such as layers (REPS), maximum number of iterations (MAX_ITER), tolerance (TOL), shots made in final measurement
(NUM_SHOTS),... can be changed.
"""

import os
from time import time

# QAOA
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

# Graph
from qaoa_lib.graphs import create_graph, draw_graph, graph_to_pauli_list
from qaoa_lib.qaoa import get_random_parameters, minimise_circuit_parameters, get_node_groupings_from_circuit_parameters, plot_histogram, get_cost_func, plot_convergence

# QI
from qiskit_quantuminspire.qi_provider import QIProvider

# IBM
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator

# Name of the test
TEST_NAME = "Test"
os.makedirs(os.path.join("test_results", TEST_NAME), exist_ok=True)

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

# Generic variables
#LOCAL = False # Real hardware
LOCAL = True  # Simulator
PLATFORM = "IBM"   
#PLATFORM = "QI"

# Caracteristics of the platform
# QI 
#BACKEND_QI = "Tuna-5"
BACKEND_QI = "QX emulator"
QUBIT_PRIORITY =  [0, 1, 2, 3, 4]

# IBM CREDENTIALS (use your credentials, do not use my time :) )
MY_TOKEN = ""
MY_CRN = ""
IBM_SIM=True
#IBM_SIM=False

BACKEND = None

RUN_ZNE=False

# Conection
if LOCAL:
    BACKEND = None
    backend_name="Local simulator"
else:
    if PLATFORM == "IBM":
        if not MY_TOKEN or not MY_CRN:
            raise ValueError("ERROR: To execute the algorithm in IBM hardware, MY_TOKEN and MY_CRN must be defined")
        
        if IBM_SIM:
            service = QiskitRuntimeService(channel="ibm_cloud", token=MY_TOKEN, instance=MY_CRN)
            backend = service.backend("ibm_fez")
            backend_name="ibm_fez_simulator"
            BACKEND = AerSimulator.from_backend(backend)

        else:
            service = QiskitRuntimeService(channel="ibm_cloud", token=MY_TOKEN, instance=MY_CRN)
            BACKEND = service.least_busy(operational=True, simulator=False, min_num_qubits=127)
            QUBIT_PRIORITY = None 
            backend_name=BACKEND.name

    elif PLATFORM == "QI":
        BACKEND = BACKEND_QI 
        provider = QIProvider()
        BACKEND = provider.get_backend(BACKEND_QI)
        backend_name=BACKEND.name
    

graph = create_graph(N, EDGES)
draw_graph(graph, filename=os.path.join("test_results", TEST_NAME, f"initial_graph_{TEST_NAME}.jpg"))

# Get Hamiltonian from graph edges
max_hamiltonian = SparsePauliOp.from_list(graph_to_pauli_list(N, EDGES))
# Generate ansatz from Hamiltonain
max_ansatz = QAOAAnsatz(max_hamiltonian, reps=REPS)
# Get initial params
x0 = get_random_parameters(max_ansatz.num_parameters)
print("Initial parameters:", x0)

# Optimise circuit parameters
x, cost_func_val = minimise_circuit_parameters(get_cost_func, x0, max_ansatz, max_hamiltonian, local=LOCAL,platform=PLATFORM, backend=BACKEND, qubit_priority_list=QUBIT_PRIORITY, num_shots=OPTIMIZER_NUM_SHOTS, max_iter=MAX_ITER, tol=TOL, run_zne=RUN_ZNE)
print("Optimised parameters:", x)
print("Cost function value:", cost_func_val)

# Solution
node_groupings, counts = get_node_groupings_from_circuit_parameters(max_ansatz, x, local=LOCAL, platform=PLATFORM, backend=BACKEND, qubit_priority_list=QUBIT_PRIORITY, num_shots=NODE_GROUPING_NUM_SHOTS)
print("Node groupings:", node_groupings)
open(os.path.join("test_results", TEST_NAME, f"config_{TEST_NAME}.txt"), "w").write(f"TEST_NAME={TEST_NAME}\nBACKEND={backend_name}\nN={N}\nEDGES={EDGES}\nOPTIMIZER_NUM_SHOTS={OPTIMIZER_NUM_SHOTS}\nNODE_GROUPING_NUM_SHOTS={NODE_GROUPING_NUM_SHOTS}\nMAX_ITER={MAX_ITER}\nTOL={TOL}\nREPS={REPS}\nINITIAL_PARAMS={list(x0)}\n")
top_solution, top_cut= plot_histogram(counts, EDGES, filename=os.path.join("test_results", TEST_NAME, f"histogram_{TEST_NAME}.jpg"))
open(os.path.join("test_results", TEST_NAME, f"final_results_{TEST_NAME}.txt"), "w").write(f"Best Solution: {top_solution}\nCut Value: {top_cut}\nInitial Parameters: {x0}\nFinal Parameters: {x}\nFinal Cost Function Value: {cost_func_val}\n")
draw_graph(graph, filename=os.path.join("test_results", TEST_NAME, f"coloured_graph_{TEST_NAME}.jpg"), node_color=["r" if node_groupings[i] == 0 else "c" for i in range(N)])
plot_convergence(filename=os.path.join("test_results", TEST_NAME, f"convergence_{TEST_NAME}.jpg"))
