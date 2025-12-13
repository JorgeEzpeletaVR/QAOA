"""
This program can run a test for Max-cut in various platforms.
Different TEST_NAME should be specified in each different run.
In each run, a folder with the name of the test in /test_results is created.
This new folder contains:
    - config_{TEST_NAME}.txt: characteristics of the test
    - initial_graph_{TEST_NAME}.jpg: the initial graph
    - histogram_{TEST_NAME}.jpg: the histogram with the measurements in the final circuit
    - coloured_graph_{TEST_NAME}.jpg: final division of the nodes  
The test can be run as follows:
    - To run it in local simulator (LOCAL=True) 
    - To run it  real hardware (LOCAL=False)
        · QI hadware: PLATFORM="QI"  and QI_BACKEND and QUBIT_PRIORITY (optional) have to ve specified.
        · IBM hadware: PLATFORM="IBM".
The parametes such as layers (REPS), maximum number of iterations (MAX_ITER), tolerance (TOL), shots made in final measurement
(NUM_SHOTS),... can be changed.
"""

import os

# QAOA
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

# Graph
from lib.graphs import create_graph, draw_graph, graph_to_pauli_list
from lib.qaoa import get_random_parameters, minimise_circuit_parameters, get_node_groupings_from_circuit_parameters, plot_histogram, cost_func

# QI
from qiskit_quantuminspire.qi_provider import QIProvider

# IBM
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService


# Name of the test
TEST_NAME = "IBM"
os.makedirs(os.path.join("test_results", TEST_NAME), exist_ok=True)

# Defintion of the graph
N = 5
EDGES = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 1, 1.0)]

# Characteristics of the algorithm
# Number of evaluations made to estimate the cost function in each iteration
OPTIMIZER_NUM_SHOTS = 128
# Number of measurements made to the final circuit (for QI max 2048)
NODE_GROUPING_NUM_SHOTS = 1024
# Maximum number of iterations of the optimizer (min num_vars+2)
MAX_ITER = 4
# Tolerance of the optimizer
TOL = 0.001
# Layers of the circuit
REPS = 1

# Caracteristics of the platform
# QI 
BACKEND_QI = "QX emulator"
QUBIT_PRIORITY =  [0, 1, 2, 3, 4]

# IBM CREDENTIALS (use your credentials, do not use my time :) )
MY_TOKEN = "bvu8zeFlxWA426m5UuarVkM05nbow6_bf7Y7v-Lp18ZQ"
MY_CRN = "crn:v1:bluemix:public:quantum-computing:us-east:a/41a25b51a97b4a419519ef7eaceadf4c:f31cb64e-2de3-4367-9694-3fc74578e7c7::"


# Generic variables
PLATFORM = "IBM"     # Options: "IBM" or "QI"
LOCAL = False        # True --> Simulator, False --> Real hardware
BACKEND = None

# Conection
if LOCAL:
    BACKEND = None
    backend_name="Local simulator"
else:
    if PLATFORM == "IBM":
        if not MY_TOKEN or not MY_CRN:
            raise ValueError("ERROR: Para ejecutar en IBM hardware necesitas definir MY_TOKEN y MY_CRN.")
        service = QiskitRuntimeService(channel="ibm_cloud", token=MY_TOKEN, instance=MY_CRN)
        BACKEND = service.least_busy(operational=True, simulator=False, min_num_qubits=127)
        QUBIT_PRIORITY = None 
        backend_name=BACKEND.name
    elif PLATFORM == "QI":
        BACKEND = BACKEND_QI 
        provider = QIProvider()
        BACKEND = provider.get_backend(BACKEND_QI)
        backend_name=BACKEND.name
    
open(os.path.join("test_results", TEST_NAME, f"config_{TEST_NAME}.txt"), "w").write(f"TEST_NAME={TEST_NAME}\nBACKEND={backend_name}\nN={N}\nEDGES={EDGES}\nOPTIMIZER_NUM_SHOTS={OPTIMIZER_NUM_SHOTS}\nNODE_GROUPING_NUM_SHOTS={NODE_GROUPING_NUM_SHOTS}\nMAX_ITER={MAX_ITER}\nTOL={TOL}\nREPS={REPS}\n")

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
x = minimise_circuit_parameters(cost_func, x0, max_ansatz, max_hamiltonian, local=LOCAL,platform=PLATFORM, backend=BACKEND, qubit_priority_list=QUBIT_PRIORITY, num_shots=OPTIMIZER_NUM_SHOTS, max_iter=MAX_ITER, tol=TOL)
print("Optimised parameters:", x)

# Solution
node_groupings, counts = get_node_groupings_from_circuit_parameters(max_ansatz, x, local=LOCAL, platform=PLATFORM, backend=BACKEND, qubit_priority_list=QUBIT_PRIORITY, num_shots=NODE_GROUPING_NUM_SHOTS)
print("Node groupings:", node_groupings)
plot_histogram(counts, EDGES, filename=os.path.join("test_results", TEST_NAME, f"histogram_{TEST_NAME}.jpg"))
draw_graph(graph, filename=os.path.join("test_results", TEST_NAME, f"coloured_graph_{TEST_NAME}.jpg"), node_color=["r" if node_groupings[i] == 0 else "c" for i in range(N)])