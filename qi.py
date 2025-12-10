import rustworkx as rx
from qiskit.primitives import StatevectorEstimator
from qiskit_ibm_runtime import EstimatorOptions, Session
from qiskit_ibm_runtime.estimator import EstimatorV2 as Estimator
from rustworkx.visualization import mpl_draw

n = 4
G = rx.PyGraph()
G.add_nodes_from(range(n))
# The edge syntax is (start, end, weight)
edges = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
G.add_edges_from(edges)

mpl_draw(
    G, pos=rx.shell_layout(G), with_labels=True, edge_labels=str, node_color="#1192E8",
).savefig("graph.jpg")

from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

hamiltonian = SparsePauliOp.from_list(
    [("IIZZ", 1), ("IZIZ", 1), ("IZZI", 1), ("ZIIZ", 1), ("ZZII", 1)]
)

ansatz = QAOAAnsatz(hamiltonian, reps=2)
# Draw
ansatz.decompose(reps=3).draw("mpl", filename="circuit.jpeg")

offset = -sum(edge[2] for edge in edges) / 2
print(f"""Offset: {offset}""")


def cost_func_vqe(params, ansatz, hamiltonian, estimator):
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
    #    cost = estimator.run(ansatz, hamiltonian, parameter_values=params).result().values[0]
    return cost


import numpy as np
#
x0 = 2 * np.pi * np.random.rand(ansatz.num_parameters)
#
# estimator = StatevectorEstimator()
# cost = cost_func_vqe(x0, ansatz, hamiltonian, estimator)
# print(cost)



from qiskit_quantuminspire.qi_provider import QIProvider
from qiskit import QuantumCircuit, generate_preset_pass_manager

provider = QIProvider()
backend = provider.get_backend("QX emulator")

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_ansatz = pm.run(ansatz)
isa_hamiltonian = hamiltonian.apply_layout(layout=isa_ansatz.layout)

# Set estimator options
estimator_options = EstimatorOptions(resilience_level=1, default_shots=10_000)

# Open a Runtime session:

with Session(backend=backend) as session:
    estimator = Estimator(mode=session, options=estimator_options)
    cost = cost_func_vqe(x0, isa_ansatz, isa_hamiltonian, estimator)

# Close session after done
session.close()
print(cost)
#
# qubit_0 = 0
# qubit_1 = 2
#
# qc = QuantumCircuit(5, 2)
#
# qc.h(qubit_0)
# qc.cx(qubit_0, qubit_1)
# qc.measure(qubit_0, cbit=0)
# qc.measure(qubit_1, cbit=1)
#
# nr_shots = backend.max_shots
# job = backend.run(qc, shots=nr_shots)
# print(job.result().get_counts())
