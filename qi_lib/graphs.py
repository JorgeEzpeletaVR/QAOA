import rustworkx as rx
from rustworkx.visualization import mpl_draw

def create_graph(num_nodes, edges):
    out = rx.PyGraph()
    out.add_nodes_from(range(num_nodes))
    # The edge syntax is (start, end, weight)
    out.add_edges_from(edges)
    return out

def draw_graph(graph, filename="graph.jpg", node_color=None):
    if node_color is None:
        node_color = "#1192E8"
    mpl_draw(
        graph, pos=rx.shell_layout(graph), with_labels=True, edge_labels=str, node_color=node_color
    ).savefig(filename)


def graph_to_pauli_list(num_nodes, edges):
    """Convert graph edges to a MaxCut Hamiltonian.

    Each edge (i, j) becomes a ZZ term acting on qubits i and j.
    """
    out = []
    for (i, j, weight) in edges:
        pauli_str = ['I'] * num_nodes
        pauli_str[num_nodes - 1 - i] = 'Z'  # Pauli strings use right-to-left indexing
        pauli_str[num_nodes - 1 - j] = 'Z'
        out.append((''.join(pauli_str), weight))
    return out

def get_eigenvalue_offset(edges):
    return -sum(edge[2] for edge in edges) / 2
