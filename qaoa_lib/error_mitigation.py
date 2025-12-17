from mitiq.zne import RichardsonFactory
from mitiq.zne.scaling import fold_global
from qiskit._accelerate.circuit import DAGCircuit
from qiskit import transpile

def get_cost_func_ZNE(params, ansatz, hamiltonian, estimator):
    """
    Return estimate of energy from estimator applying Zero Noise Extrapolation

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (Estimator): Estimator primitive instance

    Returns:
        float: Energy estimate
    """

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
