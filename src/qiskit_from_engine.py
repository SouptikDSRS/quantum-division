from src.quantum_gates import QuantumCircuit
from src.comp_n_sub import CompNSub, ApproxCompNSub


def build_circuit_from_method(method, n, k=None):
    """
    Builds full division circuit using COMP-N-SUB blocks.
    Returns YOUR custom QuantumCircuit (with Qiskit inside).
    """

    qc = QuantumCircuit(3 * n + 2, name=f"division_{method}_n{n}")

    for _ in range(n):

        if method == "exact":
            sub = CompNSub(n).build()

        elif method == "approx_truncated":
            sub = ApproxCompNSub(n, k=k or n // 2).build()

        elif method == "hybrid":
            sub = ApproxCompNSub(n, k=max(2, n // 3)).build()

        else:
            sub = CompNSub(n).build()

        # 🔥 IMPORTANT: merge circuits
        qc += sub

    return qc