"""
utils.py
~~~~~~~~
Shared helpers for circuit inspection and result formatting.
"""

from __future__ import annotations
from qiskit import QuantumCircuit


# ──────────────────────────────────────────────────────────────────────── #
#  Circuit diagnostics                                                     #
# ──────────────────────────────────────────────────────────────────────── #

def print_circuit_info(qc: QuantumCircuit) -> None:
    """Print a concise summary of a circuit's resources."""
    ops = qc.count_ops()
    print(
        f"\n{'─' * 40}\n"
        f"  Circuit : {qc.name}\n"
        f"  Qubits  : {qc.num_qubits}\n"
        f"  Depth   : {qc.depth()}\n"
        f"  Gates   : {dict(ops)}\n"
        f"{'─' * 40}\n"
    )


def circuit_resource_dict(qc: QuantumCircuit) -> dict:
    """Return circuit metadata as a plain dict (useful for logging/CSV)."""
    return {
        "name":    qc.name,
        "qubits":  qc.num_qubits,
        "depth":   qc.depth(),
        "gates":   dict(qc.count_ops()),
    }


# ──────────────────────────────────────────────────────────────────────── #
#  Result helpers                                                          #
# ──────────────────────────────────────────────────────────────────────── #

def most_likely(counts: dict[str, int]) -> str:
    """Return the bitstring with the highest count."""
    return max(counts, key=counts.get)


def counts_to_probabilities(counts: dict[str, int]) -> dict[str, float]:
    """Normalise raw counts to probability estimates."""
    total = sum(counts.values())
    return {state: count / total for state, count in counts.items()}
