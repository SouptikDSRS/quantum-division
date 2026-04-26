"""
run_simulator.py
~~~~~~~~~~~~~~~~
Simulate the quantum division circuit locally with Qiskit Aer.

Usage
-----
    python -m qiskit_impl.run_simulator          # n=3, 1024 shots
    python -m qiskit_impl.run_simulator 4 2048   # n=4, 2048 shots
"""

import sys
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit_impl.division_qiskit import DivisionQiskit
from qiskit_impl.utils import print_circuit_info


def run(n: int = 3, shots: int = 1_024) -> dict:
    """Build, compile, and simulate the division circuit for *n* bits."""
    qc = DivisionQiskit(n).build()
    qc.measure_all()

    print_circuit_info(qc)

    sim      = AerSimulator()
    compiled = transpile(qc, sim)
    job      = sim.run(compiled, shots=shots)
    counts   = job.result().get_counts()

    print(f"\nShots : {shots}")
    print(f"Unique outcomes: {len(counts)}")
    print(f"Counts: {counts}")
    return counts


if __name__ == "__main__":
    n_bits = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    shots  = int(sys.argv[2]) if len(sys.argv) > 2 else 1_024
    run(n_bits, shots)
