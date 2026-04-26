import sys
from typing import Optional
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

from qiskit_impl.division_qiskit import DivisionQiskit
from qiskit_impl.utils import print_circuit_info


class IBMRunner:
    """
    Research-grade runner for IBM Quantum execution.
    """

    def __init__(self, backend_name: Optional[str] = None):
        self.service = QiskitRuntimeService()

        self.backend = (
            self.service.backend(backend_name)
            if backend_name
            else self.service.least_busy(simulator=False)
        )

    # --------------------------------------------------
    # Circuit preparation
    # --------------------------------------------------
    def _prepare_circuit(self, n: int) -> QuantumCircuit:
        """
        Build the division circuit and add a classical register for the quotient.
        """
        qc = DivisionQiskit(n).build()

        # Classical register for quotient
        creg = ClassicalRegister(n, name="quotient")
        qc.add_register(creg)

        # Measure ONLY the quotient register
        for i in range(n):
            qc.measure(3 * n + i, creg[i])

        return qc

    # --------------------------------------------------
    # Execution
    # --------------------------------------------------
    def run(self, n: int = 2, shots: int = 1024):
        """
        Run the division circuit on the selected IBM backend.

        Args:
            n (int): Number of bits for dividend and divisor.
            shots (int): Number of shots for the job.

        Returns:
            dict: Counts of measurement outcomes.
        """
        print(f"\nBackend : {self.backend.name}")
        print(f"n       : {n} bits")

        qc = self._prepare_circuit(n)
        print_circuit_info(qc)

        # Transpile to the target backend with high optimization
        compiled = transpile(
            qc,
            backend=self.backend,
            optimization_level=3
        )

        print("\nAfter transpile:")
        print(compiled.count_ops())

        # Create sampler and run job
        sampler = Sampler(self.backend)
        job = sampler.run([compiled], shots=shots)

        print(f"\nJob ID  : {job.job_id()}")
        print("Waiting for results...")

        result = job.result()
        pub_result = result[0]

        # -----------------------------
        # Robust result extraction
        # -----------------------------
        counts = None

        # New format (Runtime V2) – use the explicit register name "quotient"
        if hasattr(pub_result.data, "quotient"):
            counts = pub_result.data.quotient.get_counts()

        # Memory format (if classical register data is returned as memory)
        elif hasattr(pub_result.data, "memory"):
            counts = Counter(pub_result.data.memory)

        # Older quasi-distribution format (fallback)
        elif hasattr(result, "quasi_dists"):
            quasi = result.quasi_dists[0]
            counts = {k: int(v * shots) for k, v in quasi.items()}

        else:
            raise RuntimeError("Unknown result format from IBM Runtime")

        print("\nCounts:")
        print(counts)

        return counts


# --------------------------------------------------
# CLI ENTRY
# --------------------------------------------------
if __name__ == "__main__":
    n_bits  = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    backend = sys.argv[2] if len(sys.argv) > 2 else None
    shots   = int(sys.argv[3]) if len(sys.argv) > 3 else 1024

    runner = IBMRunner(backend)
    runner.run(n_bits, shots)