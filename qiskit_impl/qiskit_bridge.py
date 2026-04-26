"""
qiskit_bridge.py
────────────────────────────────────────────
Bridge between Qiskit circuits and paper metrics.
Supports both ideal simulation and real hardware execution via IBM Quantum,
with optional measurement error mitigation.
"""

import random
import numpy as np
from typing import Optional, Dict, List
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit.circuit import ClassicalRegister

from qiskit_impl.division_qiskit import DivisionQiskit
from qiskit_impl.utils import print_circuit_info
from qiskit_impl.error_mitigation import ErrorMitigator   # new import


class HardwareDivisionRunner:
    """
    Runs division circuit on IBM Quantum hardware (or a simulator backend).
    Optionally applies measurement error mitigation.
    """

    def __init__(self, backend_name: Optional[str] = None, use_mitigation: bool = False):
        self.service = QiskitRuntimeService()
        if backend_name:
            self.backend = self.service.backend(backend_name)
        else:
            self.backend = self.service.least_busy(simulator=False)
        self.use_mitigation = use_mitigation
        if self.use_mitigation:
            self.mitigator = ErrorMitigator()
        else:
            self.mitigator = None

    def _prepare_circuit(self, n: int, dividend: int, divisor: int) -> QuantumCircuit:
        """
        Build division circuit with fixed input values.

        Qubit layout:
          - qubits 0..n-1: dividend (LSB at qubit 0)
          - qubits n..2n-1: divisor (LSB at qubit n)
          - qubits 2n..3n-1: remainder / quotient
        """
        qc = DivisionQiskit(n).build()

        # Add classical register for quotient
        creg = ClassicalRegister(n, name="quotient")
        qc.add_register(creg)

        # Initialize dividend and divisor registers
        for i in range(n):
            if (dividend >> i) & 1:
                qc.x(i)                 # dividend qubit i
            if (divisor >> i) & 1:
                qc.x(n + i)             # divisor qubit i

        # Measure quotient register (qubits 2n .. 3n-1)
        for i in range(n):
            qc.measure(2 * n + i, creg[i])

        return qc

    def run(self, n: int, dividend: int, divisor: int, shots: int = 1024) -> Dict[int, int]:
        """
        Execute the circuit for given dividend and divisor.
        Returns counts dictionary mapping measured quotient (int) to frequency.
        """
        qc = self._prepare_circuit(n, dividend, divisor)

        # Transpile
        compiled = transpile(qc, backend=self.backend, optimization_level=3)

        # Run using Sampler
        sampler = Sampler(self.backend)
        job = sampler.run([compiled], shots=shots)
        result = job.result()
        pub_result = result[0]

        # Extract raw counts using the named register
        if hasattr(pub_result.data, "quotient"):
            raw_counts = pub_result.data.quotient.get_counts()
        elif hasattr(pub_result.data, "memory"):
            # fallback to memory format (should not happen with named register)
            raw_counts = Counter(pub_result.data.memory)
        else:
            raise RuntimeError("Unknown result format from IBM Runtime")

        # Apply mitigation if enabled
        if self.use_mitigation and self.mitigator:
            # Get the mitigator (cached) for all qubits (3n)
            mitigator = self.mitigator.get_mitigator(self.backend, 3 * n)
            corrected = self.mitigator.apply(raw_counts, mitigator)
            # corrected is either a dict of probabilities or counts
            total = sum(corrected.values())
            if total <= 1.0:   # probabilities
                # Convert to counts by scaling to shots
                counts = {int(k, 2): int(v * shots + 0.5) for k, v in corrected.items()}
            else:               # already counts
                counts = {int(k, 2): v for k, v in corrected.items()}
        else:
            # No mitigation: convert binary strings directly
            counts = {int(k, 2): v for k, v in raw_counts.items()}

        return counts

    def run_multiple(self, n: int, num_tests: int = 100, shots: int = 1024) -> List[Dict]:
        """
        Run multiple random test cases and return list of results.
        """
        results = []
        for _ in range(num_tests):
            # Random dividend and divisor (divisor != 0)
            dividend = random.randint(0, 2**n - 1)
            divisor = random.randint(1, 2**n - 1)
            counts = self.run(n, dividend, divisor, shots)
            # Compute expected quotient
            expected = dividend // divisor
            # Accuracy: fraction of shots where measured quotient equals expected
            total_shots = sum(counts.values())
            correct_shots = counts.get(expected, 0)
            accuracy = correct_shots / total_shots
            results.append({
                "dividend": dividend,
                "divisor": divisor,
                "expected": expected,
                "counts": counts,
                "accuracy": accuracy,
            })
        return results


class QiskitDivisionRunner:
    """
    Runs Qiskit division circuit and extracts metrics
    in the SAME format as paper metrics.
    Supports both ideal simulation and real hardware,
    optionally with measurement error mitigation.
    """

    def __init__(self, backend: Optional[str] = None, use_mitigation: bool = False):
        """
        backend: If None, use AerSimulator (ideal). If string, use hardware backend.
        use_mitigation: Whether to apply measurement error mitigation (hardware only).
        """
        if backend is None:
            self.backend = AerSimulator()
            self.hardware = None
            self.use_mitigation = False   # no mitigation for simulation
        else:
            self.hardware = HardwareDivisionRunner(backend, use_mitigation)
            self.backend = None
            self.use_mitigation = use_mitigation

    # ------------------------------------------------------------ #
    # Build + simulate (ideal)
    # ------------------------------------------------------------ #
    def run_simulation(self, n: int, shots: int = 1024) -> tuple:
        """
        Run ideal simulation on AerSimulator.
        """
        qc = DivisionQiskit(n).build()
        qc.measure_all()   # measure all qubits for simulation

        compiled = transpile(qc, self.backend)
        job = self.backend.run(compiled, shots=shots)
        result = job.result()
        counts = result.get_counts()
        return counts, compiled

    # ------------------------------------------------------------ #
    # Hardware execution (using HardwareDivisionRunner)
    # ------------------------------------------------------------ #
    def run_hardware(self, n: int, num_tests: int = 100, shots: int = 1024) -> List[Dict]:
        """
        Run multiple random test cases on hardware and return results.
        """
        if self.hardware is None:
            raise RuntimeError("Hardware runner not initialized. Provide backend name.")
        return self.hardware.run_multiple(n, num_tests, shots)

    # ------------------------------------------------------------ #
    # Metrics extraction
    # ------------------------------------------------------------ #
    def get_metrics_simulation(self, n: int) -> Dict:
        """
        Extract metrics for ideal simulation (accuracy = 1.0).
        """
        # Build the original circuit to count logical gates
        qc_logical = DivisionQiskit(n).build()
        ops = qc_logical.count_ops()

        # Compute T-count from logical Toffoli gates
        t_count = ops.get("ccx", 0) * 7
        # Depth could be computed from logical circuit, but we can approximate
        t_depth = None
        cnot_count = ops.get("cx", 0) + ops.get("cz", 0)

        return {
            "method": "qiskit_ideal",
            "n_bits": n,
            "t_count": t_count,
            "t_depth": t_depth,
            "cnot_count": cnot_count,
            "accuracy": 1.0,
        }

    def get_metrics_hardware(self, n: int, num_tests: int = 100, shots: int = 1024) -> Dict:
        """
        Extract metrics from hardware runs: average accuracy and average resource usage.
        """
        results = self.run_hardware(n, num_tests, shots)
        avg_accuracy = np.mean([r["accuracy"] for r in results])

        # For resource usage, use the logical circuit (same as simulation) for consistency
        qc_logical = DivisionQiskit(n).build()
        ops = qc_logical.count_ops()
        t_count = ops.get("ccx", 0) * 7
        cnot_count = ops.get("cx", 0) + ops.get("cz", 0)
        t_depth = None

        return {
            "method": f"qiskit_hardware_{self.hardware.backend.name}",
            "n_bits": n,
            "t_count": t_count,
            "t_depth": t_depth,
            "cnot_count": cnot_count,
            "accuracy": avg_accuracy,
            "extra": {
                "num_tests": num_tests,
                "shots": shots,
                "mitigation": self.use_mitigation,
            }
        }