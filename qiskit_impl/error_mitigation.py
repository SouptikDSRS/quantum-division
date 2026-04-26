"""
error_mitigation.py
────────────────────────────────────────────
Measurement Error Mitigation utilities for IBM Quantum hardware.

Features:
✔ Local readout error mitigation (using Qiskit Experiments)
✔ Mitigator caching (IMPORTANT for speed)
✔ Easy integration with existing runner
"""

import warnings
from typing import Dict, Any

# Try to import the preferred mitigation library (qiskit-experiments)
try:
    from qiskit_experiments.library import LocalReadoutError
    _has_experiments = True
except ImportError:
    _has_experiments = False
    warnings.warn("qiskit-experiments not installed. Mitigation will be disabled.")

# Optional fallback (not fully implemented)
try:
    from qiskit_ignis.mitigation import CompleteMeasFitter
    _has_ignis = True
except ImportError:
    _has_ignis = False


class ErrorMitigator:
    """
    Handles measurement error mitigation with caching.
    Uses Qiskit Experiments (preferred) or falls back to Ignis (if available).
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def _build_mitigator(self, backend, qubits):
        """
        Build a readout error mitigator using qiskit-experiments.
        """
        if not _has_experiments:
            raise ImportError("qiskit-experiments is required for mitigation.")
        print(f"[MITIGATION] Building calibrator for {len(qubits)} qubits...")
        exp = LocalReadoutError(qubits)
        expdata = exp.run(backend)
        expdata.block_for_results()
        mitigator = expdata.analysis_results("Local Readout Mitigator").value
        print("[MITIGATION] Calibration complete.")
        return mitigator

    def get_mitigator(self, backend, num_qubits):
        """
        Get cached mitigator or build a new one.
        Returns None if mitigation is unavailable.
        """
        key = f"{backend.name}_{num_qubits}"

        if key not in self._cache:
            try:
                qubits = list(range(num_qubits))
                mitigator = self._build_mitigator(backend, qubits)
            except Exception as e:
                print(f"[MITIGATION] Failed to build mitigator: {e}")
                mitigator = None
            self._cache[key] = mitigator

        return self._cache[key]

    def apply(self, counts, mitigator):
        """
        Apply mitigation to raw counts / quasi distributions.
        If mitigator is None, returns counts unchanged.
        """
        if mitigator is None:
            return counts
        try:
            # For LocalReadoutError mitigator
            quasi_probs = mitigator.quasi_probabilities(counts)
            corrected = quasi_probs.nearest_probability_distribution()
            return corrected
        except Exception as e:
            print(f"[MITIGATION WARNING] Failed to apply mitigation: {e}")
            return counts