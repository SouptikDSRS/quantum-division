"""
comp_n_sub.py
─────────────────────────────────────────────────────────────────────────────
COMP-N-SUB primitive: Compare two n-bit integers and conditionally subtract.

Qubit layout — EXACT (3n + 2 qubits total):
  a[i]    = i            i in [0, n)   — partial remainder
  b[i]    = n + i        i in [0, n)   — divisor (read-only)
  c[i]    = 2*n + i      i in [0, n)   — carry chain ancillae (one per bit)
  qbit    = 3*n                        — quotient bit output
  scratch = 3*n + 1                    — reserved

Paper-reported costs (exact):
  T-count  = 4n - 3
  T-depth  = 2*ceil(log2 n) + 2
  CNOT     = 6n - 3

Approximate variant (top-k-bit truncation):
  T-count  = 4k - 3,  error prob <= 2^(k-n)
"""

from __future__ import annotations
import math
from typing import Optional

from .quantum_gates import (
    QuantumCircuit, Gate,
    T, Tdg, H, CNOT, Toffoli, Toffoli_approx, X, S, Sdg
)


class CompNSub:
    """Exact COMP-N-SUB using ripple-carry comparison (Cuccaro et al. 2004)."""

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.n_qubits = 3 * n + 2

    def _a(self, i): return i
    def _b(self, i): return self.n + i
    def _c(self, i): return 2 * self.n + i
    def _qbit(self): return 3 * self.n

    def _comparison_forward(self, circ):
        n = self.n
        circ.append(X(self._c(0)))   # carry seed = 1
        for i in range(n):
            a_i, b_i, c_i = self._a(i), self._b(i), self._c(i)
            circ.append(X(b_i))                        # ~B[i]
            circ.append(CNOT(c_i, b_i))
            circ.append(CNOT(c_i, a_i))
            if i < n - 1:
                c_next = self._c(i + 1)
                circ.append(Toffoli(a_i, b_i, c_next))   # carry into c[i+1]
                circ.append(CNOT(c_i, a_i))
                circ.append(CNOT(c_i, b_i))
                circ.append(X(b_i))
            else:
                qbit = self._qbit()
                circ.append(Toffoli(a_i, b_i, qbit))     # carry-out = qbit
                circ.append(CNOT(c_i, a_i))
                circ.append(CNOT(c_i, b_i))
                circ.append(X(b_i))

    def _conditional_subtraction(self, circ):
        n = self.n
        qbit = self._qbit()
        for i in range(n):
            a_i, b_i = self._a(i), self._b(i)
            circ.append(Toffoli(qbit, b_i, a_i))
            if i < n - 1:
                circ.append(Toffoli(a_i, b_i, self._a(i + 1)))
                circ.append(CNOT(a_i, b_i))

    def _comparison_uncompute(self, circ):
        n = self.n
        for i in reversed(range(n)):
            a_i, b_i, c_i = self._a(i), self._b(i), self._c(i)
            circ.append(X(b_i))
            circ.append(CNOT(c_i, b_i))
            circ.append(CNOT(c_i, a_i))
            if i < n - 1:
                c_next = self._c(i + 1)
                circ.append(Toffoli(a_i, b_i, c_next))
                circ.append(CNOT(c_i, a_i))
                circ.append(CNOT(c_i, b_i))
                circ.append(X(b_i))
            else:
                qbit = self._qbit()
                circ.append(Toffoli(a_i, b_i, qbit))
                circ.append(CNOT(c_i, a_i))
                circ.append(CNOT(c_i, b_i))
                circ.append(X(b_i))
        circ.append(X(self._c(0)))

    def build(self) -> QuantumCircuit:
        circ = QuantumCircuit(self.n_qubits, name=f"CompNSub_exact_n{self.n}")
        self._comparison_forward(circ)
        self._conditional_subtraction(circ)
        self._comparison_uncompute(circ)
        return circ

    def analytical_cost(self) -> dict:
        n = self.n
        return {
            "method":     "exact",
            "n_bits":     n,
            "t_count":    4 * n - 3,
            "t_depth":    2 * math.ceil(math.log2(n)) + 2 if n > 1 else 2,
            "cnot_count": 6 * n - 3,
            "n_qubits":   self.n_qubits,
        }


class ApproxCompNSub:
    """Approximate COMP-N-SUB using top-k-bit truncation."""

    def __init__(self, n: int, k: Optional[int] = None, epsilon: float = 0.05):
        if n < 2:
            raise ValueError("n must be >= 2")
        self.n = n
        self.epsilon = epsilon
        if k is None:
            k_min = n + math.ceil(math.log2(epsilon)) if epsilon < 1 else n
            k = max(2, min(n, k_min))
        if k > n:
            raise ValueError(f"k ({k}) > n ({n})")
        self.k = k
        self.n_qubits = 3 * n + 2

    @property
    def error_probability(self): return 2.0 ** (self.k - self.n)

    def _a(self, i): return i
    def _b(self, i): return self.n + i
    def _c(self, i): return 2 * self.n + i
    def _qbit(self): return 3 * self.n

    def _truncated_comparison(self, circ):
        n, k = self.n, self.k
        offset = n - k
        circ.append(X(self._c(offset)))
        for i in range(offset, n):
            a_i, b_i, c_i = self._a(i), self._b(i), self._c(i)
            circ.append(X(b_i))
            circ.append(CNOT(c_i, b_i))
            circ.append(CNOT(c_i, a_i))
            if i < n - 1:
                c_next = self._c(i + 1)
                circ.append(Toffoli(a_i, b_i, c_next))
                circ.append(CNOT(c_i, a_i))
                circ.append(CNOT(c_i, b_i))
                circ.append(X(b_i))
            else:
                qbit = self._qbit()
                circ.append(Toffoli(a_i, b_i, qbit))
                circ.append(CNOT(c_i, a_i))
                circ.append(CNOT(c_i, b_i))
                circ.append(X(b_i))

    def _full_conditional_subtraction(self, circ):
        n, qbit = self.n, self._qbit()
        for i in range(n):
            a_i, b_i = self._a(i), self._b(i)
            circ.append(Toffoli(qbit, b_i, a_i))
            if i < n - 1:
                circ.append(Toffoli(a_i, b_i, self._a(i + 1)))
                circ.append(CNOT(a_i, b_i))

    def _truncated_uncompute(self, circ):
        n, k = self.n, self.k
        offset = n - k
        for i in reversed(range(offset, n)):
            a_i, b_i, c_i = self._a(i), self._b(i), self._c(i)
            circ.append(X(b_i))
            circ.append(CNOT(c_i, b_i))
            circ.append(CNOT(c_i, a_i))
            if i < n - 1:
                c_next = self._c(i + 1)
                circ.append(Toffoli(a_i, b_i, c_next))
                circ.append(CNOT(c_i, a_i))
                circ.append(CNOT(c_i, b_i))
                circ.append(X(b_i))
            else:
                qbit = self._qbit()
                circ.append(Toffoli(a_i, b_i, qbit))
                circ.append(CNOT(c_i, a_i))
                circ.append(CNOT(c_i, b_i))
                circ.append(X(b_i))
        circ.append(X(self._c(offset)))

    def build(self) -> QuantumCircuit:
        circ = QuantumCircuit(self.n_qubits,
                              name=f"CompNSub_approx_n{self.n}_k{self.k}")
        self._truncated_comparison(circ)
        self._full_conditional_subtraction(circ)
        self._truncated_uncompute(circ)
        return circ

    def analytical_cost(self) -> dict:
        n, k = self.n, self.k
        return {
            "method":     "approx_truncated",
            "n_bits":     n,
            "k_bits":     k,
            "t_count":    4 * k - 3,
            "t_depth":    2 * math.ceil(math.log2(k)) + 2 if k > 1 else 2,
            "cnot_count": 6 * k - 3,
            "n_qubits":   self.n_qubits,
            "error_prob": self.error_probability,
        }


class EarlyStopCompNSub:
    """Early-stopping variant — same gate-per-step cost as exact; fewer steps."""

    def __init__(self, n: int, k: int):
        if k > n:
            raise ValueError("k > n")
        self.n, self.k = n, k
        self.n_qubits = 3 * n + 2

    @property
    def error_probability(self): return 2.0 ** (self.k - self.n)

    def analytical_cost(self) -> dict:
        n, k = self.n, self.k
        return {
            "method":     "early_stop",
            "n_bits":     n,
            "k_bits":     k,
            "t_count":    4 * n - 3,
            "t_depth":    2 * math.ceil(math.log2(n)) + 2 if n > 1 else 2,
            "cnot_count": 6 * n - 3,
            "n_qubits":   self.n_qubits,
            "error_prob": self.error_probability,
        }