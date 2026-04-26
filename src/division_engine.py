"""
division_engine.py
─────────────────────────────────────────────────────────────────────────────
Quantum restoring-division circuits built on top of COMP-N-SUB.

ALGORITHM: RESTORING DIVISION
──────────────────────────────
Classical restoring division (n-bit quotient, n-bit divisor):

  remainder ← dividend
  for i from n-1 downto 0:
      remainder ← remainder - (divisor << i)
      if remainder < 0:
          remainder ← remainder + (divisor << i)   # restore
          quotient[i] ← 0
      else:
          quotient[i] ← 1

Each iteration is one COMP-N-SUB call.

Quantum implementation
──────────────────────
  • n COMP-N-SUB calls (one per quotient bit)
  • Each call operates on a 2n+2 qubit register
  • Quotient bits accumulate in dedicated output register
  • Remainder left in-place at the end

Resource scaling (exact):
  T-count  = n × (4n − 3) ≈ 4n²
  T-depth  = n × (2⌈log₂n⌉ + 2)
  CNOT     = n × (6n − 3) ≈ 6n²

YOUR EXTENSIONS
───────────────
1. ApproxDivisionEngine   – uses ApproxCompNSub (truncated comparison)
2. EarlyStopDivisionEngine – stops after k of n iterations
3. HybridDivisionEngine   – exact for MSB steps, approx for LSB steps
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .quantum_gates import QuantumCircuit, Gate, CNOT, X
from .comp_n_sub import CompNSub, ApproxCompNSub, EarlyStopCompNSub


# ─────────────────────────────────────────────────────────────────────────────
# Resource summary dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DivisionResources:
    method: str
    n_bits: int
    t_count: int
    t_depth: int
    cnot_count: int
    n_qubits: int
    accuracy: float          # 0.0 – 1.0  (1.0 = exact)
    error_prob_per_step: float = 0.0
    extra: dict = field(default_factory=dict)

    def cost_score(self, alpha: float = 0.5) -> float:
        """
        Weighted cost score for the adaptive selector.
        alpha controls T-count vs CNOT-count trade-off.
        Lower = cheaper.
        """
        return alpha * self.t_count + (1 - alpha) * self.cnot_count

    def __repr__(self) -> str:
        return (f"DivisionResources({self.method}, n={self.n_bits}, "
                f"T={self.t_count}, T-depth={self.t_depth}, "
                f"CNOT={self.cnot_count}, acc={self.accuracy:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Exact Division Engine (paper baseline)
# ─────────────────────────────────────────────────────────────────────────────

class ExactDivisionEngine:
    """
    Exact quantum restoring division using n COMP-N-SUB calls.
    Reproduces the paper's construction exactly.

    Qubit layout (3n + 2 qubits):
      [0 … n-1]        : dividend / remainder register (A)
      [n … 2n-1]       : divisor register (B, read-only)
      [2n … 3n-1]      : quotient output register
      [3n]             : carry ancilla
      [3n+1]           : quotient-bit ancilla (recycled each step)
    """

    def __init__(self, n: int):
        self.n = n
        self.n_qubits = 3 * n + 2

    def build(self) -> QuantumCircuit:
        """Build full exact division circuit."""
        n = self.n
        circ = QuantumCircuit(self.n_qubits, name=f"ExactDiv_n{n}")
        prim = CompNSub(n)

        for step in range(n):
            # Each step applies COMP-N-SUB and stores quotient bit
            step_circ = prim.build()
            circ += step_circ

            # Move quotient bit to output register
            qbit_src = 2 * n + 1   # ancilla output of COMP-N-SUB
            qbit_dst = 2 * n + step  # permanent quotient register
            circ.append(CNOT(qbit_src, qbit_dst))
            circ.append(X(qbit_src))   # reset ancilla

        return circ

    def resources(self) -> DivisionResources:
        """Analytical resource estimates."""
        n = self.n
        prim_cost = CompNSub(n).analytical_cost()
        return DivisionResources(
            method="exact",
            n_bits=n,
            t_count=n * prim_cost["t_count"],
            t_depth=n * prim_cost["t_depth"],
            cnot_count=n * prim_cost["cnot_count"] + 2 * n,
            n_qubits=self.n_qubits,
            accuracy=1.0,
            error_prob_per_step=0.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Approximate Division Engine  ← MAIN CONTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class ApproxDivisionEngine:
    """
    Approximate quantum restoring division using truncated-comparison
    COMP-N-SUB calls.

    All n division steps use ApproxCompNSub with the same k.
    The trade-off: lower gate cost at the price of a bounded accuracy loss.

    Accuracy model
    ──────────────
    Each of the n steps has per-step error probability p = 2^(k−n).
    The probability of getting the EXACT quotient is:
        P(exact) = (1 − p)^n  ≈ 1 − n·p  for small p

    For k = n−1, p = 0.5  → useless at large n
    For k = ceil(3n/4), p = 2^(−n/4) → excellent for n ≥ 8

    Expected relative error (quotient):
        E[|Q_approx − Q_exact| / Q_exact]  ≤  n · 2^(k−n)
    """

    def __init__(self, n: int, k: Optional[int] = None,
                 epsilon: float = 0.05):
        self.n = n
        self.prim = ApproxCompNSub(n, k=k, epsilon=epsilon)
        self.k = self.prim.k
        self.n_qubits = 3 * n + 2

    def build(self) -> QuantumCircuit:
        """Build approximate division circuit."""
        n = self.n
        circ = QuantumCircuit(self.n_qubits, name=f"ApproxDiv_n{n}_k{self.k}")

        for step in range(n):
            step_circ = self.prim.build()
            circ += step_circ
            qbit_src = 2 * n + 1
            qbit_dst = 2 * n + step
            circ.append(CNOT(qbit_src, qbit_dst))
            circ.append(X(qbit_src))

        return circ

    def resources(self) -> DivisionResources:
        n = self.n
        k = self.k
        # Corrected per-step cost: n steps × (4k-3) T-gates each
        # (comparison and subtraction are interleaved in one k-stage sweep)
        p        = self.prim.error_probability
        accuracy = max(0.0, (1.0 - p) ** n)
        td       = 2 * math.ceil(math.log2(k)) + 2 if k > 1 else 2
        return DivisionResources(
            method="approx_truncated",
            n_bits=n,
            t_count=n * (4 * k - 3),
            t_depth=n * td,
            cnot_count=n * (6 * k - 3),
            n_qubits=self.n_qubits,
            accuracy=accuracy,
            error_prob_per_step=p,
            extra={"k_bits": k},
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Early-Stopping Division Engine  ← SECOND CONTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopDivisionEngine:
    """
    Early-stopping restoring division.

    Only compute the top m quotient bits (most significant), skip the rest.
    This gives an approximate quotient with:
      • m / n fraction of the gate cost
      • Relative error ≤ 2^(−m)  (missing lower bits)

    Use case: applications where only the leading bits of the quotient are
    needed (e.g. floating-point exponent computation).
    """

    def __init__(self, n: int, m: int):
        """
        n : full bit-width
        m : number of quotient bits to compute (m ≤ n)
        """
        if m > n:
            raise ValueError("m cannot exceed n")
        self.n, self.m = n, m
        self.prim = CompNSub(n)   # exact primitive for each computed step
        self.n_qubits = 3 * n + 2

    def resources(self) -> DivisionResources:
        n, m = self.n, self.m
        prim_cost = self.prim.analytical_cost()
        relative_error = 2.0 ** (-m)   # missing lower m bits
        accuracy = 1.0 - relative_error
        return DivisionResources(
            method="early_stop",
            n_bits=n,
            t_count=m * prim_cost["t_count"],
            t_depth=m * prim_cost["t_depth"],
            cnot_count=m * prim_cost["cnot_count"] + 2 * m,
            n_qubits=self.n_qubits,
            accuracy=accuracy,
            error_prob_per_step=0.0,
            extra={"m_steps": m},
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hybrid Division Engine  ← THIRD CONTRIBUTION (most powerful)
# ─────────────────────────────────────────────────────────────────────────────

class HybridDivisionEngine:
    """
    Hybrid: exact for the first h steps, approximate for remaining n-h steps.

    MOTIVATION
    ──────────
    The most significant quotient bits matter most for accuracy.
    Using exact COMP-N-SUB for the top h bits guarantees correctness for
    the high-order part of the quotient, while approximate operations on
    the lower n-h bits incur only small relative error.

    Accuracy model
    ──────────────
    Error only comes from the approximate steps:
        Relative error ≤ (n - h) · 2^(k - n)  ·  2^(-h)
    The 2^(-h) factor reflects that lower quotient bits have smaller magnitude.

    Cost
    ────
    T-count = h · (4n−3)  +  (n−h) · [(4k−3) + 4n]
    """

    def __init__(self, n: int, h: int, k: Optional[int] = None,
                 epsilon: float = 0.05):
        """
        n       : bit-width
        h       : number of EXACT steps (most-significant bits)
        k       : comparison bits for approximate steps
        epsilon : error tolerance for approximate steps
        """
        if h > n:
            raise ValueError("h cannot exceed n")
        self.n, self.h = n, h
        self.exact_prim = CompNSub(n)
        self.approx_prim = ApproxCompNSub(n, k=k, epsilon=epsilon)
        self.k = self.approx_prim.k
        self.n_qubits = 3 * n + 2

    def resources(self) -> DivisionResources:
        n, h = self.n, self.h
        ec = self.exact_prim.analytical_cost()   # per-step exact: T = 4n-3
        # Approx per-step cost using corrected formula: T = 4k-3 (not (4k-3)+4n)
        k  = self.k

        t_exact  = h * ec["t_count"]                  # h × (4n-3)
        t_approx = (n - h) * (4 * k - 3)              # (n-h) × (4k-3)
        td_exact  = h * ec["t_depth"]
        td_approx = (n - h) * (2 * math.ceil(math.log2(k)) + 2 if k > 1 else 2)
        cnot_exact  = h * ec["cnot_count"]
        cnot_approx = (n - h) * (6 * k - 3)

        p = self.approx_prim.error_probability
        # Error only from approx steps, weighted by significance (2^-h)
        rel_error = (n - h) * p * (2.0 ** (-h))
        accuracy  = max(0.0, 1.0 - rel_error)

        return DivisionResources(
            method="hybrid",
            n_bits=n,
            t_count=t_exact + t_approx,
            t_depth=td_exact + td_approx,
            cnot_count=cnot_exact + cnot_approx,
            n_qubits=self.n_qubits,
            accuracy=accuracy,
            error_prob_per_step=p,
            extra={"h_exact_steps": h, "k_bits": self.k, "approx_steps": n - h},
        )
# ─────────────────────────────────────────────────────────────
# Unified Wrapper (for AI optimizer)
# ─────────────────────────────────────────────────────────────

class DivisionEngine:
    """
    Wrapper class to unify all division engines for AI optimizer.
    """

    def run_exact(self, n: int):
        engine = ExactDivisionEngine(n)
        return engine.resources().__dict__

    def run_hybrid(self, n: int, k: int, h: int):
        engine = HybridDivisionEngine(n=n, h=h, k=k)
        return engine.resources().__dict__

    def run_approx(self, n: int, k: int):
        engine = ApproxDivisionEngine(n=n, k=k)
        return engine.resources().__dict__

    def run_early_stop(self, n: int, m: int):
        engine = EarlyStopDivisionEngine(n=n, m=m)
        return engine.resources().__dict__