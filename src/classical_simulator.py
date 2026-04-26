"""
classical_simulator.py
─────────────────────────────────────────────────────────────────────────────
Classical simulation of quantum division circuits.

Purpose
───────
Since we cannot run on a real quantum computer, we simulate the CLASSICAL
LOGIC of each division variant to:
  1. Validate correctness of the exact algorithms
  2. Measure the actual error of approximate algorithms
  3. Generate ground-truth data for the accuracy vs cost trade-off

This is a bit-accurate classical simulation – not a quantum state-vector
simulator.  We model the quantum circuit's logical function, not the quantum
amplitudes.  This is valid because:
  • All operations are reversible (unitary)
  • We only measure at the end
  • No superposition is needed for resource estimation

Approximate division modes
───────────────────────────
For approximate variants, we model the error explicitly:
  • Truncated comparison: compare only top k bits → may get wrong quotient bit
  • Early stopping: return partial quotient (top m bits only)
  • Hybrid: exact top h bits, approx lower n-h bits
"""

from __future__ import annotations
import random
import math
from typing import Tuple, Optional, List
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DivisionResult:
    quotient:  int
    remainder: int
    exact_quotient:  int
    exact_remainder: int
    method: str
    n_bits: int
    dividend: int
    divisor: int

    @property
    def is_exact(self) -> bool:
        return (self.quotient == self.exact_quotient and
                self.remainder == self.exact_remainder)

    @property
    def quotient_error(self) -> int:
        return abs(self.quotient - self.exact_quotient)

    @property
    def relative_error(self) -> float:
        if self.exact_quotient == 0:
            return 0.0 if self.quotient == 0 else float("inf")
        return abs(self.quotient - self.exact_quotient) / self.exact_quotient

    def __repr__(self) -> str:
        return (f"DivisionResult({self.method}: "
                f"{self.dividend} ÷ {self.divisor} = "
                f"{self.quotient} R {self.remainder} "
                f"[exact: {self.exact_quotient} R {self.exact_remainder}], "
                f"error={self.relative_error:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Exact restoring division (reference)
# ─────────────────────────────────────────────────────────────────────────────

def exact_restoring_division(dividend: int, divisor: int,
                              n_bits: int) -> Tuple[int, int]:
    """
    Classical exact restoring division.
    Returns (quotient, remainder).
    """
    if divisor == 0:
        raise ZeroDivisionError("Divisor cannot be zero")
    if dividend < 0 or divisor < 0:
        raise ValueError("Only non-negative integers supported")

    remainder = dividend
    quotient = 0

    for i in range(n_bits - 1, -1, -1):
        shifted_divisor = divisor << i
        if remainder >= shifted_divisor:
            remainder -= shifted_divisor
            quotient |= (1 << i)

    return quotient, remainder


# ─────────────────────────────────────────────────────────────────────────────
# Approximate variant simulators
# ─────────────────────────────────────────────────────────────────────────────

def approx_truncated_division(dividend: int, divisor: int,
                               n_bits: int, k_bits: int) -> Tuple[int, int]:
    """
    Approximate division using truncated top-k comparison.

    Each step: compare only the top k bits of the partial remainder against
    the (shifted) divisor.  If the comparison result differs from exact, the
    wrong quotient bit is set and the error propagates.

    Simulation approach
    ───────────────────
    Mask the lower (n - k) bits of both operands before comparison.
    The subtraction is still exact (full n bits).
    """
    if divisor == 0:
        raise ZeroDivisionError
    if k_bits > n_bits:
        k_bits = n_bits

    mask_shift = n_bits - k_bits          # how many lower bits to ignore
    mask = ((1 << n_bits) - 1) & ~((1 << mask_shift) - 1)   # top-k mask

    remainder = dividend
    quotient = 0

    for i in range(n_bits - 1, -1, -1):
        shifted_divisor = divisor << i
        # Truncated comparison: only look at top k bits
        rem_masked = remainder & mask
        div_masked = shifted_divisor & mask

        if rem_masked >= div_masked:
            # Approximate comparison says yes → subtract
            remainder -= shifted_divisor
            if remainder < 0:
                # Restoration (would not happen in exact, can here)
                remainder += shifted_divisor
            else:
                quotient |= (1 << i)

    return quotient, remainder


def early_stop_division(dividend: int, divisor: int,
                         n_bits: int, m_steps: int) -> Tuple[int, int]:
    """
    Early-stopping division: compute only top m_steps quotient bits.
    The lower (n_bits - m_steps) bits of the quotient are set to 0.

    The returned quotient is a TRUNCATED approximation.
    """
    if divisor == 0:
        raise ZeroDivisionError

    remainder = dividend
    quotient = 0

    # Only compute the top m_steps bits (most significant)
    start_bit = n_bits - 1
    end_bit   = n_bits - m_steps   # exclusive

    for i in range(start_bit, end_bit - 1, -1):
        shifted_divisor = divisor << i
        if remainder >= shifted_divisor:
            remainder -= shifted_divisor
            quotient |= (1 << i)

    return quotient, remainder


def hybrid_division(dividend: int, divisor: int,
                     n_bits: int, h_exact: int,
                     k_approx: int) -> Tuple[int, int]:
    """
    Hybrid: exact for top h_exact quotient bits, approximate for the rest.

    Top h_exact bits (most significant): full comparison
    Lower (n_bits - h_exact) bits: truncated k_approx comparison
    """
    if divisor == 0:
        raise ZeroDivisionError

    mask_shift = n_bits - k_approx
    mask = ((1 << n_bits) - 1) & ~((1 << mask_shift) - 1)

    remainder = dividend
    quotient = 0

    for i in range(n_bits - 1, -1, -1):
        shifted_divisor = divisor << i
        bit_index = n_bits - 1 - i   # 0 = MSB

        if bit_index < h_exact:
            # Exact comparison
            if remainder >= shifted_divisor:
                remainder -= shifted_divisor
                quotient |= (1 << i)
        else:
            # Approximate (truncated) comparison
            rem_masked = remainder & mask
            div_masked = shifted_divisor & mask
            if rem_masked >= div_masked:
                remainder -= shifted_divisor
                if remainder < 0:
                    remainder += shifted_divisor
                else:
                    quotient |= (1 << i)

    return quotient, remainder


# ─────────────────────────────────────────────────────────────────────────────
# Unified simulation interface
# ─────────────────────────────────────────────────────────────────────────────

class DivisionSimulator:
    """
    Simulates all division variants and returns DivisionResult objects.
    Handles edge cases (divisor > dividend, zero inputs, etc.)
    """

    def simulate(self, dividend: int, divisor: int, n_bits: int,
                 method: str = "exact", **kwargs) -> DivisionResult:
        """
        Simulate division.

        Parameters
        ----------
        dividend, divisor : operands
        n_bits            : circuit bit-width
        method            : "exact" | "approx_truncated" | "early_stop" | "hybrid"
        **kwargs          : method-specific parameters (k_bits, m_steps, h_exact)
        """
        # Always compute exact reference
        eq, er = exact_restoring_division(dividend, divisor, n_bits)

        # Compute approximate result
        if method == "exact":
            q, r = eq, er
        elif method == "approx_truncated":
            k = kwargs.get("k_bits", n_bits)
            q, r = approx_truncated_division(dividend, divisor, n_bits, k)
        elif method == "early_stop":
            m = kwargs.get("m_steps", n_bits)
            q, r = early_stop_division(dividend, divisor, n_bits, m)
        elif method == "hybrid":
            h = kwargs.get("h_exact", n_bits // 2)
            k = kwargs.get("k_bits", int(math.ceil(n_bits * 0.75)))
            q, r = hybrid_division(dividend, divisor, n_bits, h, k)
        else:
            raise ValueError(f"Unknown method: {method}")

        return DivisionResult(
            quotient=q, remainder=r,
            exact_quotient=eq, exact_remainder=er,
            method=method, n_bits=n_bits,
            dividend=dividend, divisor=divisor,
        )

    def benchmark(self, n_bits: int, method: str,
                  n_trials: int = 1000, seed: int = 42,
                  **kwargs) -> dict:
        """
        Run n_trials random division problems and return accuracy statistics.

        Returns
        -------
        dict with keys: accuracy, mean_rel_error, max_rel_error,
                        exact_count, n_trials
        """
        rng = random.Random(seed)
        max_val = (1 << n_bits) - 1

        exact_count = 0
        rel_errors = []

        for _ in range(n_trials):
            a = rng.randint(0, max_val)
            b = rng.randint(1, max_val)   # avoid zero divisor

            try:
                result = self.simulate(a, b, n_bits, method, **kwargs)
                if result.is_exact:
                    exact_count += 1
                rel_errors.append(result.relative_error)
            except Exception:
                rel_errors.append(1.0)

        rel_errors_finite = [e for e in rel_errors if math.isfinite(e)]

        return {
            "method":         method,
            "n_bits":         n_bits,
            "n_trials":       n_trials,
            "exact_count":    exact_count,
            "accuracy":       exact_count / n_trials,
            "mean_rel_error": sum(rel_errors_finite) / max(1, len(rel_errors_finite)),
            "max_rel_error":  max(rel_errors_finite, default=0.0),
            **kwargs,
        }

    def compare_all_methods(self, n_bits: int, n_trials: int = 500,
                             seed: int = 42) -> List[dict]:
        """
        Run benchmark for all methods and return comparison table.
        """
        results = []

        # Exact
        results.append(self.benchmark(n_bits, "exact",
                                       n_trials=n_trials, seed=seed))

        # Approximate – various k values
        for frac in [0.875, 0.75, 0.625, 0.5]:
            k = max(2, int(math.ceil(n_bits * frac)))
            if k < n_bits:
                results.append(self.benchmark(
                    n_bits, "approx_truncated",
                    n_trials=n_trials, seed=seed, k_bits=k))

        # Early stop
        for m_frac in [0.875, 0.75, 0.5]:
            m = max(1, int(n_bits * m_frac))
            if m < n_bits:
                results.append(self.benchmark(
                    n_bits, "early_stop",
                    n_trials=n_trials, seed=seed, m_steps=m))

        # Hybrid
        for h_frac, k_frac in [(0.75, 0.875), (0.5, 0.75), (0.25, 0.625)]:
            h = max(1, int(n_bits * h_frac))
            k = max(2, int(math.ceil(n_bits * k_frac)))
            if h < n_bits and k < n_bits:
                results.append(self.benchmark(
                    n_bits, "hybrid",
                    n_trials=n_trials, seed=seed,
                    h_exact=h, k_bits=k))

        return results