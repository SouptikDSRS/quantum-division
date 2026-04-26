"""
adaptive_selector.py
─────────────────────────────────────────────────────────────────────────────
Adaptive Selector: automatically chooses the best division circuit for a
given input, accuracy requirement, and cost budget.

THIS IS YOUR MAIN RESEARCH CONTRIBUTION #2
──────────────────────────────────────────
The paper provides fixed circuits.  This selector provides a SMART SYSTEM
that picks the optimal circuit variant based on:

  1. Input magnitude  (small inputs → fewer iterations needed)
  2. Accuracy target  (high accuracy → more exact steps)
  3. Cost budget      (limited T-count → force approximation)
  4. Application mode (pre-defined profiles for different use-cases)

DECISION LOGIC (simplified):
────────────────────────────

  if accuracy_target == 1.0:
      → ExactDivisionEngine

  elif input is small (< 2^(n/2)):
      → EarlyStopDivisionEngine(m = effective_bits)

  elif accuracy_target >= 0.99:
      → HybridDivisionEngine(h = 3n/4, k = 7n/8)

  elif accuracy_target >= 0.95:
      → HybridDivisionEngine(h = n/2, k = 3n/4)

  else:
      → ApproxDivisionEngine(k = n/2)

The selector also exposes:
  • Pareto-optimal frontier enumeration
  • Cost-vs-accuracy trade-off table generation
  • Profile-based selection (FP, crypto, ML inference, etc.)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from .division_engine import (
    ExactDivisionEngine,
    ApproxDivisionEngine,
    EarlyStopDivisionEngine,
    HybridDivisionEngine,
    DivisionResources,
)


# ─────────────────────────────────────────────────────────────────────────────
# Application profiles
# ─────────────────────────────────────────────────────────────────────────────

class AppProfile(Enum):
    """
    Pre-defined profiles that encode typical accuracy/cost trade-offs
    for different application domains.
    """
    EXACT          = "exact"           # Shor's algorithm, cryptography
    HIGH_PRECISION = "high_precision"  # Scientific computing (99.9% accuracy)
    STANDARD       = "standard"        # General purpose (99% accuracy)
    FAST           = "fast"            # Signal processing (95% accuracy)
    ULTRA_FAST     = "ultra_fast"      # Rough estimation (90% accuracy)
    CUSTOM         = "custom"          # User-specified


@dataclass
class SelectionCriteria:
    """
    Input to the adaptive selector.

    Parameters
    ----------
    n_bits          : bit-width of operands
    dividend_value  : classical value of dividend (if known, for input-aware
                      optimisation). None = worst-case assumption.
    accuracy_target : minimum acceptable accuracy (0.0 – 1.0)
    t_count_budget  : maximum allowed T-count (None = unlimited)
    profile         : application profile (overrides accuracy_target if set)
    """
    n_bits: int
    dividend_value: Optional[int] = None
    accuracy_target: float = 0.99
    t_count_budget: Optional[int] = None
    profile: AppProfile = AppProfile.CUSTOM


@dataclass
class SelectionResult:
    """Output of the adaptive selector."""
    engine_class: str
    engine_kwargs: dict
    resources: DivisionResources
    selection_reason: str
    alternatives: List[DivisionResources]


# ─────────────────────────────────────────────────────────────────────────────
# Core selector
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveSelector:
    """
    Adaptive circuit selector for quantum division.

    Usage
    ─────
    >>> sel = AdaptiveSelector()
    >>> result = sel.select(SelectionCriteria(n_bits=16, accuracy_target=0.99))
    >>> print(result.resources)
    >>> engine = sel.build_engine(result)
    """

    # Profile → accuracy target map
    _PROFILE_ACCURACY = {
        AppProfile.EXACT:          1.000,
        AppProfile.HIGH_PRECISION: 0.999,
        AppProfile.STANDARD:       0.990,
        AppProfile.FAST:           0.950,
        AppProfile.ULTRA_FAST:     0.900,
    }

    def select(self, criteria: SelectionCriteria) -> SelectionResult:
        """
        Main entry point.  Returns the best SelectionResult for the criteria.
        """
        n = criteria.n_bits
        acc_target = self._resolve_accuracy(criteria)
        budget = criteria.t_count_budget

        # ── compute input-aware effective bit-width ────────────────────────
        eff_bits = self._effective_bits(n, criteria.dividend_value)

        # ── generate all candidate options ────────────────────────────────
        candidates = self._enumerate_candidates(n, eff_bits, acc_target)

        # ── filter by T-count budget ───────────────────────────────────────
        if budget is not None:
            candidates = [c for c in candidates if c.t_count <= budget]
        if not candidates:
            # relax budget slightly and pick cheapest
            candidates = self._enumerate_candidates(n, eff_bits, 0.0)
            candidates.sort(key=lambda r: r.t_count)
            candidates = candidates[:1]

        # ── filter by accuracy ─────────────────────────────────────────────
        valid = [c for c in candidates if c.accuracy >= acc_target]
        if not valid:
            # No candidate meets accuracy – pick best available
            valid = sorted(candidates, key=lambda r: -r.accuracy)[:1]

        # ── pick lowest cost among valid ───────────────────────────────────
        best = min(valid, key=lambda r: r.cost_score())

        reason = self._explain(best, acc_target, eff_bits, n)
        alternatives = [c for c in candidates if c.method != best.method]

        return SelectionResult(
            engine_class=best.method,
            engine_kwargs=self._resources_to_kwargs(best, n),
            resources=best,
            selection_reason=reason,
            alternatives=alternatives,
        )

    # ── resolution helpers ────────────────────────────────────────────────

    def _resolve_accuracy(self, criteria: SelectionCriteria) -> float:
        if criteria.profile != AppProfile.CUSTOM:
            return self._PROFILE_ACCURACY.get(criteria.profile,
                                               criteria.accuracy_target)
        return criteria.accuracy_target

    def _effective_bits(self, n: int, value: Optional[int]) -> int:
        """
        If the dividend value is known, we only need as many bits as it
        actually uses.  This enables early stopping for small inputs.
        """
        if value is None:
            return n
        if value == 0:
            return 1
        return min(n, int(math.floor(math.log2(value))) + 1)

    # ── candidate enumeration ─────────────────────────────────────────────

    def _enumerate_candidates(self, n: int, eff_bits: int,
                               acc_target: float) -> List[DivisionResources]:
        """Generate resource estimates for all applicable strategies."""
        candidates: List[DivisionResources] = []

        # 1. Exact
        candidates.append(ExactDivisionEngine(n).resources())

        # 2. Early stopping (if input is small)
        if eff_bits < n:
            for m in range(max(1, eff_bits - 1), n):
                r = EarlyStopDivisionEngine(n, m).resources()
                candidates.append(r)

        # 3. Approximate (various k values)
        for frac in [0.5, 0.625, 0.75, 0.875]:
            k = max(2, int(math.ceil(n * frac)))
            if k < n:
                r = ApproxDivisionEngine(n, k=k).resources()
                candidates.append(r)

        # 4. Hybrid (various h/k combos)
        for h_frac in [0.25, 0.5, 0.75]:
            for k_frac in [0.625, 0.75, 0.875]:
                h = max(1, int(n * h_frac))
                k = max(2, int(math.ceil(n * k_frac)))
                if k < n and h < n:
                    r = HybridDivisionEngine(n, h, k=k).resources()
                    candidates.append(r)

        # De-duplicate by method label, keep best per method
        seen: dict = {}
        for r in candidates:
            key = r.method + str(r.extra)
            if key not in seen or seen[key].t_count > r.t_count:
                seen[key] = r
        return list(seen.values())

    def _resources_to_kwargs(self, r: DivisionResources, n: int) -> dict:
        """Convert a DivisionResources back to engine constructor kwargs."""
        method = r.method
        if method == "exact":
            return {"n": n}
        elif method == "approx_truncated":
            return {"n": n, "k": r.extra.get("k_bits")}
        elif method == "early_stop":
            return {"n": n, "m": r.extra.get("m_steps", n)}
        elif method == "hybrid":
            return {"n": n, "h": r.extra.get("h_exact_steps"),
                    "k": r.extra.get("k_bits")}
        return {"n": n}

    def _explain(self, r: DivisionResources, acc_target: float,
                 eff_bits: int, n: int) -> str:
        method = r.method
        if method == "exact":
            return "Exact division selected: accuracy requirement is 1.0 or very high."
        elif method == "approx_truncated":
            k = r.extra.get("k_bits", "?")
            return (f"Approximate (truncated top-{k}/{n} bits) selected: "
                    f"accuracy target {acc_target:.3f} met with "
                    f"{r.accuracy:.4f}, saving "
                    f"{100*(1 - r.t_count / ExactDivisionEngine(n).resources().t_count):.1f}% "
                    f"T-gates.")
        elif method == "early_stop":
            m = r.extra.get("m_steps", "?")
            return (f"Early-stop division ({m}/{n} steps) selected: "
                    f"input uses only ~{eff_bits} bits, "
                    f"remaining steps are unnecessary.")
        elif method == "hybrid":
            h = r.extra.get("h_exact_steps", "?")
            k = r.extra.get("k_bits", "?")
            return (f"Hybrid division selected: exact for top {h} bits, "
                    f"approximate (k={k}) for lower {n-h} bits. "
                    f"Accuracy={r.accuracy:.4f}, best cost-accuracy balance.")
        return f"Selected {method}."

    # ── build helpers ─────────────────────────────────────────────────────

    def build_engine(self, result: SelectionResult):
        """Instantiate the chosen engine from a SelectionResult."""
        method = result.engine_class
        kw = result.engine_kwargs
        if method == "exact":
            return ExactDivisionEngine(**kw)
        elif method == "approx_truncated":
            return ApproxDivisionEngine(**kw)
        elif method == "early_stop":
            return EarlyStopDivisionEngine(**kw)
        elif method == "hybrid":
            return HybridDivisionEngine(**kw)
        return ExactDivisionEngine(n=kw["n"])

    # ── Pareto frontier ───────────────────────────────────────────────────

    def pareto_frontier(self, n: int) -> List[DivisionResources]:
        """
        Compute the Pareto-optimal frontier of (T-count, accuracy) for
        bit-width n.

        A solution is Pareto-optimal if no other solution is both cheaper
        AND more accurate.
        """
        all_candidates = self._enumerate_candidates(n, n, acc_target=0.0)

        # Sort by T-count ascending
        all_candidates.sort(key=lambda r: r.t_count)

        pareto: List[DivisionResources] = []
        best_acc = -1.0
        for r in all_candidates:
            if r.accuracy > best_acc:
                pareto.append(r)
                best_acc = r.accuracy

        return pareto

    # ── trade-off table ───────────────────────────────────────────────────

    def tradeoff_table(self, n: int) -> List[dict]:
        """
        Generate a full cost-vs-accuracy trade-off table for n-bit division.
        Suitable for direct export to CSV / LaTeX.
        """
        all_candidates = self._enumerate_candidates(n, n, acc_target=0.0)
        exact_t = ExactDivisionEngine(n).resources().t_count

        rows = []
        for r in sorted(all_candidates, key=lambda x: -x.accuracy):
            rows.append({
                "method":          r.method,
                "n_bits":          r.n_bits,
                "t_count":         r.t_count,
                "t_depth":         r.t_depth,
                "cnot_count":      r.cnot_count,
                "n_qubits":        r.n_qubits,
                "accuracy":        round(r.accuracy, 6),
                "t_savings_%":     round(100 * (1 - r.t_count / exact_t), 2),
                **r.extra,
            })
        return rows