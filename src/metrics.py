"""
metrics.py
─────────────────────────────────────────────────────────────────────────────
Evaluation metrics and reporting utilities.

Computes, formats, and exports:
  • Gate count comparisons (T, CNOT, total)
  • Circuit depth metrics
  • Accuracy statistics
  • Cost reduction percentages vs paper baseline
  • LaTeX table generation
  • CSV export
"""

from __future__ import annotations
import csv
import io
import math
from typing import List, Dict, Optional

from .division_engine import (
    ExactDivisionEngine,
    ApproxDivisionEngine,
    EarlyStopDivisionEngine,
    HybridDivisionEngine,
    DivisionResources,
)
from .adaptive_selector import AdaptiveSelector


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_savings(r: DivisionResources, baseline: DivisionResources) -> dict:
    """
    Compute percentage savings vs baseline.
    """
    def pct(new, old):
        if old == 0:
            return 0.0
        return 100.0 * (old - new) / old

    return {
        "t_count_savings_%":  round(pct(r.t_count, baseline.t_count), 2),
        "t_depth_savings_%":  round(pct(r.t_depth, baseline.t_depth), 2),
        "cnot_savings_%":     round(pct(r.cnot_count, baseline.cnot_count), 2),
        "accuracy_loss_%":    round(100.0 * (1.0 - r.accuracy), 4),
        "accuracy":           round(r.accuracy, 6),
    }


def full_comparison_table(n_bits: int) -> List[Dict]:
    """
    Generate a comprehensive comparison table for all methods at a given
    bit-width.  This is the main results table for the paper.
    """
    baseline = ExactDivisionEngine(n_bits).resources()
    sel = AdaptiveSelector()
    rows = sel.tradeoff_table(n_bits)

    enriched = []
    for row in rows:
        # Create a minimal DivisionResources from the row dict
        r = DivisionResources(
            method=row["method"],
            n_bits=row["n_bits"],
            t_count=row["t_count"],
            t_depth=row["t_depth"],
            cnot_count=row["cnot_count"],
            n_qubits=row["n_qubits"],
            accuracy=row["accuracy"],
        )
        savings = compute_savings(r, baseline)
        enriched.append({**row, **savings})

    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# Scaling analysis
# ─────────────────────────────────────────────────────────────────────────────

def scaling_analysis(bit_widths: List[int],
                     methods: Optional[List[str]] = None) -> List[Dict]:
    """
    Compute resource costs for all methods across multiple bit-widths.
    Used to generate scaling plots.

    Method labels
    ─────────────
    exact          : paper baseline — n steps × (4n-3) T-gates
    approx_k75     : truncated comparison k=75%n — ~26% T-savings, ~95% acc
    approx_k50     : truncated comparison k=50%n — ~52% T-savings, ~94% acc
    hybrid_best    : h=50%n exact + 50%n approx(k=75%n) — best acc/cost ratio
    hybrid_fast    : h=25%n exact + 75%n approx(k=75%n) — aggressive savings
    early_stop_m75 : stop after 75% of iterations — 25% T-savings
    """
    if methods is None:
        methods = ["exact", "approx_k75", "approx_k50",
                   "hybrid_best", "hybrid_fast", "early_stop_m75"]

    rows = []
    for n in bit_widths:
        exact = ExactDivisionEngine(n).resources()

        for method in methods:
            if method == "exact":
                r = exact

            elif method == "approx_k75":
                k = max(2, int(math.ceil(n * 0.75)))
                r = ApproxDivisionEngine(n, k=k).resources()

            elif method == "approx_k50":
                k = max(2, int(math.ceil(n * 0.50)))
                r = ApproxDivisionEngine(n, k=k).resources()

            elif method == "hybrid_best":
                # 50% exact steps + 50% approx steps at k=75%n
                # Best accuracy-per-T-gate ratio
                h = max(1, int(n * 0.50))
                k = max(2, int(math.ceil(n * 0.75)))
                r = HybridDivisionEngine(n, h, k=k).resources()

            elif method == "hybrid_fast":
                # 25% exact steps + 75% approx steps at k=75%n
                # More aggressive savings with good accuracy on MSBs
                h = max(1, int(n * 0.25))
                k = max(2, int(math.ceil(n * 0.75)))
                r = HybridDivisionEngine(n, h, k=k).resources()

            elif method == "early_stop_m75":
                m = max(1, int(n * 0.75))
                r = EarlyStopDivisionEngine(n, m).resources()

            else:
                continue

            savings = compute_savings(r, exact)
            rows.append({
                "n_bits":  n,
                "method":  method,
                "t_count": r.t_count,
                "t_depth": r.t_depth,
                "cnot_count": r.cnot_count,
                "accuracy": r.accuracy,
                **savings,
            })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table generator
# ─────────────────────────────────────────────────────────────────────────────

def to_latex_table(rows: List[Dict],
                   columns: Optional[List[str]] = None,
                   caption: str = "Resource comparison",
                   label: str = "tab:resource_comparison") -> str:
    """
    Generate a LaTeX table from a list of row dicts.
    """
    if not rows:
        return ""

    if columns is None:
        columns = ["method", "n_bits", "t_count", "t_depth",
                   "cnot_count", "accuracy", "t_count_savings_%"]

    col_headers = {
        "method":            r"Method",
        "n_bits":            r"$n$",
        "t_count":           r"T-count",
        "t_depth":           r"T-depth",
        "cnot_count":        r"CNOT",
        "n_qubits":          r"Qubits",
        "accuracy":          r"Accuracy",
        "t_count_savings_%": r"T-saving (\%)",
        "t_depth_savings_%": r"Depth-saving (\%)",
        "cnot_savings_%":    r"CNOT-saving (\%)",
        "accuracy_loss_%":   r"Acc. loss (\%)",
        "k_bits":            r"$k$",
        "h_exact_steps":     r"$h$",
        "m_steps":           r"$m$",
    }

    n_cols = len(columns)
    col_fmt = "l" + "r" * (n_cols - 1)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + col_fmt + "}",
        r"\toprule",
        " & ".join(col_headers.get(c, c) for c in columns) + r" \\",
        r"\midrule",
    ]

    prev_method = None
    for row in rows:
        # Add a midrule between method groups
        if row.get("method") != prev_method and prev_method is not None:
            lines.append(r"\midrule")
        prev_method = row.get("method")

        cells = []
        for c in columns:
            val = row.get(c, "—")
            if isinstance(val, float):
                if c == "accuracy":
                    cells.append(f"{val:.4f}")
                elif "%" in c or "savings" in c or "loss" in c:
                    cells.append(f"{val:.2f}")
                else:
                    cells.append(f"{val:.3f}")
            elif val is None:
                cells.append("—")
            else:
                cells.append(str(val))
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table}",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def to_csv(rows: List[Dict]) -> str:
    """Convert a list of row dicts to CSV string."""
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()),
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(n_bits: int) -> None:
    """Print a formatted summary table for a given bit-width."""
    print(f"\n{'='*70}")
    print(f"  Quantum Division Resource Summary  (n = {n_bits} bits)")
    print(f"{'='*70}")

    baseline = ExactDivisionEngine(n_bits).resources()
    header = f"{'Method':<28} {'T-count':>8} {'T-depth':>8} {'CNOT':>8} {'Acc':>8} {'T-save%':>8}"
    print(header)
    print("-" * 70)

    rows = full_comparison_table(n_bits)
    shown = set()
    for row in sorted(rows, key=lambda r: -r["accuracy"]):
        key = row["method"] + str(row.get("k_bits","")) + str(row.get("h_exact_steps",""))
        if key in shown:
            continue
        shown.add(key)

        label = row["method"]
        if row.get("k_bits"):
            label += f"(k={row['k_bits']})"
        if row.get("h_exact_steps"):
            label += f"(h={row['h_exact_steps']})"

        print(f"{label:<28} {row['t_count']:>8} {row['t_depth']:>8} "
              f"{row['cnot_count']:>8} {row['accuracy']:>8.4f} "
              f"{row['t_count_savings_%']:>8.2f}%")

    print("-" * 70)
    print(f"Baseline (exact): T-count={baseline.t_count}, "
          f"T-depth={baseline.t_depth}, CNOT={baseline.cnot_count}")
    print()