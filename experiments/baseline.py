"""
baseline.py
─────────────────────────────────────────────────────────────────────────────
Baseline experiment: reproduce the paper's exact circuit resource counts.

This script validates that our ExactDivisionEngine matches the paper's
reported T-count, T-depth, and CNOT count formulas.

Paper's formulas (Table I):
  COMP-N-SUB (n bits):
    T-count  = 4n − 3
    T-depth  = 2⌈log₂n⌉ + 2
    CNOT     = 6n − 3

  Full division (n bits, n steps):
    T-count  = n(4n − 3)
    T-depth  = n(2⌈log₂n⌉ + 2)
    CNOT     = n(6n − 3)

Run with:
  python -m experiments.baseline
"""

import sys
import os
import math

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quantum_division.comp_n_sub import CompNSub
from quantum_division.division_engine import ExactDivisionEngine
from quantum_division.metrics import to_csv, to_latex_table


def paper_comp_n_sub_cost(n: int) -> dict:
    """Paper's analytically-reported COMP-N-SUB costs."""
    return {
        "t_count":    4 * n - 3,
        "t_depth":    2 * math.ceil(math.log2(n)) + 2 if n > 1 else 2,
        "cnot_count": 6 * n - 3,
    }


def paper_division_cost(n: int) -> dict:
    """Paper's full division circuit costs (n steps)."""
    c = paper_comp_n_sub_cost(n)
    return {
        "t_count":    n * c["t_count"],
        "t_depth":    n * c["t_depth"],
        "cnot_count": n * c["cnot_count"],
    }


def run_baseline():
    print("=" * 65)
    print("  BASELINE: Paper-Matching Exact Division Resource Counts")
    print("=" * 65)

    bit_widths = [4, 8, 12, 16, 24, 32, 64]

    # ── COMP-N-SUB table ───────────────────────────────────────────────────
    print("\n[1] COMP-N-SUB costs (matches paper Table I):\n")
    print(f"  {'n':>4}  {'T-count(paper)':>16} {'T-count(ours)':>14} "
          f"{'CNOT(paper)':>12} {'CNOT(ours)':>11}  {'Match?':>6}")
    print("  " + "-" * 67)

    comp_rows = []
    all_match = True
    for n in bit_widths:
        prim = CompNSub(n)
        ours = prim.analytical_cost()
        paper = paper_comp_n_sub_cost(n)

        tc_match = ours["t_count"] == paper["t_count"]
        cn_match = ours["cnot_count"] == paper["cnot_count"]
        match = "✓" if (tc_match and cn_match) else "✗"
        if not (tc_match and cn_match):
            all_match = False

        print(f"  {n:>4}  {paper['t_count']:>16} {ours['t_count']:>14} "
              f"{paper['cnot_count']:>12} {ours['cnot_count']:>11}  {match:>6}")

        comp_rows.append({
            "n_bits": n,
            "t_count_paper": paper["t_count"],
            "t_count_ours":  ours["t_count"],
            "cnot_paper":    paper["cnot_count"],
            "cnot_ours":     ours["cnot_count"],
            "t_depth_paper": paper["t_depth"],
            "t_depth_ours":  ours["t_depth"],
            "match":         tc_match and cn_match,
        })

    print(f"\n  Overall match: {'✓ ALL CORRECT' if all_match else '✗ MISMATCH FOUND'}")

    # ── Full division table ─────────────────────────────────────────────────
    print("\n[2] Full division circuit costs (n iterations):\n")
    print(f"  {'n':>4}  {'T-count(paper)':>16} {'T-count(ours)':>14} "
          f"{'T-depth(paper)':>15} {'T-depth(ours)':>14}")
    print("  " + "-" * 67)

    div_rows = []
    for n in bit_widths:
        eng = ExactDivisionEngine(n)
        ours = eng.resources()
        paper = paper_division_cost(n)
        paper_td = paper_comp_n_sub_cost(n)["t_depth"] * n

        print(f"  {n:>4}  {paper['t_count']:>16} {ours.t_count:>14} "
              f"  {paper_td:>15} {ours.t_depth:>14}")
        div_rows.append({
            "n_bits":          n,
            "t_count_paper":   paper["t_count"],
            "t_count_ours":    ours.t_count,
            "t_depth_paper":   paper_td,
            "t_depth_ours":    ours.t_depth,
            "cnot_paper":      paper["cnot_count"],
            "cnot_ours":       ours.cnot_count,
        })

    # ── Save outputs ────────────────────────────────────────────────────────
    os.makedirs("results/tables", exist_ok=True)

    csv_path = "results/tables/baseline_comp_n_sub.csv"
    with open(csv_path, "w") as f:
        f.write(to_csv(comp_rows))
    print(f"\n  [✓] COMP-N-SUB table saved → {csv_path}")

    csv_path2 = "results/tables/baseline_division.csv"
    with open(csv_path2, "w") as f:
        f.write(to_csv(div_rows))
    print(f"  [✓] Division table saved   → {csv_path2}")

    latex = to_latex_table(
        div_rows,
        columns=["n_bits", "t_count_paper", "t_count_ours",
                 "t_depth_paper", "t_depth_ours", "cnot_paper", "cnot_ours"],
        caption="Exact division circuit costs vs paper baseline",
        label="tab:baseline",
    )
    tex_path = "results/tables/baseline_division.tex"
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"  [✓] LaTeX table saved      → {tex_path}")

    print()
    return div_rows


if __name__ == "__main__":
    run_baseline()