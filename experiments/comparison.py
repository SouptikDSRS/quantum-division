"""
comparison.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive comparison: Paper (exact) vs Our system (adaptive + approx) 
+ Hardware results (IBM Quantum). Optimized for limited free IBM Quantum time.

This script generates the main results of the paper:
  • Table 1: Resource comparison across bit-widths (incl. hardware)
  • Table 2: Cost-accuracy trade-off at n=16 (incl. hardware)
  • Table 3: Adaptive selector performance
  • Pareto frontier data
  • Scaling data (simulated only)

Run with:
  python -m experiments.comparison [--hardware] [--backend NAME] [--tests N] [--shots S] [--mitigate]
"""

import sys
import os
import json
import argparse
import numpy as np
from typing import Optional, List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Paper modules (simulation, adaptive, etc.)
from quantum_division.adaptive_selector import (
    AdaptiveSelector, SelectionCriteria, AppProfile
)
from quantum_division.division_engine import ExactDivisionEngine
from quantum_division.metrics import (
    full_comparison_table,
    scaling_analysis,
    print_summary,
    to_csv,
    to_latex_table,
)
from quantum_division.classical_simulator import DivisionSimulator

# Qiskit bridge (simulation + hardware)
from qiskit_impl.qiskit_bridge import QiskitDivisionRunner


# ─────────────────────────────────────────────────────────────────────────────
# Hardware caching (to avoid repeated IBM jobs)
# ─────────────────────────────────────────────────────────────────────────────
class HardwareCache:
    """Cache hardware results to disk and memory."""
    def __init__(self, backend: str, num_tests: int, shots: int, use_mitigation: bool = False):
        self.backend = backend
        self.num_tests = num_tests
        self.shots = shots
        self.use_mitigation = use_mitigation
        self.cache_file = "results/hardware_cache.json"
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_cache(self):
        os.makedirs("results", exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self._cache, f, indent=2)

    def get(self, n: int) -> Optional[Dict[str, Any]]:
        key = f"{n}_{self.backend}_{self.num_tests}_{self.shots}_{self.use_mitigation}"
        if key not in self._cache:
            print(f"  Running hardware for n={n} (this may take a minute)...")
            try:
                runner = QiskitDivisionRunner(
                    backend=self.backend,
                    use_mitigation=self.use_mitigation
                )
                metrics = runner.get_metrics_hardware(
                    n, num_tests=self.num_tests, shots=self.shots
                )
                # Quick validity check (ignore if T‑count zero or accuracy too low)
                if metrics.get("t_count", 0) == 0 or metrics.get("accuracy", 0) < 0.01:
                    print(f"    [!] Hardware results appear invalid (t_count={metrics.get('t_count')}, accuracy={metrics.get('accuracy')}). Skipping.")
                    self._cache[key] = None
                else:
                    print(f"    accuracy = {metrics['accuracy']:.6f}, t_count = {metrics['t_count']}")
                    self._cache[key] = metrics
                self._save_cache()
            except Exception as e:
                print(f"    [!] Hardware run failed: {e}")
                self._cache[key] = None
                self._save_cache()
        return self._cache[key]


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced table generation (includes hardware if requested)
# ─────────────────────────────────────────────────────────────────────────────
def full_comparison_table_with_hardware(
    n: int,
    hardware_metrics: Optional[Dict] = None
) -> List[Dict]:
    """Return rows from paper metrics plus optional hardware row."""
    rows = full_comparison_table(n)  # returns list of dicts

    if hardware_metrics:
        hw_row = {
            "method": "hardware_ibm",               # explicit tag
            "n_bits": hardware_metrics["n_bits"],
            "t_count": hardware_metrics["t_count"],
            "t_depth": hardware_metrics["t_depth"],
            "cnot_count": hardware_metrics["cnot_count"],
            "accuracy": hardware_metrics["accuracy"],
        }
        exact_t = ExactDivisionEngine(n).resources().t_count
        hw_row["t_count_savings_%"] = 100.0 * (1.0 - hw_row["t_count"] / max(1, exact_t))
        rows.append(hw_row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Table 1: Resource comparison across bit-widths
# ─────────────────────────────────────────────────────────────────────────────
def table1_resource_comparison(
    bit_widths: List[int],
    hardware_cache: Optional[HardwareCache] = None,
    hardware_bit_widths: Optional[List[int]] = None,  # which n to actually run hardware for
):
    print("\n" + "=" * 65)
    print("  TABLE 1: Resource Comparison (Paper vs Ours + Hardware)")
    print("=" * 65)

    all_rows = []
    for n in bit_widths:
        # Only request hardware if n is in the allowed list and cache exists
        hw_metrics = None
        if hardware_cache and hardware_bit_widths and n in hardware_bit_widths:
            hw_metrics = hardware_cache.get(n)
        rows = full_comparison_table_with_hardware(n, hw_metrics)

        # Keep only representative rows (best per method family)
        seen_methods = set()
        for row in sorted(rows, key=lambda r: -r["accuracy"]):
            method_key = row["method"]
            if method_key not in seen_methods:
                seen_methods.add(method_key)
                all_rows.append({**row, "n_bits": n})

    # Print condensed view
    print(f"\n  {'n':>4}  {'Method':<22} {'T-count':>8} {'Acc':>8} {'T-save%':>9}")
    print("  " + "-" * 55)
    for row in all_rows:
        print(f"  {row['n_bits']:>4}  {row['method']:<22} "
              f"{row['t_count']:>8} {row['accuracy']:>8.4f} "
              f"{row.get('t_count_savings_%', 0):>9.2f}%")

    # Save CSV and LaTeX
    os.makedirs("results/tables", exist_ok=True)
    csv_str = to_csv(all_rows)
    path = "results/tables/table1_resource_comparison.csv"
    with open(path, "w") as f:
        f.write(csv_str)
    print(f"\n  [✓] Saved → {path}")

    latex = to_latex_table(
        all_rows,
        columns=["n_bits", "method", "t_count", "t_depth",
                 "cnot_count", "accuracy", "t_count_savings_%"],
        caption="Quantum division resource comparison (paper vs our methods vs hardware)",
        label="tab:comparison",
    )
    tex_path = "results/tables/table1_resource_comparison.tex"
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"  [✓] LaTeX → {tex_path}")

    return all_rows


# ─────────────────────────────────────────────────────────────────────────────
# Table 2: Full cost-accuracy trade-off at n=16
# ─────────────────────────────────────────────────────────────────────────────
def table2_cost_accuracy_tradeoff(
    n: int = 16,
    hardware_metrics: Optional[Dict] = None
):
    print(f"\n{'='*65}")
    print(f"  TABLE 2: Full Cost-Accuracy Trade-off (n={n})")
    print("=" * 65)

    rows = full_comparison_table_with_hardware(n, hardware_metrics)

    # Safe call to print_summary (handles both signature styles)
    try:
        print_summary(n, rows)
    except TypeError:
        print_summary(n)

    os.makedirs("results/tables", exist_ok=True)
    csv_str = to_csv(rows)
    path = f"results/tables/table2_tradeoff_n{n}.csv"
    with open(path, "w") as f:
        f.write(csv_str)

    latex = to_latex_table(
        rows,
        columns=["method", "t_count", "t_depth", "cnot_count",
                 "accuracy", "t_count_savings_%", "cnot_savings_%"],
        caption=f"Full cost-accuracy trade-off for $n={n}$-bit division (hardware included if enabled)",
        label="tab:tradeoff",
    )
    tex_path = f"results/tables/table2_tradeoff_n{n}.tex"
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"  [✓] Saved → {path}, {tex_path}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Table 3: Adaptive selector – multiple profiles
# ─────────────────────────────────────────────────────────────────────────────
def table3_adaptive_selector(bit_widths: List[int] = None):
    if bit_widths is None:
        bit_widths = [8, 16, 32, 64]

    print(f"\n{'='*65}")
    print("  TABLE 3: Adaptive Selector – Profile-Based Decisions")
    print("=" * 65)

    sel = AdaptiveSelector()
    profiles = [
        AppProfile.EXACT,
        AppProfile.HIGH_PRECISION,
        AppProfile.STANDARD,
        AppProfile.FAST,
        AppProfile.ULTRA_FAST,
    ]

    all_rows = []
    print(f"\n  {'n':>4}  {'Profile':<18} {'Method':<22} "
          f"{'T-count':>8} {'Acc':>8} {'T-save%':>9}")
    print("  " + "-" * 72)

    for n in bit_widths:
        exact_t = ExactDivisionEngine(n).resources().t_count
        for profile in profiles:
            crit = SelectionCriteria(n_bits=n, profile=profile)
            result = sel.select(crit)
            r = result.resources
            savings = 100.0 * (1.0 - r.t_count / max(1, exact_t))

            print(f"  {n:>4}  {profile.value:<18} "
                  f"{result.engine_class:<22} {r.t_count:>8} "
                  f"{r.accuracy:>8.4f} {savings:>9.2f}%")

            all_rows.append({
                "n_bits":        n,
                "profile":       profile.value,
                "method":        result.engine_class,
                "t_count":       r.t_count,
                "t_depth":       r.t_depth,
                "cnot_count":    r.cnot_count,
                "accuracy":      round(r.accuracy, 6),
                "t_savings_%":   round(savings, 2),
                "reason":        result.selection_reason[:60],
                **r.extra,
            })

    os.makedirs("results/tables", exist_ok=True)
    path = "results/tables/table3_adaptive_selector.csv"
    with open(path, "w") as f:
        f.write(to_csv(all_rows))
    print(f"\n  [✓] Saved → {path}")
    return all_rows


# ─────────────────────────────────────────────────────────────────────────────
# Pareto frontier
# ─────────────────────────────────────────────────────────────────────────────
def compute_pareto(n: int = 16):
    print(f"\n{'='*65}")
    print(f"  PARETO FRONTIER (n={n})")
    print("=" * 65)

    sel = AdaptiveSelector()
    pareto = sel.pareto_frontier(n)

    print(f"\n  {'Method':<30} {'T-count':>8} {'Acc':>10}")
    print("  " + "-" * 52)
    for r in sorted(pareto, key=lambda x: x.t_count):
        print(f"  {r.method:<30} {r.t_count:>8} {r.accuracy:>10.6f}")

    rows = [{"method": r.method, "t_count": r.t_count,
              "t_depth": r.t_depth, "cnot_count": r.cnot_count,
              "accuracy": r.accuracy, **r.extra}
             for r in pareto]

    path = f"results/tables/pareto_n{n}.csv"
    os.makedirs("results/tables", exist_ok=True)
    with open(path, "w") as f:
        f.write(to_csv(rows))
    print(f"\n  [✓] Saved → {path}")
    return pareto, rows


# ─────────────────────────────────────────────────────────────────────────────
# Scaling analysis (simulated only)
# ─────────────────────────────────────────────────────────────────────────────
def run_scaling():
    print(f"\n{'='*65}")
    print("  SCALING ANALYSIS (T-count vs n) – Simulated only")
    print("=" * 65)

    from quantum_division.metrics import scaling_analysis
    bit_widths = [4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64]
    rows = scaling_analysis(bit_widths)   # uses corrected hybrid_best / hybrid_fast

    print(f"\n  {'n':>4}  {'Method':<22} {'T-count':>10} {'T-save%':>10} {'Acc':>8}")
    print("  " + "-" * 58)
    for row in rows:
        print(f"  {row['n_bits']:>4}  {row['method']:<22} "
              f"{row['t_count']:>10} {row['t_count_savings_%']:>10.2f}%"
              f" {row['accuracy']:>8.4f}")

    os.makedirs("results/tables", exist_ok=True)
    path = "results/tables/scaling_analysis.csv"
    with open(path, "w") as f:
        f.write(to_csv(rows))
    print(f"\n  [✓] Saved → {path}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Plotting (enhanced to include hardware points)
# ─────────────────────────────────────────────────────────────────────────────
def generate_plots(
    all_rows_16: List[Dict],
    pareto,
    scaling_rows: List[Dict],
    hardware_data_16: Optional[Dict] = None
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        print(f"\n{'='*65}")
        print("  GENERATING PLOTS")
        print("=" * 65)

        # Plot 1: T-count scaling (simulated only)
        from quantum_division.plotting import plot_scaling, plot_pareto
        path = plot_scaling(scaling_rows)
        print(f"  [✓] Scaling plot            → {path}")

        # Plot 2: Pareto frontier (simulated only)
        pareto_rows_16 = [
            {"method": r.method, "t_count": r.t_count,
             "accuracy": r.accuracy, **r.extra}
            for r in pareto
        ]
        path = plot_pareto(pareto_rows_16, all_rows_16, n_bits=16)
        print(f"  [✓] Pareto frontier plot    → {path}")

        # Plot 3: Accuracy vs T-savings scatter (all n, all methods + hardware)
        # First, collect all points from all bit-widths
        all_pts = []
        for n in [8, 12, 16, 24, 32]:
            rows = full_comparison_table(n)
            all_pts.extend(rows)

        # Add hardware point if present and valid (we already filtered)
        if hardware_data_16:
            exact_t_16 = ExactDivisionEngine(16).resources().t_count
            hw_point = {
                "method": "hardware_ibm",
                "t_count_savings_%": 100.0 * (1.0 - hardware_data_16["t_count"] / exact_t_16),
                "accuracy": hardware_data_16["accuracy"],
            }
            all_pts.append(hw_point)

        # Define colors/markers
        colours = {
            "exact": "#1565C0",
            "approx_truncated": "#C62828",
            "early_stop": "#6A1B9A",
            "hybrid": "#2E7D32",
            "hardware_ibm": "#F57C00",
        }
        markers = {
            "exact": "o",
            "approx_truncated": "s",
            "early_stop": "v",
            "hybrid": "D",
            "hardware_ibm": "*",
        }

        fig, ax = plt.subplots(figsize=(11, 7))
        plotted = set()
        for pt in all_pts:
            m = pt["method"]
            if m not in plotted:
                label = m
                plotted.add(m)
            else:
                label = None
            ax.scatter(pt["t_count_savings_%"], pt["accuracy"] * 100,
                       color=colours.get(m, "gray"),
                       marker=markers.get(m, "o"),
                       alpha=0.65, s=60, label=label)

        # Reference lines
        ax.axvline(0, color="gray", lw=1, ls="--", label="Exact baseline")
        ax.axhline(99.9, color="#43A047", lw=1.2, ls=":",  label="99.9% accuracy")
        ax.axhline(99.0, color="#FFA000", lw=1.2, ls=":",  label="99.0% accuracy")
        ax.axhline(95.0, color="#E53935", lw=1.2, ls=":",  label="95.0% accuracy")
        ax.axhspan(99.9, 101.5, alpha=0.06, color="green")

        # Annotate best hybrid point at n=16
        hybrid_pts = [r for r in all_pts if r.get("method") == "hybrid" and "t_count_savings_%" in r]
        if hybrid_pts:
            best = min(hybrid_pts, key=lambda r: r["t_count_savings_%"])
            ax.annotate("Best hybrid\n(n=16)",
                        (best["t_count_savings_%"], best["accuracy"] * 100),
                        xytext=(best["t_count_savings_%"] - 6, best["accuracy"] * 100 - 3),
                        arrowprops=dict(arrowstyle="->", color="black"),
                        fontsize=9)

        if hardware_data_16:
            ax.annotate(f"Hardware\n(n=16)",
                        (hw_point["t_count_savings_%"], hw_point["accuracy"] * 100),
                        xytext=(hw_point["t_count_savings_%"] + 5, hw_point["accuracy"] * 100 - 2),
                        arrowprops=dict(arrowstyle="->", color="black"),
                        fontsize=9)

        from matplotlib.patches import Patch
        legend_handles = [Patch(color=c, label=m) for m, c in colours.items()]
        ax.legend(handles=legend_handles, loc="lower left", fontsize=9)
        ax.set_xlabel("T-count Savings vs Exact Baseline (%)", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Accuracy vs T-count Savings  (all bit-widths, all methods)\n"
                     "Markers: ○ exact  □ approx  ▽ early-stop  ◇ hybrid  ★ hardware",
                     fontsize=13, fontweight="bold")
        ax.set_ylim(75, 102)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p3 = "results/plots/accuracy_vs_savings.png"
        os.makedirs("results/plots", exist_ok=True)
        plt.savefig(p3, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [✓] Accuracy vs savings     → {p3}")

        # Plot 4: Adaptive selector decision table (unchanged)
        sel = AdaptiveSelector()
        profiles_data = []
        for n in [8, 16, 32, 64]:
            for profile in AppProfile:
                if profile == AppProfile.CUSTOM:
                    continue
                r = sel.select(SelectionCriteria(n_bits=n, profile=profile))
                exact_t = ExactDivisionEngine(n).resources().t_count
                profiles_data.append({
                    "n": n, "profile": profile.value,
                    "method": r.engine_class,
                    "t_count": r.resources.t_count,
                    "accuracy": r.resources.accuracy,
                    "saving": 100 * (exact_t - r.resources.t_count) / exact_t,
                })

        ns   = sorted(set(d["n"] for d in profiles_data))
        profs = [p.value for p in AppProfile if p != AppProfile.CUSTOM]
        x = np.arange(len(profs))
        width = 0.2
        cmap4 = plt.cm.Blues

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for i, n in enumerate(ns):
            subset = [d for d in profiles_data if d["n"] == n]
            savings = [d["saving"] for d in subset]
            axes[0].bar(x + i * width, savings, width,
                        label=f"n={n}", color=cmap4((i + 2) / (len(ns) + 2)))

        axes[0].set_xticks(x + width * (len(ns) - 1) / 2)
        axes[0].set_xticklabels(profs, rotation=20, ha="right")
        axes[0].set_ylabel("T-count Savings (%)")
        axes[0].set_title("Adaptive Selector: T-count Savings by Profile",
                          fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis="y")
        axes[0].axhline(0, color="gray", lw=1)

        for i, n in enumerate(ns):
            subset = [d for d in profiles_data if d["n"] == n]
            accs = [d["accuracy"] * 100 for d in subset]
            axes[1].bar(x + i * width, accs, width,
                        label=f"n={n}", color=cmap4((i + 2) / (len(ns) + 2)))

        axes[1].set_xticks(x + width * (len(ns) - 1) / 2)
        axes[1].set_xticklabels(profs, rotation=20, ha="right")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_ylim(85, 101)
        axes[1].set_title("Adaptive Selector: Accuracy by Profile",
                          fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")
        axes[1].axhline(99, color="orange", lw=1, ls=":")

        plt.suptitle("Adaptive Selector Performance Across Profiles & Bit-widths",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        p4 = "results/plots/selector_profiles.png"
        plt.savefig(p4, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [✓] Selector profiles plot  → {p4}")

    except Exception as e:
        import traceback
        print(f"\n  [!] Plotting error: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run_all(
    hardware_enabled: bool = False,
    backend: Optional[str] = None,
    num_tests: int = 10,
    shots: int = 1024,
    mitigate: bool = False
):
    # Enforce backend when hardware is requested
    if hardware_enabled and not backend:
        print("Error: --backend must be specified when using --hardware")
        sys.exit(1)

    print("\n" + "█" * 65)
    print("  FULL COMPARISON: Paper vs Approximate + Adaptive System")
    if hardware_enabled:
        print(f"  WITH HARDWARE RUNS on {backend}")
        print(f"  (num_tests={num_tests}, shots={shots})")
        if mitigate:
            print("  Measurement error mitigation ENABLED")
    print("█" * 65)

    # Bit widths for tables
    bit_widths_table1 = [8, 12, 16, 24, 32]   # full range for simulation
    bit_widths_table3 = [8, 16, 32, 64]

    # Hardware cache (only used if hardware_enabled)
    hardware_cache = None
    hardware_bit_widths = None
    if hardware_enabled:
        hardware_cache = HardwareCache(backend, num_tests, shots, use_mitigation=mitigate)
        # Only run hardware for n=16 to save time
        hardware_bit_widths = [8]

    # Table 1: across bit widths
    table1_resource_comparison(
        bit_widths_table1,
        hardware_cache,
        hardware_bit_widths
    )

    # Table 2: n=16 with hardware (if enabled and valid)
    hw_16 = hardware_cache.get(16) if hardware_cache else None
    table2_cost_accuracy_tradeoff(16, hw_16)

    # Table 3: adaptive selector (simulated only)
    table3_adaptive_selector(bit_widths_table3)

    # Pareto frontier (simulated only)
    pareto, _ = compute_pareto(16)

    # Scaling analysis (simulated only)
    scaling_rows = run_scaling()

    # Prepare data for plots
    all_rows_16 = full_comparison_table(16)  # without hardware
    hardware_data_16 = hw_16 if hw_16 else None

    generate_plots(all_rows_16, pareto, scaling_rows, hardware_data_16)

    print("\n" + "█" * 65)
    print("  ALL EXPERIMENTS COMPLETE")
    print("█" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full comparison including optional hardware experiments.")
    parser.add_argument("--hardware", action="store_true", help="Enable hardware runs on IBM Quantum")
    parser.add_argument("--backend", type=str, default=None, help="IBM Quantum backend name (e.g., ibm_kingston)")
    parser.add_argument("--tests", type=int, default=10, help="Number of random test cases for hardware (default=10)")
    parser.add_argument("--shots", type=int, default=1024, help="Number of shots per circuit (default=1024)")
    parser.add_argument("--mitigate", action="store_true", help="Enable measurement error mitigation (requires qiskit-experiments)")
    args = parser.parse_args()

    run_all(
        hardware_enabled=args.hardware,
        backend=args.backend,
        num_tests=args.tests,
        shots=args.shots,
        mitigate=args.mitigate
    )