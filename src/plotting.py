"""
plotting.py
─────────────────────────────────────────────────────────────────────────────
Visualization utilities for quantum division results.

Generates:
  1. T-count vs n-bits scaling plot (all methods)
  2. Accuracy vs T-savings scatter (Pareto frontier highlighted)
  3. Adaptive selector decision map
  4. Benchmark accuracy heatmap
"""

from __future__ import annotations
import math
import os
from typing import List, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib safety import
# ─────────────────────────────────────────────────────────────────────────────

def _check_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Scaling plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_scaling(scaling_rows: List[Dict],
                 output_path: str = "results/plots/scaling.png") -> str:
    """
    Plot T-count vs n-bits for all methods.
    """
    plt = _check_matplotlib()
    import matplotlib.pyplot as plt2
    import numpy as np

    fig, axes = plt2.subplots(1, 2, figsize=(14, 6))

    methods = sorted(set(r["method"] for r in scaling_rows))
    colours = plt2.cm.tab10.colors

    METHOD_STYLE = {
        "exact":           ("-",  "o",  2.5),
        "approx_k75":      ("--", "s",  1.8),
        "approx_k50":      (":",  "^",  1.8),
        "hybrid_best":     ("-.", "D",  1.8),
        "hybrid_fast":     ("--", "P",  1.8),
        "early_stop_m75":  ("--", "v",  1.8),
    }
    method_labels = {
        "exact":           "Exact (paper baseline)",
        "approx_k75":      "Approx k=75%n  (~26% save)",
        "approx_k50":      "Approx k=50%n  (~52% save)",
        "hybrid_best":     "Hybrid h=50%, k=75%  (best acc/cost)",
        "hybrid_fast":     "Hybrid h=25%, k=75%  (aggressive)",
        "early_stop_m75":  "Early-stop m=75%n  (25% save)",
    }

    # Group by method
    by_method: Dict[str, list] = {m: [] for m in methods}
    for row in scaling_rows:
        by_method.setdefault(row["method"], []).append(row)

    colours = plt2.cm.tab10.colors
    method_colour = {m: colours[i % len(colours)] for i, m in enumerate(by_method)}

    # — T-count plot ——————————————————————————————————————————————————————————
    ax1 = axes[0]
    for method, rows in by_method.items():
        rows_sorted = sorted(rows, key=lambda r: r["n_bits"])
        xs = [r["n_bits"] for r in rows_sorted]
        ys = [r["t_count"] for r in rows_sorted]
        ls, marker, lw = METHOD_STYLE.get(method, ("-", "o", 1.5))
        ax1.plot(xs, ys,
                 linestyle=ls, marker=marker, linewidth=lw,
                 color=method_colour[method],
                 label=method_labels.get(method, method))

    ax1.set_xlabel("Bit-width (n)", fontsize=13)
    ax1.set_ylabel("T-count", fontsize=13)
    ax1.set_title("T-count Scaling vs Bit-width", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # — T-savings (%) plot ————————————————————————————————————————————————————
    ax2 = axes[1]
    for method, rows in by_method.items():
        if method == "exact":
            continue
        rows_sorted = sorted(rows, key=lambda r: r["n_bits"])
        xs = [r["n_bits"] for r in rows_sorted]
        ys = [r["t_count_savings_%"] for r in rows_sorted]
        ls, marker, lw = METHOD_STYLE.get(method, ("-", "o", 1.5))
        ax2.plot(xs, ys,
                 linestyle=ls, marker=marker, linewidth=lw,
                 color=method_colour[method],
                 label=method_labels.get(method, method))

    ax2.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax2.set_xlabel("Bit-width (n)", fontsize=13)
    ax2.set_ylabel("T-count Saving (%)", fontsize=13)
    ax2.set_title("T-count Savings vs Paper Baseline", fontsize=14,
                  fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt2.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt2.savefig(output_path, dpi=150, bbox_inches="tight")
    plt2.close()
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pareto frontier plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto(pareto_rows: List[Dict], all_rows: List[Dict],
                n_bits: int,
                output_path: str = "results/plots/pareto.png") -> str:
    """
    Scatter plot: T-count vs Accuracy with Pareto frontier highlighted.
    """
    plt = _check_matplotlib()
    import matplotlib.pyplot as plt2

    fig, ax = plt2.subplots(figsize=(10, 7))

    method_colours = {
        "exact":           "blue",
        "approx_truncated": "red",
        "early_stop":      "green",
        "hybrid":          "purple",
    }

    # All points (background, semi-transparent)
    for row in all_rows:
        c = method_colours.get(row.get("method",""), "gray")
        ax.scatter(row["t_count"], row["accuracy"],
                   color=c, alpha=0.25, s=40, zorder=2)

    # Pareto points (highlighted)
    pareto_sorted = sorted(pareto_rows, key=lambda r: r["t_count"])
    px = [r["t_count"] for r in pareto_sorted]
    py = [r["accuracy"] for r in pareto_sorted]

    ax.plot(px, py, "k--", linewidth=1.5, zorder=3, label="Pareto frontier")
    ax.scatter(px, py, color="black", s=80, zorder=4, marker="*")

    # Annotate Pareto points
    for r in pareto_sorted:
        label = r.get("method", "")
        if r.get("k_bits"):
            label += f"\nk={r['k_bits']}"
        ax.annotate(label,
                    (r["t_count"], r["accuracy"]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, alpha=0.8)

    # Legend proxies
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=10, label=m)
        for m, c in method_colours.items()
    ]
    legend_elements.append(
        Line2D([0], [0], color="k", linewidth=1.5,
               linestyle="--", label="Pareto frontier")
    )
    ax.legend(handles=legend_elements, fontsize=9)

    ax.set_xlabel("T-count", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(f"Cost–Accuracy Trade-off (n={n_bits} bits)\n"
                 f"Pareto Frontier Highlighted",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.10)

    plt2.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt2.savefig(output_path, dpi=150, bbox_inches="tight")
    plt2.close()
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 3. Adaptive selector decision map
# ─────────────────────────────────────────────────────────────────────────────

def plot_selector_decisions(n_bits: int,
                             output_path: str = "results/plots/selector_map.png"
                             ) -> str:
    """
    Heatmap showing which circuit the adaptive selector chooses for
    different accuracy targets and T-count budgets.
    """
    plt = _check_matplotlib()
    import matplotlib.pyplot as plt2
    import numpy as np

    from .adaptive_selector import AdaptiveSelector, SelectionCriteria

    sel = AdaptiveSelector()
    exact_t = n_bits * (4 * n_bits - 3)

    acc_targets = [round(0.80 + i * 0.02, 2) for i in range(11)]   # 0.80–1.00
    budget_fracs = [round(0.2 + i * 0.1, 1) for i in range(9)]    # 0.2–1.0

    method_id = {"exact": 0, "approx_truncated": 1,
                 "early_stop": 2, "hybrid": 3}
    grid = np.zeros((len(acc_targets), len(budget_fracs)), dtype=int)
    method_at = {}

    for i, acc in enumerate(acc_targets):
        for j, frac in enumerate(budget_fracs):
            budget = int(exact_t * frac)
            crit = SelectionCriteria(
                n_bits=n_bits,
                accuracy_target=acc,
                t_count_budget=budget,
            )
            result = sel.select(crit)
            mid = method_id.get(result.engine_class, 3)
            grid[i, j] = mid
            method_at[(i, j)] = result.engine_class

    cmap = plt2.cm.get_cmap("Set1", 4)
    fig, ax = plt2.subplots(figsize=(12, 7))
    im = ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=3.5,
                   aspect="auto", origin="lower")

    ax.set_xticks(range(len(budget_fracs)))
    ax.set_xticklabels([f"{f*100:.0f}%" for f in budget_fracs])
    ax.set_yticks(range(len(acc_targets)))
    ax.set_yticklabels([f"{a:.2f}" for a in acc_targets])
    ax.set_xlabel("T-count Budget (% of exact)", fontsize=12)
    ax.set_ylabel("Accuracy Target", fontsize=12)
    ax.set_title(f"Adaptive Selector Decision Map (n={n_bits})",
                 fontsize=14, fontweight="bold")

    from matplotlib.patches import Patch
    legend = [
        Patch(color=cmap(0), label="Exact"),
        Patch(color=cmap(1), label="Approx Truncated"),
        Patch(color=cmap(2), label="Early Stop"),
        Patch(color=cmap(3), label="Hybrid"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9)
    plt2.colorbar(im, ax=ax, ticks=[0, 1, 2, 3],
                  label="Selected Method")

    plt2.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt2.savefig(output_path, dpi=150, bbox_inches="tight")
    plt2.close()
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 4. Benchmark accuracy bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_benchmark_accuracy(benchmark_rows: List[Dict],
                             output_path: str = "results/plots/accuracy.png"
                             ) -> str:
    """
    Bar chart comparing empirical accuracy of each method.
    """
    plt = _check_matplotlib()
    import matplotlib.pyplot as plt2
    import numpy as np

    labels = []
    accuracies = []
    t_counts = []

    for row in benchmark_rows:
        m = row["method"]
        suffix = ""
        if row.get("k_bits"):
            suffix = f"\nk={row['k_bits']}"
        elif row.get("m_steps"):
            suffix = f"\nm={row['m_steps']}"
        elif row.get("h_exact") and row.get("k_bits"):
            suffix = f"\nh={row['h_exact']},k={row['k_bits']}"
        labels.append(m + suffix)
        accuracies.append(row["accuracy"])

    x = np.arange(len(labels))
    colours = ["#2196F3" if a >= 0.99 else
               "#4CAF50" if a >= 0.95 else
               "#FF9800" if a >= 0.90 else
               "#F44336" for a in accuracies]

    fig, ax = plt2.subplots(figsize=(max(10, len(labels) * 1.2), 6))
    bars = ax.bar(x, accuracies, color=colours, edgecolor="white",
                  linewidth=1.5, width=0.7)

    ax.set_ylim(0, 1.08)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--",
               label="Perfect accuracy")
    ax.axhline(0.99, color="green", linewidth=1, linestyle=":",
               label="99% threshold")
    ax.axhline(0.95, color="orange", linewidth=1, linestyle=":",
               label="95% threshold")

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Empirical Accuracy", fontsize=13)
    ax.set_title("Empirical Accuracy by Division Method", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt2.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt2.savefig(output_path, dpi=150, bbox_inches="tight")
    plt2.close()
    return output_path