"""
Quantum Division - Quantum algorithms for division operations
"""

from .classical_simulator import DivisionSimulator
from .adaptive_selector import AdaptiveSelector, SelectionCriteria, AppProfile
from .division_engine import (
    DivisionEngine,
    ExactDivisionEngine,
    ApproxDivisionEngine,
    DivisionResult
)
from .comp_n_sub import CompNSub, ApproxCompNSub
from .pennylane_simulator import PennyLaneSimulator
from .metrics import scaling_analysis, full_comparison_table, to_csv, to_latex_table
from .llm_optimizer import run_ai_optimizer
from .quantum_gates import *
from .qiskit_from_engine import *
from .plotting import plot_scaling, plot_pareto

__version__ = "0.1.0"

__all__ = [
    "DivisionSimulator",
    "AdaptiveSelector", 
    "SelectionCriteria",
    "AppProfile",
    "DivisionEngine",
    "ExactDivisionEngine",
    "ApproxDivisionEngine",
    "DivisionResult",
    "CompNSub",
    "ApproxCompNSub",
    "PennyLaneSimulator",
    "scaling_analysis",
    "full_comparison_table",
    "to_csv",
    "to_latex_table",
    "run_ai_optimizer",
    "plot_scaling",
    "plot_pareto",
]