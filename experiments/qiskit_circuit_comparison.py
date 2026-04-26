"""
qiskit_real_experiment.py
────────────────────────────────────────────────────────────
Real Qiskit-based implementation of COMP-N-SUB division.

Features:
• Exact vs Approximate division circuits
• Real simulation using Aer
• Circuit visualization (saved as images)
• Metrics: depth, gate count

Run:
    python -m experiments.qiskit_real_experiment
"""

import os
import matplotlib
matplotlib.use("Agg")  # Fix Qt issue (no GUI required)

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ==========================================================
# CONFIGURATION
# ==========================================================

RESULT_DIR = "results/qiskit"
os.makedirs(RESULT_DIR, exist_ok=True)


# ==========================================================
# CORE BUILDING BLOCK: COMP-N-SUB (REAL CIRCUIT)
# ==========================================================

def comp_n_sub_qiskit(n: int, approx: bool = False, k: int = None) -> QuantumCircuit:
    """
    Construct COMP-N-SUB circuit in Qiskit.

    Parameters:
        n       : bit-width
        approx  : use approximation (truncate comparison)
        k       : number of MSBs to compare (if approx)

    Returns:
        QuantumCircuit
    """

    total_qubits = 3 * n + 2
    qc = QuantumCircuit(total_qubits, name="COMP_N_SUB")

    # Registers
    carry = [2 * n + i for i in range(n)]
    qbit = 3 * n + 1

    # Initialize carry
    qc.x(carry[0])

    # Define comparison range
    if approx:
        k = k or (n // 2)
        start = n - k
    else:
        start = 0

    # -------------------------
    # COMPARISON LOGIC
    # -------------------------
    for i in range(start, n):
        a = i
        b = n + i

        qc.x(b)
        qc.cx(a, b)

        if i < n - 1:
            qc.ccx(carry[i], b, carry[i + 1])
        else:
            qc.ccx(carry[i], b, qbit)

        qc.cx(a, b)
        qc.x(b)

    # -------------------------
    # CONDITIONAL SUBTRACTION
    # -------------------------
    for i in range(n):
        qc.ccx(qbit, n + i, i)

    return qc


# ==========================================================
# FULL DIVISION CIRCUITS
# ==========================================================

def build_exact_division(n: int) -> QuantumCircuit:
    """Exact division circuit"""
    qc = QuantumCircuit(3 * n + 2, name="Exact_Division")

    for _ in range(n):
        qc.compose(comp_n_sub_qiskit(n, approx=False), inplace=True)

    return qc


def build_approx_division(n: int, k: int) -> QuantumCircuit:
    """Approximate division circuit"""
    qc = QuantumCircuit(3 * n + 2, name="Approx_Division")

    for _ in range(n):
        qc.compose(comp_n_sub_qiskit(n, approx=True, k=k), inplace=True)

    return qc


# ==========================================================
# SIMULATION
# ==========================================================

def simulate_circuit(qc: QuantumCircuit, name: str):
    """Run circuit on Aer simulator"""

    simulator = AerSimulator()

    qc_meas = qc.copy()
    qc_meas.measure_all()

    compiled = transpile(qc_meas, simulator)
    result = simulator.run(compiled, shots=1024).result()

    counts = result.get_counts()

    print(f"\n📊 Simulation Results → {name}")
    print(counts)

    return counts


# ==========================================================
# VISUALIZATION
# ==========================================================

def save_circuit_diagram(qc: QuantumCircuit, filename: str):
    """Save circuit diagram as image"""
    path = os.path.join(RESULT_DIR, filename)
    qc.draw("mpl", filename=path)
    print(f"🖼 Circuit saved → {path}")


# ==========================================================
# MAIN EXPERIMENT PIPELINE
# ==========================================================

def run_qiskit_experiment(n: int = 4, k: int = 2):
    """
    Run full experiment:
    • Build circuits
    • Save diagrams
    • Simulate
    • Print metrics
    """

    print("\n" + "=" * 60)
    print("🚀 REAL QUANTUM EXPERIMENT (QISKIT)")
    print("=" * 60)

    # Build circuits
    qc_exact = build_exact_division(n)
    qc_approx = build_approx_division(n, k)

    # Save diagrams
    save_circuit_diagram(qc_exact, "exact_division.png")
    save_circuit_diagram(qc_approx, "approx_division.png")

    # Simulate
    counts_exact = simulate_circuit(qc_exact, "Exact Division")
    counts_approx = simulate_circuit(qc_approx, "Approx Division")

    # Metrics
    print("\n📈 Circuit Metrics")
    print("-" * 40)
    print(f"Exact   → Depth: {qc_exact.depth():<5} Gates: {qc_exact.size()}")
    print(f"Approx  → Depth: {qc_approx.depth():<5} Gates: {qc_approx.size()}")

    return {
        "exact_counts": counts_exact,
        "approx_counts": counts_approx,
    }


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    run_qiskit_experiment(n=4, k=2)