"""
app.py
─────────────────────────────────────────────────────────────────────────────
Flask web application for the Quantum Division Explorer.

Routes
──────
GET  /                → main page (input form)
POST /compute         → run simulation, return results as JSON
GET  /pareto?n=<int>  → Pareto frontier data for n bits
GET  /scaling         → scaling analysis data across bit-widths
GET  /tradeoff?n=<int>→ full cost-accuracy trade-off table
"""

import os
import math
from flask import Flask, render_template, request, jsonify

from src.classical_simulator import DivisionSimulator
from src.adaptive_selector import AdaptiveSelector, SelectionCriteria, AppProfile
from src.division_engine import (
    ExactDivisionEngine,
    ApproxDivisionEngine,
    EarlyStopDivisionEngine,
    HybridDivisionEngine,
)
from src.metrics import scaling_analysis, full_comparison_table

app = Flask(__name__)

simulator = DivisionSimulator()
selector  = AdaptiveSelector()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_resources(method: str, n: int, k=None, m=None, h=None):
    """Instantiate the correct engine and return its DivisionResources."""
    if method == "exact":
        return ExactDivisionEngine(n).resources()
    elif method == "approx_truncated":
        k = k if k is not None else max(2, int(math.ceil(n * 0.75)))
        return ApproxDivisionEngine(n, k=k).resources()
    elif method == "early_stop":
        m = m if m is not None else max(1, int(math.ceil(n * 0.75)))
        return EarlyStopDivisionEngine(n, m).resources()
    elif method == "hybrid":
        h = h if h is not None else max(1, n // 2)
        k = k if k is not None else max(2, int(math.ceil(n * 0.75)))
        return HybridDivisionEngine(n, h, k=k).resources()
    return None


def _resources_to_dict(r):
    if r is None:
        return None
    return {
        "t_count":   r.t_count,
        "t_depth":   r.t_depth,
        "cnot_count": r.cnot_count,
        "n_qubits":  r.n_qubits,
        "accuracy":  round(r.accuracy, 6),
        "method":    r.method,
    }


def _profile_recommendations(n: int) -> dict:
    """Return adaptive-selector picks for every AppProfile."""
    profiles = {}
    for profile in [
        AppProfile.EXACT,
        AppProfile.HIGH_PRECISION,
        AppProfile.STANDARD,
        AppProfile.FAST,
        AppProfile.ULTRA_FAST,
    ]:
        crit = SelectionCriteria(n_bits=n, profile=profile)
        result = selector.select(crit)
        r = result.resources
        profiles[profile.value] = {
            "method":    result.engine_class,
            "t_count":   r.t_count,
            "t_depth":   r.t_depth,
            "cnot_count": r.cnot_count,
            "accuracy":  round(r.accuracy, 6),
            "reason":    result.selection_reason,
            "extras":    r.extra,
        }
    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/compute", methods=["POST"])
def compute():
    """
    Accepts JSON or form-data.  Returns JSON with:
      - simulation result (quotient, remainder, error)
      - resource estimates
      - adaptive-selector profile recommendations
      - selection reason
    """
    data = request.get_json(silent=True) or request.form

    # ── parse inputs ──────────────────────────────────────────────────────
    try:
        dividend = int(data.get("dividend", 0))
        divisor  = int(data.get("divisor",  1))
        n_bits   = int(data.get("n_bits",  16))
        method   = str(data.get("method", "auto")).strip()
        accuracy_target = float(data.get("accuracy_target", 0.99))
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    if divisor == 0:
        return jsonify({"error": "Divisor cannot be zero."}), 400
    if n_bits < 1 or n_bits > 128:
        return jsonify({"error": "n_bits must be between 1 and 128."}), 400

    # ── parse method-specific params ──────────────────────────────────────
    k = None
    m = None
    h = None
    try:
        raw_k = data.get("k")
        raw_m = data.get("m")
        raw_h = data.get("h")
        if raw_k not in (None, "", "null"):
            k = int(raw_k)
        if raw_m not in (None, "", "null"):
            m = int(raw_m)
        if raw_h not in (None, "", "null"):
            h = int(raw_h)
    except (ValueError, TypeError):
        pass

    # ── decide method ─────────────────────────────────────────────────────
    sim_method  = method
    sim_kwargs  = {}
    resources   = None
    reason      = ""
    selected_k  = k
    selected_m  = m
    selected_h  = h

    if method == "auto":
        crit = SelectionCriteria(
            n_bits=n_bits,
            dividend_value=dividend,
            accuracy_target=accuracy_target,
            t_count_budget=None,
            profile=AppProfile.CUSTOM,
        )
        sel_result  = selector.select(crit)
        sim_method  = sel_result.engine_class
        resources   = sel_result.resources
        reason      = sel_result.selection_reason

        # Extract params from selected resources for display
        selected_k = resources.extra.get("k_bits")
        selected_m = resources.extra.get("m_steps")
        selected_h = resources.extra.get("h_exact_steps")

        # Build sim_kwargs from resources.extra
        if sim_method == "approx_truncated" and selected_k:
            sim_kwargs = {"k_bits": selected_k}
        elif sim_method == "early_stop" and selected_m:
            sim_kwargs = {"m_steps": selected_m}
        elif sim_method == "hybrid":
            if selected_h:
                sim_kwargs["h_exact"] = selected_h
            if selected_k:
                sim_kwargs["k_bits"] = selected_k

    else:
        # Manual selection
        if method == "approx_truncated":
            selected_k = selected_k if selected_k is not None else max(2, int(math.ceil(n_bits * 0.75)))
            sim_kwargs = {"k_bits": selected_k}
            resources  = _get_resources(method, n_bits, k=selected_k)
            reason     = f"User-selected approximate division with k={selected_k} comparison bits."

        elif method == "early_stop":
            selected_m = selected_m if selected_m is not None else max(1, int(math.ceil(n_bits * 0.75)))
            sim_kwargs = {"m_steps": selected_m}
            resources  = _get_resources(method, n_bits, m=selected_m)
            reason     = f"User-selected early-stop division with m={selected_m} steps out of n={n_bits}."

        elif method == "hybrid":
            selected_h = selected_h if selected_h is not None else max(1, n_bits // 2)
            selected_k = selected_k if selected_k is not None else max(2, int(math.ceil(n_bits * 0.75)))
            sim_kwargs = {"h_exact": selected_h, "k_bits": selected_k}
            resources  = _get_resources(method, n_bits, h=selected_h, k=selected_k)
            reason     = (f"User-selected hybrid division: "
                          f"{selected_h} exact MSB steps + "
                          f"{n_bits - selected_h} approximate steps (k={selected_k}).")

        elif method == "exact":
            sim_kwargs = {}
            resources  = _get_resources("exact", n_bits)
            reason     = "User-selected exact restoring division (paper baseline)."

        else:
            sim_method = "exact"
            sim_kwargs = {}
            resources  = _get_resources("exact", n_bits)
            reason     = "Unknown method — fallback to exact division."

    # ── run simulation ────────────────────────────────────────────────────
    try:
        sim_result = simulator.simulate(
            dividend, divisor, n_bits,
            method=sim_method, **sim_kwargs
        )
    except Exception as e:
        return jsonify({"error": f"Simulation error: {e}"}), 500

    # ── build baseline for savings calculation ────────────────────────────
    exact_resources = ExactDivisionEngine(n_bits).resources()
    t_savings_pct   = 0.0
    cnot_savings_pct = 0.0
    if resources and exact_resources.t_count > 0:
        t_savings_pct    = 100.0 * (exact_resources.t_count - resources.t_count) / exact_resources.t_count
        cnot_savings_pct = 100.0 * (exact_resources.cnot_count - resources.cnot_count) / exact_resources.cnot_count

    # ── assemble response ─────────────────────────────────────────────────
    response = {
        "input": {
            "dividend": dividend,
            "divisor":  divisor,
            "n_bits":   n_bits,
        },
        "result": {
            "quotient":          sim_result.quotient,
            "remainder":         sim_result.remainder,
            "exact_quotient":    sim_result.exact_quotient,
            "exact_remainder":   sim_result.exact_remainder,
            "is_exact":          sim_result.is_exact,
            "quotient_error":    sim_result.quotient_error,
            "relative_error":    round(sim_result.relative_error, 8)
                                 if math.isfinite(sim_result.relative_error) else None,
        },
        "method": {
            "name":        sim_method,
            "params": {
                "k": selected_k,
                "m": selected_m,
                "h": selected_h,
            },
            "reason": reason,
        },
        "resources":       _resources_to_dict(resources),
        "exact_resources": _resources_to_dict(exact_resources),
        "savings": {
            "t_count_pct":    round(t_savings_pct, 2),
            "cnot_pct":       round(cnot_savings_pct, 2),
        },
        "profiles": _profile_recommendations(n_bits),
    }

    return jsonify(response)


@app.route("/pareto")
def pareto():
    """Return Pareto-optimal frontier data for a given n."""
    try:
        n = int(request.args.get("n", 16))
        if n < 2 or n > 128:
            return jsonify({"error": "n must be between 2 and 128"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid n"}), 400

    frontier = selector.pareto_frontier(n)
    rows = []
    for r in frontier:
        rows.append({
            "method":    r.method,
            "t_count":   r.t_count,
            "t_depth":   r.t_depth,
            "cnot_count": r.cnot_count,
            "accuracy":  round(r.accuracy, 6),
            **r.extra,
        })
    return jsonify({"n": n, "pareto": rows})


@app.route("/tradeoff")
def tradeoff():
    """Return full cost-accuracy trade-off table for n bits."""
    try:
        n = int(request.args.get("n", 16))
        if n < 2 or n > 128:
            return jsonify({"error": "n must be between 2 and 128"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid n"}), 400

    rows = selector.tradeoff_table(n)
    return jsonify({"n": n, "tradeoff": rows})


@app.route("/scaling")
def scaling():
    """Return scaling analysis across several bit-widths."""
    bit_widths = [4, 8, 12, 16, 24, 32, 48, 64]
    rows = scaling_analysis(bit_widths)
    return jsonify({"scaling": rows})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)