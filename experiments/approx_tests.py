"""
approx_tests.py — Real PennyLane quantum simulation of division circuits.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pennylane_simulator import PennyLaneSimulator
from src.comp_n_sub import CompNSub, ApproxCompNSub
from src.division_engine import ExactDivisionEngine, ApproxDivisionEngine
from src.metrics import to_csv


def test_exact_correctness_pennylane(n_bits=4):
    print(f"\n[1] PennyLane correctness test (n={n_bits}, exhaustive)")
    sim = PennyLaneSimulator()
    errors, count = [], 0
    limit = min(16, (1 << n_bits))
    for a in range(limit):
        for b in range(1, limit):
            res = sim.run_division(a, b, n_bits, method="exact")
            if not res.is_exact:
                errors.append((a, b, res.quotient, res.remainder,
                                res.exact_quotient, res.exact_remainder))
            count += 1
    if errors:
        print(f"  x {len(errors)} errors out of {count}:")
        for row in errors[:5]:
            print(f"    {row[0]} / {row[1]}: got {row[2]}R{row[3]}, expected {row[4]}R{row[5]}")
    else:
        print(f"  OK All {count} tests passed (PennyLane simulation correct)")
    return len(errors) == 0


def run_accuracy_benchmark_pennylane(n_bits=4, n_trials=40):
    print(f"\n[2] PennyLane accuracy benchmark (n={n_bits}, {n_trials} trials)")
    sim = PennyLaneSimulator(device="default.qubit")
    k3 = max(2, int(math.ceil(n_bits * 0.75)))
    k2 = max(2, int(math.ceil(n_bits * 0.50)))
    m  = max(1, int(n_bits * 0.75))
    h  = max(1, int(n_bits // 2))
    configs = [
        ("exact",            {}),
        ("approx_truncated", {"k_bits": k3}),
        ("approx_truncated", {"k_bits": k2}),
        ("early_stop",       {"m_steps": m}),
        ("hybrid",           {"h_exact": h, "k_bits": k3}),
    ]
    results = []
    print(f"\n  {'Method':<35} {'Accuracy':>10} {'Mean Err':>12}")
    print("  " + "-" * 60)
    for method, kwargs in configs:
        r = sim.benchmark(n_bits=n_bits, method=method, n_trials=n_trials,
                          seed=42, verbose=False, **kwargs)
        label = method
        if kwargs.get("k_bits"):   label += f"(k={kwargs['k_bits']})"
        if kwargs.get("m_steps"):  label += f"(m={kwargs['m_steps']})"
        if kwargs.get("h_exact"):  label += f"(h={kwargs['h_exact']})"
        print(f"  {label:<35} {r['accuracy']:>10.4f} {r['mean_rel_error']:>12.6f}")
        results.append(r)
    return results


def test_gate_costs_pennylane(n_bits=4):
    print(f"\n[3] Gate cost check via PennyLane (n={n_bits})")
    configs = [
        (CompNSub(n_bits),                                     "CompNSub exact"),
        (ApproxCompNSub(n_bits, k=max(2, int(math.ceil(n_bits*0.75)))),
                                                               "ApproxCompNSub k=75%n"),
    ]
    all_ok = True
    for prim, label in configs:
        circ = prim.build()
        analytical = prim.analytical_cost()
        our_t  = circ.t_count
        our_cx = circ.cnot_count
        t_ratio = our_t / max(1, analytical["t_count"])
        ok = 0.5 <= t_ratio <= 15.0
        status = "OK" if ok else "FAIL"
        if not ok: all_ok = False
        print(f"  {status} {label}:")
        print(f"     Analytical T={analytical['t_count']} CNOT={analytical['cnot_count']}")
        print(f"     Circuit    T={our_t} CNOT={our_cx} (ratio={t_ratio:.2f}x)")
    return all_ok


def draw_circuits(n_bits=3):
    print(f"\n[4] PennyLane circuit drawing (n={n_bits})")
    sim = PennyLaneSimulator()
    for method, kwargs in [("exact", {}), ("approx_truncated", {"k_bits": max(2, int(math.ceil(n_bits*0.75)))})]:
        label = method + (f"_k{kwargs.get('k_bits','')}" if kwargs else "")
        print(f"\n  -- {label} --")
        try:
            drawing = sim.draw_circuit(n_bits, method=method, **kwargs)
            for line in drawing.split("\n")[:25]:
                print("  " + line)
        except Exception as e:
            print(f"  (drawing unavailable: {e})")


def analyse_error_distribution(n_bits=4, n_trials=30):
    import random as _random
    print(f"\n[5] Error distribution (PennyLane, n={n_bits}, {n_trials} trials)")
    sim = PennyLaneSimulator()
    rng = _random.Random(99)
    max_val = (1 << n_bits) - 1
    k3 = max(2, int(math.ceil(n_bits * 0.75)))
    h  = max(1, int(n_bits * 0.5))
    configs = [
        ("exact",            {}),
        ("approx_truncated", {"k_bits": k3}),
        ("hybrid",           {"h_exact": h, "k_bits": k3}),
    ]
    for method, kwargs in configs:
        label = method + (f"_k{kwargs.get('k_bits','')}" if kwargs.get("k_bits") else "")
        errors = []
        for _ in range(n_trials):
            a, b = rng.randint(0, max_val), rng.randint(1, max_val)
            try:
                r = sim.run_division(a, b, n_bits, method=method, **kwargs)
                if math.isfinite(r.relative_error):
                    errors.append(r.relative_error)
            except Exception:
                errors.append(1.0)
        exact_rate = sum(1 for e in errors if e == 0) / max(1, len(errors))
        mean_err   = sum(errors) / max(1, len(errors))
        print(f"\n  {label}: exact={exact_rate*100:.1f}%, mean_err={mean_err:.4f}")
        for bucket, lo, hi in [("0%",0,0),("<1%",1e-9,0.01),("1-5%",0.01,0.05),(">5%",0.05,1e9)]:
            cnt = sum(1 for e in errors if lo <= e < hi) if lo > 0 else sum(1 for e in errors if e == 0)
            bar = "=" * (cnt * 20 // max(1, n_trials))
            print(f"    {bucket:>6}: {cnt:>4}  [{bar}]")


def run_all(save_results=True):
    print("=" * 65)
    print("  APPROXIMATION TESTS - Real PennyLane Quantum Simulation")
    print("=" * 65)
    correct  = test_exact_correctness_pennylane(n_bits=4)
    bench    = run_accuracy_benchmark_pennylane(n_bits=4, n_trials=40)
    costs_ok = test_gate_costs_pennylane(n_bits=4)
    draw_circuits(n_bits=3)
    analyse_error_distribution(n_bits=4, n_trials=30)
    if save_results:
        os.makedirs("results/tables", exist_ok=True)
        path = "results/tables/pennylane_accuracy_benchmark.csv"
        with open(path, "w") as f:
            f.write(to_csv(bench))
        print(f"\n  Saved -> {path}")
    print("\n" + "=" * 65)
    print(f"  SUMMARY: exact_correct={correct}, costs_ok={costs_ok}")
    print("=" * 65)
    return bench


if __name__ == "__main__":
    run_all()