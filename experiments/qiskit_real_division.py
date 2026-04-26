from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ==========================================
# SIMPLE REVERSIBLE SUBTRACTOR
# ==========================================
def subtractor(qc, a, b, anc):
    """
    Performs A = A - B (very simplified reversible version)
    """
    for i in range(len(a)):
        qc.cx(b[i], a[i])   # A ^= B


# ==========================================
# COMPARATOR (A >= B)
# ==========================================
def comparator(qc, a, b, flag):
    """
    Very simplified comparator
    Sets flag if A >= B (approx behavior)
    """
    for i in range(len(a)):
        qc.cx(a[i], flag)
        qc.cx(b[i], flag)


# ==========================================
# EXACT DIVISION (SIMPLIFIED)
# ==========================================
def build_exact_division(n):
    """
    Real quantum circuit (simplified restoring division)
    """
    qc = QuantumCircuit(3*n + 1, n)

    A = list(range(0, n))        # dividend
    B = list(range(n, 2*n))      # divisor
    Q = list(range(2*n, 3*n))    # quotient
    flag = 3*n                   # comparison ancilla

    # Initialize example values (for demo)
    qc.x(A[0])   # dividend = 1
    qc.x(B[0])   # divisor = 1

    for i in range(n):
        comparator(qc, A, B, flag)

        # If flag = 1 → subtract
        for j in range(n):
            qc.ccx(flag, B[j], A[j])

        # Store quotient bit
        qc.cx(flag, Q[i])

        # Reset flag
        qc.reset(flag)

    qc.measure(Q, range(n))

    return qc


# ==========================================
# APPROX DIVISION (YOUR CONTRIBUTION)
# ==========================================
def build_approx_division(n, k):
    """
    Approx version: compare only top-k bits
    """
    qc = QuantumCircuit(3*n + 1, n)

    A = list(range(0, n))
    B = list(range(n, 2*n))
    Q = list(range(2*n, 3*n))
    flag = 3*n

    # Example init
    qc.x(A[0])
    qc.x(B[0])

    for i in range(n):
        # Compare only top-k bits
        for j in range(n-k, n):
            qc.cx(A[j], flag)
            qc.cx(B[j], flag)

        # Conditional subtract
        for j in range(n):
            qc.ccx(flag, B[j], A[j])

        qc.cx(flag, Q[i])
        qc.reset(flag)

    qc.measure(Q, range(n))

    return qc


# ==========================================
# RUN SIMULATION
# ==========================================
if __name__ == "__main__":
    n = 3

    qc_exact = build_exact_division(n)
    qc_approx = build_approx_division(n, k=2)

    sim = AerSimulator()

    result1 = sim.run(transpile(qc_exact, sim)).result()
    result2 = sim.run(transpile(qc_approx, sim)).result()

    print("Exact Result:", result1.get_counts())
    print("Approx Result:", result2.get_counts())

    # Draw circuits
    qc_exact.draw('mpl', filename='exact_real.png')
    qc_approx.draw('mpl', filename='approx_real.png')