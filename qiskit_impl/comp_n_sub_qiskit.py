from qiskit import QuantumCircuit


class CompNSubQiskit:
    """
    Parameterised n-bit Comparator-and-Subtract quantum circuit.

    Register layout  (3n + 1 qubits total, every index derived from n):
    ┌─────────────────┬────────────────┬───────────────────────────────────┐
    │ Qubits          │ Role           │ Notes                             │
    ├─────────────────┼────────────────┼───────────────────────────────────┤
    │ 0 .. n-1        │ Register A     │ modified in-place on subtraction  │
    │ n .. 2n-1       │ Register B     │ divisor, logically read-only      │
    │ 2n              │ carry_seed     │ carry-IN for bit 0, init to |1⟩   │
    │ 2n+1 .. 3n-1    │ carry_out[i]   │ n-1 ancillae, one per bit 0..n-2  │
    │ 3n              │ qbit           │ output: 1 iff A ≥ B               │
    └─────────────────┴────────────────┴───────────────────────────────────┘

    Why carry_seed is separate from carry_out[0]
    ---------------------------------------------
    carry_out[i] stores the carry-OUT of bit i.
    bit 0 needs a carry-IN that is |1⟩ (to seed the comparator).
    If we reused carry_out[0] as both carry-in and carry-out for bit 0,
    the Toffoli ccx(c_in, b_0, c_out) would have c_in == c_out — a
    duplicate qubit argument that Qiskit correctly rejects.
    Adding one dedicated carry_seed qubit eliminates all collisions.
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n

    # ------------------------------------------------------------------ #
    #  Index helpers – single source of truth                             #
    # ------------------------------------------------------------------ #

    @property
    def n_qubits(self) -> int:
        return 3 * self.n + 1

    def _a(self, i: int) -> int:
        """Qubit i of register A  (0 ≤ i < n)."""
        return i

    def _b(self, i: int) -> int:
        """Qubit i of register B  (0 ≤ i < n)."""
        return self.n + i

    @property
    def _carry_seed(self) -> int:
        """Dedicated carry-IN ancilla for bit 0.  Initialised to |1⟩."""
        return 2 * self.n

    def _carry_out(self, i: int) -> int:
        """Carry-OUT ancilla for bit i  (0 ≤ i < n-1)."""
        assert 0 <= i < self.n - 1, f"carry_out index out of range: {i}"
        return 2 * self.n + 1 + i      # occupies [2n+1 .. 3n-1]

    @property
    def _qbit(self) -> int:
        """Output qubit: |1⟩ iff A ≥ B after the comparison stage."""
        return 3 * self.n

    def _cin(self, i: int) -> int:
        """Carry-IN qubit for bit position i."""
        return self._carry_seed if i == 0 else self._carry_out(i - 1)

    def _cout(self, i: int) -> int:
        """Carry-OUT qubit for bit position i."""
        return self._carry_out(i) if i < self.n - 1 else self._qbit

    # ------------------------------------------------------------------ #
    #  Circuit construction                                               #
    # ------------------------------------------------------------------ #

    def build(self) -> QuantumCircuit:
        n  = self.n
        qc = QuantumCircuit(self.n_qubits, name=f"CompNSub_{n}")

        # ── Seed: carry-IN for bit 0 starts as |1⟩ ───────────────────── #
        qc.x(self._carry_seed)

        # ── Forward pass: ripple-carry comparison ─────────────────────── #
        # At each bit i:
        #   1. Compute borrow bit into b_i  (X then CX)
        #   2. Propagate carry: ccx(c_in, b_i, c_out)  — all distinct ✓
        #   3. Restore b_i
        # After the loop, _qbit == |1⟩  iff  A ≥ B.
        for i in range(n):
            a_i   = self._a(i)
            b_i   = self._b(i)
            c_in  = self._cin(i)
            c_out = self._cout(i)

            qc.x(b_i)
            qc.cx(a_i, b_i)
            qc.ccx(c_in, b_i, c_out)   # c_in ≠ b_i ≠ c_out by construction
            qc.cx(a_i, b_i)
            qc.x(b_i)

        # ── Conditional subtraction: A ← A XOR B  if qbit == |1⟩ ─────── #
        for i in range(n):
            qc.ccx(self._qbit, self._b(i), self._a(i))

        # ── Uncompute carry chain (restore all ancillae to |0⟩) ──────── #
        # Replay the forward carry propagation in reverse order.
        # This restores carry_out[0..n-2] and carry_seed to |0⟩,
        # leaving only _qbit "dirty" (intentionally — the caller reads it).
        for i in range(n - 2, -1, -1):
            a_i   = self._a(i)
            b_i   = self._b(i)
            c_in  = self._cin(i)
            c_out = self._cout(i)

            qc.x(b_i)
            qc.cx(a_i, b_i)
            qc.ccx(c_in, b_i, c_out)   # same gate reverses the XOR
            qc.cx(a_i, b_i)
            qc.x(b_i)

        # Uncompute the seed ancilla
        qc.x(self._carry_seed)

        return qc