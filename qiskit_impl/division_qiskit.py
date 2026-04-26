from qiskit import QuantumCircuit
from qiskit_impl.comp_n_sub_qiskit import CompNSubQiskit


class DivisionQiskit:
    """
    Parameterised n-bit quantum restoring-division circuit.

    Register layout  (4n + 1 qubits total, every index derived from n):
    ┌──────────────────┬────────────────────────────────────────────────┐
    │ Qubits           │ Role                                           │
    ├──────────────────┼────────────────────────────────────────────────┤
    │ 0 .. 3n          │ CompNSub sub-circuit  (3n+1 qubits)           │
    │   0 .. n-1       │   remainder / partial dividend  (register A)  │
    │   n .. 2n-1      │   divisor  (register B)                       │
    │   2n             │   carry_seed ancilla                           │
    │   2n+1 .. 3n-1   │   carry_out ancillae                          │
    │   3n             │   qbit  (quotient bit, reset each iteration)  │
    │ 3n+1 .. 4n       │ Quotient accumulator  (n qubits)              │
    └──────────────────┴────────────────────────────────────────────────┘
    Total: 4n + 1 qubits
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n

    # ------------------------------------------------------------------ #
    #  Index helpers                                                      #
    # ------------------------------------------------------------------ #

    @property
    def n_qubits(self) -> int:
        return 4 * self.n + 1          # CompNSub(3n+1) + quotient_acc(n)

    @property
    def _qbit(self) -> int:
        """Quotient-bit ancilla (output of CompNSub, reset each iteration)."""
        return 3 * self.n

    def _quotient_acc(self, step: int) -> int:
        """Quotient accumulator qubit for iteration *step* (0-based)."""
        return 3 * self.n + 1 + step  # [3n+1 .. 4n]

    def _comp_qubits(self) -> list:
        """Explicit qubit indices that CompNSub maps onto."""
        return list(range(3 * self.n + 1))

    # ------------------------------------------------------------------ #
    #  Circuit construction                                               #
    # ------------------------------------------------------------------ #

    def build(self) -> QuantumCircuit:
        n    = self.n
        qc   = QuantumCircuit(self.n_qubits, name=f"Division_{n}")
        comp = CompNSubQiskit(n)

        for step in range(n):
            # Embed comparator-subtractor on its dedicated qubits
            qc.compose(comp.build(), qubits=self._comp_qubits(), inplace=True)

            # Copy quotient bit into the accumulator for this step
            qc.cx(self._qbit, self._quotient_acc(step))

            # Reset qbit to |0⟩ so it is clean for the next iteration
            qc.x(self._qbit)

        return qc