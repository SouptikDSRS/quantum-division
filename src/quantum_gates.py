"""
quantum_gates.py
─────────────────────────────────────────────────────────────────────────────
Real PennyLane quantum gate abstractions.

Every gate function DIRECTLY applies a PennyLane operation inside a QNode
context AND records its resource cost so the rest of the codebase can call
circuit.t_count / circuit.cnot_count as before.

Gate cost model (fault-tolerant, matches the paper):
  T / Tdg  : cost = 1
  CNOT     : cost = 1
  Clifford (X, H, S, CZ, SWAP, …) : cost ≈ 0
  Toffoli  : T-count = 7, T-depth = 4, CNOT = 6   (Amy et al. 2013)

QuantumCircuit now owns a list of (op_fn, wires) pairs that are executed
inside a PennyLane QNode, plus the resource counters kept exactly as before.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Any

import pennylane as qml
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Cost constants
# ─────────────────────────────────────────────────────────────────────────────

T_COST     = 1
CNOT_COST  = 1
CLIFF_COST = 0


# ─────────────────────────────────────────────────────────────────────────────
# Gate record  (resource cost + the PennyLane operation to apply)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Gate:
    """
    Represents one gate: its PennyLane op callable, the wire indices it acts
    on, and its resource costs.
    """
    name: str
    qubits: Tuple[int, ...]
    t_count: int = 0
    t_depth: int = 0
    cnot_count: int = 0
    clifford_count: int = 0
    # The actual PennyLane operation callable (no args, wires already bound)
    _op: Any = field(default=None, repr=False, compare=False)

    def apply(self):
        """Execute the PennyLane operation (call inside a QNode)."""
        if self._op is not None:
            self._op()

    def __repr__(self) -> str:
        return (f"{self.name}(qubits={self.qubits}, "
                f"T={self.t_count}, CNOT={self.cnot_count})")


# ─────────────────────────────────────────────────────────────────────────────
# Gate constructors  (each returns a Gate with a bound PennyLane op)
# ─────────────────────────────────────────────────────────────────────────────

def T(q: int) -> Gate:
    g = Gate("T", (q,), t_count=1, t_depth=1)
    g._op = lambda: qml.T(wires=q)
    return g

def Tdg(q: int) -> Gate:
    g = Gate("Tdg", (q,), t_count=1, t_depth=1)
    g._op = lambda: qml.adjoint(qml.T)(wires=q)
    return g

def H(q: int) -> Gate:
    g = Gate("H", (q,), clifford_count=1)
    g._op = lambda: qml.Hadamard(wires=q)
    return g

def S(q: int) -> Gate:
    g = Gate("S", (q,), clifford_count=1)
    g._op = lambda: qml.S(wires=q)
    return g

def Sdg(q: int) -> Gate:
    g = Gate("Sdg", (q,), clifford_count=1)
    g._op = lambda: qml.adjoint(qml.S)(wires=q)
    return g

def X(q: int) -> Gate:
    g = Gate("X", (q,), clifford_count=1)
    g._op = lambda: qml.PauliX(wires=q)
    return g

def CNOT(ctrl: int, tgt: int) -> Gate:
    g = Gate("CNOT", (ctrl, tgt), cnot_count=1)
    g._op = lambda: qml.CNOT(wires=[ctrl, tgt])
    return g

def CZ(ctrl: int, tgt: int) -> Gate:
    g = Gate("CZ", (ctrl, tgt), clifford_count=1)
    g._op = lambda: qml.CZ(wires=[ctrl, tgt])
    return g

def SWAP(q1: int, q2: int) -> Gate:
    g = Gate("SWAP", (q1, q2), cnot_count=3)
    g._op = lambda: qml.SWAP(wires=[q1, q2])
    return g

def Toffoli(ctrl1: int, ctrl2: int, tgt: int) -> Gate:
    """
    Standard Toffoli (CCX).
    T-count=7, T-depth=4, CNOT=6  (Amy et al. arXiv:1206.0758)
    PennyLane: qml.Toffoli
    """
    g = Gate("Toffoli", (ctrl1, ctrl2, tgt),
             t_count=7, t_depth=4, cnot_count=6)
    g._op = lambda: qml.Toffoli(wires=[ctrl1, ctrl2, tgt])
    return g

def Toffoli_approx(ctrl1: int, ctrl2: int, tgt: int,
                   epsilon: float = 0.01) -> Gate:
    """
    Approximate Toffoli via repeat-until-success (Bocharov et al. 2015).
    Cost model:  t_count = max(4, 3·log₂(1/ε)),  cnot = max(3, 2·log₂(1/ε))

    PennyLane doesn't have a native approximate Toffoli, so we decompose it
    into the exact Toffoli — this gives the SAME logical action but with the
    resource counters reflecting the approximate cost.
    """
    tc = max(4, int(3 * math.log2(1.0 / epsilon)))
    cc = max(3, int(2 * math.log2(1.0 / epsilon)))
    td = max(2, tc // 2)
    g = Gate(f"Toffoli_approx(ε={epsilon:.3f})", (ctrl1, ctrl2, tgt),
             t_count=tc, t_depth=td, cnot_count=cc)
    # Logical action is the same — use exact Toffoli in the real circuit
    g._op = lambda: qml.Toffoli(wires=[ctrl1, ctrl2, tgt])
    return g


# ─────────────────────────────────────────────────────────────────────────────
# QuantumCircuit — stores gates, executes them in PennyLane QNodes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuantumCircuit:
    """
    A circuit builder that:
      • Stores Gate records for resource counting (exactly as before).
      • Can execute on a real PennyLane simulator via to_qnode() or run().
    """
    n_qubits: int
    name: str = "circuit"
    gates: List[Gate] = field(default_factory=list)

    # ── resource aggregates ────────────────────────────────────────────────

    @property
    def t_count(self) -> int:
        return sum(g.t_count for g in self.gates)

    @property
    def t_depth(self) -> int:
        qubit_t_layer: dict = {}
        max_layer = 0
        for g in self.gates:
            if g.t_count > 0:
                current_max = max((qubit_t_layer.get(q, 0) for q in g.qubits),
                                  default=0)
                new_layer = current_max + g.t_depth
                for q in g.qubits:
                    qubit_t_layer[q] = new_layer
                max_layer = max(max_layer, new_layer)
        return max_layer

    @property
    def cnot_count(self) -> int:
        return sum(g.cnot_count for g in self.gates)

    @property
    def gate_count(self) -> int:
        return len(self.gates)

    @property
    def depth(self) -> int:
        qubit_depth: dict = {}
        for g in self.gates:
            current_max = max((qubit_depth.get(q, 0) for q in g.qubits),
                              default=0)
            for q in g.qubits:
                qubit_depth[q] = current_max + 1
        return max(qubit_depth.values(), default=0)

    # ── gate list mutation ─────────────────────────────────────────────────

    def append(self, gate: Gate) -> "QuantumCircuit":
        self.gates.append(gate)
        return self

    def extend(self, gates: List[Gate]) -> "QuantumCircuit":
        self.gates.extend(gates)
        return self

    def __iadd__(self, other: "QuantumCircuit") -> "QuantumCircuit":
        self.gates.extend(other.gates)
        return self

    # ── PennyLane execution ────────────────────────────────────────────────

    def _circuit_fn(self, init_state: np.ndarray = None):
        """
        The PennyLane circuit function — applies all stored gates in order.
        Call this inside a QNode.
        """
        if init_state is not None:
            qml.BasisState(init_state, wires=range(self.n_qubits))
        for g in self.gates:
            g.apply()

    def to_qnode(self, device: str = "default.qubit"):
        """
        Return a PennyLane QNode that executes this circuit and returns
        the full statevector probability distribution.

        Parameters
        ----------
        device : PennyLane device string (default: "default.qubit")
        """
        dev = qml.device(device, wires=self.n_qubits)

        @qml.qnode(dev)
        def node(init_state=None):
            self._circuit_fn(init_state)
            return qml.probs(wires=range(self.n_qubits))

        return node

    def run(self, init_state: np.ndarray = None,
            device: str = "default.qubit") -> np.ndarray:
        """
        Execute the circuit on a PennyLane simulator and return the
        probability vector over all computational basis states.

        Parameters
        ----------
        init_state : integer array of length n_qubits giving the initial
                     computational basis state, e.g. [1,0,1,0,...].
                     If None, starts in |00…0⟩.
        device     : PennyLane device string.

        Returns
        -------
        probs : np.ndarray of shape (2**n_qubits,)
        """
        node = self.to_qnode(device)
        return np.array(node(init_state))

    def statevector(self, init_state: np.ndarray = None,
                    device: str = "default.qubit") -> np.ndarray:
        """
        Return the full statevector (complex amplitudes).
        """
        dev = qml.device(device, wires=self.n_qubits)

        @qml.qnode(dev)
        def node(init_state=None):
            self._circuit_fn(init_state)
            return qml.state()

        return np.array(node(init_state))

    def draw(self) -> str:
        """Return a string drawing of the circuit (PennyLane drawer)."""
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev)
        def node():
            self._circuit_fn()
            return qml.probs(wires=range(self.n_qubits))

        return qml.draw(node)()

    def pennylane_resources(self) -> dict:
        """
        Use PennyLane's built-in resource estimator (qml.resource) on the
        circuit and return its output dict.
        """
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev)
        def node():
            self._circuit_fn()
            return qml.probs(wires=range(self.n_qubits))

        specs = qml.specs(node)()
        return specs

    # ── reporting ──────────────────────────────────────────────────────────

    def resource_summary(self) -> dict:
        return {
            "name":       self.name,
            "n_qubits":   self.n_qubits,
            "gate_count": self.gate_count,
            "t_count":    self.t_count,
            "t_depth":    self.t_depth,
            "cnot_count": self.cnot_count,
            "depth":      self.depth,
        }

    def __repr__(self) -> str:
        r = self.resource_summary()
        return (f"QuantumCircuit('{r['name']}', qubits={r['n_qubits']}, "
                f"gates={r['gate_count']}, T={r['t_count']}, "
                f"T-depth={r['t_depth']}, CNOT={r['cnot_count']})")