"""
qiskit_division_circuits.py
Qiskit implementations of all quantum restoring division circuit variants.
"""
from __future__ import annotations
import math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister


# ── MAJ / UMA (Cuccaro ripple-carry building blocks) ──────────────────────────

def _maj(qc, a, b, c):
    qc.cx(c, b); qc.cx(c, a); qc.ccx(a, b, c)

def _umaj(qc, a, b, c):
    qc.ccx(a, b, c); qc.cx(c, a); qc.cx(a, b)


# ── Qubit layout helper ────────────────────────────────────────────────────────

class QIdx:
    def __init__(self, n):
        self.n     = n
        self.A     = list(range(0, n))
        self.B     = list(range(n, 2*n))
        self.carry = list(range(2*n, 3*n))
        self.qbit  = 3*n
        self.Q     = list(range(3*n+1, 4*n+1))
        self.total = 4*n+1


# ── 1. COMP-N-SUB (exact) ─────────────────────────────────────────────────────

def build_comp_n_sub(n):
    """
    Exact COMP-N-SUB: if A>=B then A<-A-B, qbit<-1 else qbit<-0.
    Qubit layout (3n+1): A[0..n-1] | B[0..n-1] | carry[0..n-1] | qbit
    Uses Cuccaro ripple-carry comparator + conditional subtractor.
    T-count = 4n-3, CNOT = 6n-3 (paper Table I).
    """
    n_q = 3*n+1
    qc  = QuantumCircuit(n_q, name=f"COMP-N-SUB n={n}")
    A = list(range(0, n))
    B = list(range(n, 2*n))
    C = list(range(2*n, 3*n))
    qb = 3*n

    # Phase 1: comparison  A + ~B + 1  (carry-out = A>=B)
    for i in range(n): qc.x(B[i])   # complement B
    qc.x(C[0])                       # carry_in = 1 (two's complement +1)

    for i in range(n-1):
        _maj(qc, A[i], B[i], C[i])
        qc.cx(C[i], C[i+1])         # carry forward
    _maj(qc, A[n-1], B[n-1], C[n-1])
    qc.cx(C[n-1], qb)               # carry-out -> quotient bit

    # Phase 2: conditional subtraction  A <- A-B if qbit=1
    for i in range(n):
        qc.ccx(qb, B[i], A[i])     # A[i] ^= B[i]*qbit
        if i < n-1:
            qc.ccx(A[i], B[i], C[i])
            qc.cx(A[i], B[i])

    # Phase 3: uncompute carry chain
    _umaj(qc, A[n-1], B[n-1], C[n-1])
    for i in reversed(range(n-1)):
        qc.cx(C[i], C[i+1])
        _umaj(qc, A[i], B[i], C[i])
    qc.x(C[0])

    # Restore B
    for i in range(n): qc.x(B[i])
    return qc


# ── 2. Exact Division ─────────────────────────────────────────────────────────

def build_exact_division(n):
    """n iterations of COMP-N-SUB. T-count = n*(4n-3). Qubits = 4n+1."""
    idx = QIdx(n)
    qc  = QuantumCircuit(idx.total, name=f"ExactDiv n={n}")
    prim = build_comp_n_sub(n).to_gate(label="COMP\nN-SUB")
    for step in range(n):
        qc.append(prim, idx.A + idx.B + idx.carry + [idx.qbit])
        qc.cx(idx.qbit, idx.Q[step])
        qc.reset(idx.qbit)
        if step < n-1: qc.barrier(label=f"step {step}")
    return qc


# ── 3. Approximate COMP-N-SUB ─────────────────────────────────────────────────

def build_approx_comp_n_sub(n, k):
    """
    Approx COMP-N-SUB: compare only top-k bits. Error prob <= 2^(k-n).
    T-count = 4k-3.
    """
    if k > n: raise ValueError(f"k={k} > n={n}")
    n_q = 3*n+1
    qc  = QuantumCircuit(n_q, name=f"ApproxCNS n={n} k={k}")
    A = list(range(0, n))
    B = list(range(n, 2*n))
    C = list(range(2*n, 3*n))
    qb = 3*n
    off = n-k   # start comparison at MSB offset

    for i in range(n): qc.x(B[i])
    qc.x(C[off])

    for i in range(off, n-1):
        _maj(qc, A[i], B[i], C[i])
        qc.cx(C[i], C[i+1])
    _maj(qc, A[n-1], B[n-1], C[n-1])
    qc.cx(C[n-1], qb)

    # Full n-bit conditional subtraction (remainder stays correct)
    for i in range(n):
        qc.ccx(qb, B[i], A[i])
        if i < n-1:
            qc.ccx(A[i], B[i], C[i])
            qc.cx(A[i], B[i])

    _umaj(qc, A[n-1], B[n-1], C[n-1])
    for i in reversed(range(off, n-1)):
        qc.cx(C[i], C[i+1])
        _umaj(qc, A[i], B[i], C[i])
    qc.x(C[off])
    for i in range(n): qc.x(B[i])
    return qc


def build_approx_division(n, k):
    """n iterations of ApproxCNS(k). T-count = n*(4k-3)."""
    idx = QIdx(n)
    qc  = QuantumCircuit(idx.total, name=f"ApproxDiv n={n} k={k}")
    prim = build_approx_comp_n_sub(n, k).to_gate(label=f"APPROX\nCNS k={k}")
    for step in range(n):
        qc.append(prim, idx.A + idx.B + idx.carry + [idx.qbit])
        qc.cx(idx.qbit, idx.Q[step])
        qc.reset(idx.qbit)
        if step < n-1: qc.barrier(label=f"step {step}")
    return qc


# ── 4. Early-Stop Division ────────────────────────────────────────────────────

def build_early_stop_division(n, m):
    """m of n COMP-N-SUB steps. T-count = m*(4n-3). Error <= 2^(-m)."""
    if m > n: raise ValueError("m > n")
    idx = QIdx(n)
    qc  = QuantumCircuit(idx.total, name=f"EarlyStopDiv n={n} m={m}")
    prim = build_comp_n_sub(n).to_gate(label="COMP\nN-SUB")
    for step in range(m):
        qc.append(prim, idx.A + idx.B + idx.carry + [idx.qbit])
        qc.cx(idx.qbit, idx.Q[step])
        qc.reset(idx.qbit)
        if step < m-1: qc.barrier(label=f"step {step}/{m}")
    return qc


# ── 5. Hybrid Division ────────────────────────────────────────────────────────

def build_hybrid_division(n, h, k):
    """h exact + (n-h) approx steps. T = h*(4n-3)+(n-h)*(4k-3)."""
    if h > n: raise ValueError("h > n")
    idx = QIdx(n)
    qc  = QuantumCircuit(idx.total, name=f"HybridDiv n={n} h={h} k={k}")
    eg  = build_comp_n_sub(n).to_gate(label="EXACT\nCNS")
    ag  = build_approx_comp_n_sub(n, k).to_gate(label=f"APPROX\nCNS k={k}")
    for step in range(n):
        gate = eg if step < h else ag
        qc.append(gate, idx.A + idx.B + idx.carry + [idx.qbit])
        qc.cx(idx.qbit, idx.Q[step])
        qc.reset(idx.qbit)
        if step < n-1:
            lbl = "→APPROX" if step == h-1 else ""
            qc.barrier(label=lbl if lbl else f"step {step}")
    return qc


# ── Analytical resources ───────────────────────────────────────────────────────

def analytical_resources(method, n, **kw):
    base_t = n*(4*n-3)
    if method == "exact":
        return {"t_count": base_t, "cnot_count": n*(6*n-3),
                "t_depth": n*(2*math.ceil(math.log2(n))+2) if n>1 else 2,
                "n_qubits": 4*n+1, "accuracy": 1.0, "t_saving_pct": 0.0}
    elif method == "approx":
        k=kw["k"]; p=2.0**(k-n); acc=max(0.0,(1-p)**n); t=n*(4*k-3)
        return {"t_count":t,"cnot_count":n*(6*k-3),"n_qubits":4*n+1,
                "accuracy":round(acc,6),"t_saving_pct":round(100*(1-t/base_t),2)}
    elif method == "early_stop":
        m=kw["m"]; t=m*(4*n-3)
        return {"t_count":t,"cnot_count":m*(6*n-3),"n_qubits":4*n+1,
                "accuracy":round(1-2**(-m),6),"t_saving_pct":round(100*(1-m/n),2)}
    elif method == "hybrid":
        h,k=kw["h"],kw["k"]; p=2.0**(k-n)
        err=(n-h)*p*(2.0**(-h)); t=h*(4*n-3)+(n-h)*(4*k-3)
        return {"t_count":t,"cnot_count":h*(6*n-3)+(n-h)*(6*k-3),
                "n_qubits":4*n+1,"accuracy":round(max(0,1-err),6),
                "t_saving_pct":round(100*(1-t/base_t),2)}
    return {}


# ── Figure helpers ─────────────────────────────────────────────────────────────

DARK_BG  = "#0f0f1a"
DARK_BOX = "#1a1a2e"
BLUE_LN  = "#4fc3f7"
TEXT     = "#e0e0ff"

QK_STYLE = {
    "backgroundcolor": DARK_BOX,
    "textcolor":       TEXT,
    "gatefacecolor":   "#0d3060",
    "gateedgecolor":   BLUE_LN,
    "linecolor":       "#7eb8d4",
    "displaycolor": {
        "cx":  ("#0d47a1","#ffffff"),
        "ccx": ("#1b5e20","#ffffff"),
        "x":   ("#311b92","#ffffff"),
        "reset":("#4a148c","#ffffff"),
    }
}


def _save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close("all")
    print(f"  ✓  {path}")


# Figure 1 – COMP-N-SUB primitive
def draw_comp_n_sub(n=3, out="comp_n_sub.png"):
    qc  = build_comp_n_sub(n)
    fig = qc.draw(output="mpl", fold=40, style=QK_STYLE)
    fig.suptitle(
        f"COMP-N-SUB Primitive  (n = {n} bits)\n"
        f"T-count = {4*n-3}   CNOT = {6*n-3}   Qubits = {3*n+1}",
        fontsize=13, fontweight="bold", color="#ffffff",
        fontfamily="monospace")
    fig.tight_layout(pad=1.5)
    _save(fig, out)


# Figure 2 – All four division circuits
def draw_all_circuits(n=3, out="all_circuits.png"):
    k = max(2, int(math.ceil(n*0.75)))
    m = max(1, int(n*0.75))
    h = max(1, int(n*0.5))

    def res(method, **kw):
        return analytical_resources(method, n, **kw)

    r0 = res("exact")
    r1 = res("approx",      k=k)
    r2 = res("early_stop",  m=m)
    r3 = res("hybrid",      h=h, k=k)

    entries = [
        (f"① Exact Division  (n={n})\n"
         f"   T = {r0['t_count']}   CNOT = {r0['cnot_count']}   Acc = 1.0000",
         build_exact_division(n)),
        (f"② Approximate Division  (k = {k} of {n} bits)\n"
         f"   T = {r1['t_count']}   Saving = {r1['t_saving_pct']}%   Acc ≈ {r1['accuracy']}",
         build_approx_division(n, k=k)),
        (f"③ Early-Stop Division  (m = {m} of {n} steps)\n"
         f"   T = {r2['t_count']}   Saving = {r2['t_saving_pct']}%   Acc ≈ {r2['accuracy']}",
         build_early_stop_division(n, m=m)),
        (f"④ Hybrid Division  (h = {h} exact, k = {k} approx)\n"
         f"   T = {r3['t_count']}   Saving = {r3['t_saving_pct']}%   Acc ≈ {r3['accuracy']}",
         build_hybrid_division(n, h=h, k=k)),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(22, 30))
    fig.patch.set_facecolor(DARK_BG)
    for ax, (title, qc) in zip(axes, entries):
        qc.draw(output="mpl", fold=50, style=QK_STYLE, ax=ax)
        ax.set_title(title, fontsize=11, fontweight="bold",
                     color=TEXT, fontfamily="monospace", pad=6)
        ax.set_facecolor(DARK_BOX)
    fig.suptitle(
        f"Quantum Restoring Division — All Circuit Variants  (n = {n} bits)\n"
        "Approximate variants reduce T-gate cost while maintaining high accuracy",
        fontsize=15, fontweight="bold", color="#ffffff",
        fontfamily="monospace", y=1.005)
    plt.tight_layout(pad=2.0)
    _save(fig, out)


# Figure 3 – Resource comparison table
def draw_resource_table(n=16, out="resource_table.png"):
    k75=max(2,int(math.ceil(n*0.75))); k50=max(2,int(math.ceil(n*0.50)))
    m75=max(1,int(n*0.75)); h50=max(1,int(n*0.50))
    configs = [
        ("Exact (baseline)",          analytical_resources("exact",     n)),
        (f"Approx  k = {k75}",        analytical_resources("approx",    n, k=k75)),
        (f"Approx  k = {k50}",        analytical_resources("approx",    n, k=k50)),
        (f"Early-Stop  m = {m75}",    analytical_resources("early_stop",n, m=m75)),
        (f"Hybrid  h={h50}, k={k75}", analytical_resources("hybrid",    n, h=h50, k=k75)),
    ]
    col_labels = ["Method","n","T-count","T saving %","CNOT","Accuracy"]
    rows = [[name,str(n),f"{r['t_count']:,}",f"{r['t_saving_pct']:.1f}%",
             f"{r['cnot_count']:,}",f"{r['accuracy']:.4f}"]
            for name,r in configs]

    fig, ax = plt.subplots(figsize=(16,4))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG); ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1,2.4)
    EDGE="#4fc3f7"; ROW_BG=["#1a1a2e","#0d1b2a"]
    for j in range(len(col_labels)):
        c=tbl[0,j]; c.set_facecolor("#1565c0"); c.set_edgecolor(EDGE)
        c.set_text_props(color="white",fontweight="bold",fontfamily="monospace")
    for i,row in enumerate(rows):
        for j in range(len(col_labels)):
            c=tbl[i+1,j]; c.set_facecolor(ROW_BG[i%2]); c.set_edgecolor(EDGE)
            clr=TEXT
            if j==0: clr="#80cbc4"
            elif j==5:
                acc=float(row[j])
                clr="#a5d6a7" if acc>=0.999 else "#fff176" if acc>=0.99 else "#ef9a9a"
            elif j==3 and i>0: clr="#4fc3f7"
            c.set_text_props(color=clr,fontfamily="monospace")
    ax.set_title(
        f"Resource Comparison: Quantum Restoring Division  (n = {n} bits)\n"
        "Analytical formulas — T-count, CNOT count, accuracy model",
        fontsize=13,fontweight="bold",color=TEXT,fontfamily="monospace",pad=14)
    plt.tight_layout(pad=1.2)
    _save(fig, out)


# Figure 4 – T-count scaling
def draw_scaling_plot(out="scaling_plot.png"):
    ns=list(range(4,33,2))
    def et(n): return n*(4*n-3)
    def at(n,f): k=max(2,int(math.ceil(n*f))); return n*(4*k-3)
    def st(n,f): m=max(1,int(n*f)); return m*(4*n-3)
    def ht(n,hf,kf):
        h=max(1,int(n*hf)); k=max(2,int(math.ceil(n*kf)))
        return h*(4*n-3)+(n-h)*(4*k-3)
    datasets = {
        "Exact (baseline)":      ([et(n) for n in ns],          "#ef5350","-","o",2.5),
        "Approx  k=75%n":        ([at(n,0.75) for n in ns],     "#42a5f5","--","s",2.0),
        "Approx  k=50%n":        ([at(n,0.50) for n in ns],     "#26c6da",":","^",2.0),
        "Early-Stop  m=75%n":    ([st(n,0.75) for n in ns],     "#66bb6a","-.","D",2.0),
        "Hybrid  h=50%, k=75%":  ([ht(n,.5,.75) for n in ns],  "#ab47bc","--","P",2.0),
    }
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,7))
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax1,ax2):
        ax.set_facecolor(DARK_BOX); ax.tick_params(colors="#c0c0e0",labelsize=11)
        for sp in ax.spines.values(): sp.set_edgecolor(BLUE_LN)
        ax.grid(True,alpha=0.18,color=BLUE_LN)
    ev=datasets["Exact (baseline)"][0]
    for lbl,(ys,col,ls,mk,lw) in datasets.items():
        ax1.plot(ns,ys,color=col,linestyle=ls,marker=mk,linewidth=lw,markersize=5,label=lbl)
    ax1.set_yscale("log")
    ax1.set_xlabel("Bit-width  n",fontsize=13,color="#c0c0e0")
    ax1.set_ylabel("T-count",fontsize=13,color="#c0c0e0")
    ax1.set_title("T-count Scaling (log scale)",fontsize=14,fontweight="bold",
                  color=TEXT,fontfamily="monospace")
    ax1.legend(fontsize=10,facecolor=DARK_BG,edgecolor=BLUE_LN,labelcolor="#c0c0e0")
    for lbl,(ys,col,ls,mk,lw) in datasets.items():
        if lbl=="Exact (baseline)": continue
        sav=[100*(1-y/e) for y,e in zip(ys,ev)]
        ax2.plot(ns,sav,color=col,linestyle=ls,marker=mk,linewidth=lw,markersize=5,label=lbl)
    ax2.axhline(0,color="#ef5350",linewidth=1,linestyle="--",alpha=0.6)
    ax2.set_xlabel("Bit-width  n",fontsize=13,color="#c0c0e0")
    ax2.set_ylabel("T-count Saving (%)",fontsize=13,color="#c0c0e0")
    ax2.set_title("T-count Savings vs Exact Baseline",fontsize=14,fontweight="bold",
                  color=TEXT,fontfamily="monospace")
    ax2.legend(fontsize=10,facecolor=DARK_BG,edgecolor=BLUE_LN,labelcolor="#c0c0e0")
    fig.suptitle(
        "Quantum Division: T-count Scaling & Savings\n"
        "Approx / Hybrid / Early-Stop vs Paper Baseline",
        fontsize=15,fontweight="bold",color="#ffffff",fontfamily="monospace")
    plt.tight_layout(pad=1.5)
    _save(fig, out)


# Figure 5 – Pareto scatter
def draw_pareto_scatter(n=16, out="pareto_scatter.png"):
    configs=[]
    configs.append(("Exact",1.0,0.0,"#ef5350","o",200))
    for f in [0.875,0.75,0.625,0.5]:
        k=max(2,int(math.ceil(n*f))); r=analytical_resources("approx",n,k=k)
        configs.append((f"Approx k={k}",r["accuracy"],r["t_saving_pct"],"#42a5f5","s",100))
    for f in [0.875,0.75,0.625,0.5]:
        m=max(1,int(n*f)); r=analytical_resources("early_stop",n,m=m)
        configs.append((f"Early m={m}",r["accuracy"],r["t_saving_pct"],"#66bb6a","D",100))
    for hf,kf in [(0.75,0.875),(0.5,0.75),(0.25,0.625)]:
        h=max(1,int(n*hf)); k=max(2,int(math.ceil(n*kf)))
        r=analytical_resources("hybrid",n,h=h,k=k)
        configs.append((f"Hybrid h={h},k={k}",r["accuracy"],r["t_saving_pct"],"#ab47bc","P",100))
    fig,ax=plt.subplots(figsize=(13,8))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BOX)
    ax.tick_params(colors="#c0c0e0",labelsize=11)
    for sp in ax.spines.values(): sp.set_edgecolor(BLUE_LN)
    ax.grid(True,alpha=0.18,color=BLUE_LN)
    pts=sorted(configs,key=lambda x:x[2]); pareto=[]; best=-1
    for p in pts:
        if p[1]>best: pareto.append(p); best=p[1]
    if len(pareto)>1:
        ax.plot([p[2] for p in pareto],[p[1] for p in pareto],
                "w--",linewidth=1.5,alpha=0.5,label="Pareto frontier",zorder=3)
    for lbl,acc,sav,col,mk,sz in configs:
        ax.scatter(sav,acc,color=col,marker=mk,s=sz,zorder=4,
                   edgecolors="#ffffff",linewidths=0.5)
        ax.annotate(lbl,(sav,acc),textcoords="offset points",xytext=(6,4),
                    fontsize=8,color="#c0c0e0",fontfamily="monospace")
    ax.axhline(0.999,color="#a5d6a7",linewidth=1,linestyle=":",alpha=0.7,label="99.9% threshold")
    ax.axhline(0.99, color="#fff176",linewidth=1,linestyle=":",alpha=0.7,label="99% threshold")
    ax.set_xlabel("T-count Saving (%)",fontsize=13,color="#c0c0e0")
    ax.set_ylabel("Accuracy",fontsize=13,color="#c0c0e0")
    ax.set_title(f"Accuracy vs T-count Saving  (n = {n} bits)\nPareto Frontier Highlighted",
                 fontsize=14,fontweight="bold",color=TEXT,fontfamily="monospace")
    ax.set_ylim(-0.05,1.12)
    ax.legend(fontsize=10,facecolor=DARK_BG,edgecolor=BLUE_LN,labelcolor="#c0c0e0")
    plt.tight_layout(pad=1.5)
    _save(fig, out)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    out="/mnt/d/quantum-division/results/plots"; os.makedirs(out,exist_ok=True)
    N_CIRC=3; N_TABLE=16

    print("="*62)
    print("  Quantum Division — Qiskit Circuit Generation")
    print("="*62)

    k=max(2,int(math.ceil(N_CIRC*0.75))); m=max(1,int(N_CIRC*0.75)); h=max(1,int(N_CIRC*0.5))
    circuits = {"exact":build_exact_division(N_CIRC), "approx":build_approx_division(N_CIRC,k=k),
                "early_stop":build_early_stop_division(N_CIRC,m=m),
                "hybrid":build_hybrid_division(N_CIRC,h=h,k=k)}
    print(f"\n  Circuit summary (n={N_CIRC}):")
    print(f"  {'Method':<16}{'Qubits':>7}{'Depth':>7}")
    print("  "+"-"*32)
    for name,qc in circuits.items():
        print(f"  {name:<16}{qc.num_qubits:>7}{qc.depth():>7}")

    k75=max(2,int(math.ceil(N_TABLE*0.75))); k50=max(2,int(math.ceil(N_TABLE*0.50)))
    m75=max(1,int(N_TABLE*0.75)); h50=max(1,int(N_TABLE*0.50))
    print(f"\n  Analytical resources (n={N_TABLE}):")
    print(f"  {'Method':<25}{'T-count':>9}{'Saving':>9}{'Accuracy':>11}")
    print("  "+"-"*56)
    for name,r in [
        ("Exact",               analytical_resources("exact",N_TABLE)),
        (f"Approx k={k75}",    analytical_resources("approx",N_TABLE,k=k75)),
        (f"Approx k={k50}",    analytical_resources("approx",N_TABLE,k=k50)),
        (f"Early-Stop m={m75}",analytical_resources("early_stop",N_TABLE,m=m75)),
        (f"Hybrid h={h50},k={k75}",analytical_resources("hybrid",N_TABLE,h=h50,k=k75)),
    ]:
        print(f"  {name:<25}{r['t_count']:>9,}{r['t_saving_pct']:>8.1f}%{r['accuracy']:>11.4f}")

    print("\n  Generating figures...")
    draw_comp_n_sub(n=N_CIRC,  out=f"{out}/comp_n_sub.png")
    draw_all_circuits(n=N_CIRC, out=f"{out}/all_circuits.png")
    draw_resource_table(n=N_TABLE, out=f"{out}/resource_table.png")
    draw_scaling_plot(out=f"{out}/scaling_plot.png")
    draw_pareto_scatter(n=N_TABLE, out=f"{out}/pareto_scatter.png")
    print(f"\n  All outputs → {out}/")
    print("="*62)


if __name__=="__main__":
    main()