# Quantum Division Using COMP-N-SUB Algorithm

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Qiskit](https://img.shields.io/badge/Qiskit-0.44+-purple.svg)
![PennyLane](https://img.shields.io/badge/PennyLane-0.30+-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![IBM Quantum](https://img.shields.io/badge/IBM%20Quantum-Ready-blue.svg)

## 📌 Overview

This repository implements **quantum division circuits** using the **COMP-N-SUB (Comparison and Subtraction)** algorithm. It provides exact and approximate division techniques with adaptive resource selection, designed for near-term quantum computers.

The implementation supports:
- **Qiskit** (IBM quantum computers)
- **PennyLane** (simulators)
- **Classical simulation** for baseline comparison

---

## 🚀 Key Features

### 🔹 Exact Division
- COMP-N-SUB based restoring division algorithm
- Verified correctness for n-bit numbers (tested up to 16-bit)
- Full quantum circuit generation

### 🔹 Approximate Division Variants
- **Truncated Comparison** - Compares only top k bits (up to 48% T-count reduction)
- **Early Stopping** - Computes first m quotient bits (up to 28% T-count savings)
- **Hybrid Execution** - Exact high bits + approximate low bits

### 🔹 Adaptive Circuit Selector
- Automatically selects optimal circuit based on:
  - Accuracy requirements
  - Resource constraints (qubits, gate depth, T-count)
  - Hardware noise profiles

### 🔹 Multiple Backends
- `classical_simulator.py` - Fast classical simulation
- `pennylane_simulator.py` - PennyLane quantum simulation
- `qiskit_impl/` - IBM Qiskit implementation for real hardware

### 🔹 Performance Analysis
- **T-count & T-depth** optimization
- **Scaling analysis** (4, 8, 12, 16 bits)
- **Pareto frontier** of accuracy vs. resource savings
- Hardware noise analysis

---

## 📦 Installation

### From GitHub (recommended)
```bash
pip install git+https://github.com/SouptikDSRS/quantum-division.git
