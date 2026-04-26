from setuptools import setup, find_packages

setup(
    name="quantum-division",
    version="0.1.0",
    description="Quantum algorithms for division using Qiskit and PennyLane",
    author="SouptikDSRS",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21",
        "pennylane>=0.30",
        "qiskit>=0.44",
        "matplotlib>=3.5",
        "pandas>=1.5",
        "scipy>=1.9",
        # add any others from qiskit_impl/requirements.txt
    ],
    python_requires=">=3.8",
)