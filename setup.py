from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantum-division",
    version="0.1.0",
    author="SouptikDSRS",
    author_email="your.email@example.com",
    description="Quantum algorithms for division using Qiskit and PennyLane",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SouptikDSRS/quantum-division",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Quantum Computing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pennylane>=0.30.0",
        "qiskit>=0.44.0",
        "qiskit-ibm-runtime>=0.7.0",
        "matplotlib>=3.5.0",
        "pandas>=1.5.0",
        "scipy>=1.9.0",
        "plotly>=5.0.0",
        "flask>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "web": [
            "flask>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quantum-division=app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
