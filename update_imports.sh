#!/bin/bash

echo "Updating imports to use quantum_division package..."

# Update app.py
sed -i 's/from src\./from quantum_division\./g' app.py

# Update experiments/*.py
for file in experiments/*.py; do
    if [ -f "$file" ]; then
        sed -i 's/from src\./from quantum_division\./g' "$file"
        sed -i 's/import src\./import quantum_division\./g' "$file"
        echo "Updated: $file"
    fi
done

# Update qiskit_impl/*.py  
for file in qiskit_impl/*.py; do
    if [ -f "$file" ]; then
        sed -i 's/from src\./from quantum_division\./g' "$file"
        sed -i 's/import src\./import quantum_division\./g' "$file"
        echo "Updated: $file"
    fi
done

# Update docstrings in src/pennylane_simulator.py
if [ -f "src/pennylane_simulator.py" ]; then
    sed -i 's/>>> from src\./>>> from quantum_division\./g' src/pennylane_simulator.py
    echo "Updated docstrings in: src/pennylane_simulator.py"
fi

echo "Import updates complete!"
echo "NOTE: You'll need to install the package with 'pip install -e .' before running"
