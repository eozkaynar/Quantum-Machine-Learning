import torch
import pennylane as qml

# Quantum circuit definition
n_qubits = 16

dev = qml.device("default.qubit",wires=n_qubits) # Without noise
# dev = qml.device("qiskit.aer",wires=n_qubits,noise_model = noise_model) # With noise 

@qml.qnode(dev)
def qnode(inputs, **weights):
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
        qml.Rot(*weights[f"w0{str(i).zfill(3)}"], wires=i)