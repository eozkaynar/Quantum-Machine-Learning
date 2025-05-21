import torch
import pennylane as qml
from pennylane import compile

# Quantum circuit definition
n_qubits = 16

# Define quantum device (without noise)
# dev = qml.device("default.qubit",wires=n_qubits) # Without noise
# CUDA for fast process 
# dev = qml.device("lightning.gpu", wires=n_qubits, shots=None)   # A100 / RTX
dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)

# dev = qml.device("qiskit.aer",wires=n_qubits,noise_model = noise_model) # With noise 

# This decorator registers the function as a quantum node (QNode) using the given device.
# It enables the function to define a quantum circuit that runs on the specified quantum simulator or hardware backend.
# Register this function as a quantum circuit (QNode) on the defined device
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def qnode(inputs, **weights):
    # Layer 1: RX encoding + Rot gates
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
        qml.Rot(*weights[f"w0{str(i).zfill(2)}"], wires=i)

    # Entanglement Layer 1: CRX gates
    for i in range(n_qubits-1):
        qml.CRX(weights[f"x0{str(i).zfill(2)}"],wires=[i, i+1])
    qml.CRX(weights["x015"], wires=[15,0])

    # Layer 2: RX encoding + Rot gates
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
        qml.Rot(*weights[f"w1{str(i).zfill(2)}"], wires=i)
    
    # Entanglement Layer 2: CRX gates
    for i in range(n_qubits-1):
        qml.CRX(weights[f"x1{str(i).zfill(2)}"],wires=[i, i+1])
    qml.CRX(weights["x115"], wires=[15,0])

    # Measurement: return Pauli-Z expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
# Vectorized batches 
# batched_qnode   = map_batch(qnode, "inputs")                # <<< DEĞİŞTİ
# compiled_qnode  = qml.transforms.compile()(batched_qnode) 
# compiled_qnode = compile(qnode) 
# Weight shapes
weight_shapes = {
    **{f"w0{str(i).zfill(2)}": 3 for i in range(n_qubits)},
    **{f"w1{str(i).zfill(2)}": 3 for i in range(n_qubits)},
    **{f"x0{str(i).zfill(2)}": 1 for i in range(n_qubits)},
    **{f"x1{str(i).zfill(2)}": 1 for i in range(n_qubits)},
}

class QMLP(torch.nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.pool   = torch.nn.AvgPool2d(7)  # 28x28 → 4x4 → 16 features
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.fc1    = torch.nn.Linear(16,n_classes)

    def forward(self, x):
        b = x.size(0)
        # GPU kısım
        x = self.pool(x).view(b, n_qubits)
        # CPU kısım
        x_cpu = x.to("cpu")
        q_out = self.qlayer(x_cpu)
        # GPU kısım
        q_out = q_out.to(x.device)
        out   = self.fc1(q_out)
        return torch.nn.functional.log_softmax(out, dim=1)
