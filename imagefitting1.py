import pennylane as qml
import torch
from torch import nn

dev = qml.device("default.qubit", wires = 2)
@qml.qnode(dev, interface = "torch")
def quantum_circuit(inputs, weights):
    qml.RY(inputs[0], wires=0)
    qml.RY(inputs[1], wires=1)

    qml.CNOT(wires=[0, 1])
    qml.RY(inputs[0] + inputs[1], wires=1)
    qml.CNOT(wires=[1, 0])

    # Variational layer
    qml.Rot(weights[0], weights[1], weights[2], wires=0)
    qml.Rot(weights[3], weights[4], weights[5], wires=1)
    qml.CNOT(wires=[0, 1])

    return qml.expval(qml.PauliZ(1))

class HybridImageFittingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(2,2)
        self.bn1 = nn.BatchNorm1d(2)
        self.qparams = nn.Parameter(0.1 * torch.randn(6))
        self.linear2 = nn.Linear(1,1)
    
    def quantum_layer(self,x):
        qout = torch.stack([quantum_circuit(sample, self.qparams) for sample in x])
        qout = qout.unsqueeze(1).float()
        return qout

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.quantum_layer(x)
        x = self.linear2(x)
        return x

#if __name__ == "__main__":
#    model = HybridImageFittingModel()
#    dummy_input = torch.rand(4,1,16,16).float()
#    out = model(dummy_input)
#    print("model output:\n", out)
