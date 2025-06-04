import pennylane as qml
from pennylane import qnn
import torch
from torch import nn
import functools

dev = qml.device("default.qubit", wires = 6)
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    for layer in range(weights.shape[0]):
        for i in range(6):    
            qml.RY(inputs[i], wires=i)

        for i in range(5):
            qml.CNOT(wires=[i, i+1])
            qml.RY(inputs[i] + inputs[i+1], wires=i+1)
            qml.CNOT(wires=[i+1, i])

        # Variational layer
        for i in range(6):
            qml.Rot(weights[layer][i][0], weights[layer][i][1], weights[layer][i][2], wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(6)]

class HybridImageFittingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 3
        weight_shapes = {"weights": (self.n_layers, 6, 3)}
        
        self.q_layer = qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(2,6)
        self.bn1 = nn.BatchNorm1d(6)
        #self.qparams = nn.Parameter(torch.randn(6, 3))
        self.linear2 = nn.Linear(6,1)
    
    '''def quantum_layer(self,x):
        qout = []
        for sample in x:
            qres = quantum_circuit(sample, self.qparams)
            qout.append(qres)
        qout = torch.stack(qout).unsqueeze(1).float()
        return qout'''

    def forward(self, x):
        x = x.detach().requires_grad_(False)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.bn1(x)
        #x = self.q_layer(x)
        q_out = []
        for i in range(x.shape[0]):
            q_out.append(self.q_layer(x[i]))
        x = torch.stack(q_out)
        
        x = self.linear2(x)
        return x

#if __name__ == "__main__":
#    model = HybridImageFittingModel()
#    dummy_input = torch.rand(4,1,16,16).float()
#    out = model(dummy_input)
#    print("model output:\n", out)
