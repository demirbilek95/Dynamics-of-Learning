import numpy as np
import torch

class MLP(torch.nn.Module):
    def __init__(self, k, activation):
        super().__init__()
        self.k = k
        self.activation = activation
        self.flat = torch.nn.Flatten() # X comes in as a n x 1 x 56 x 28 -> we need n 1568-size vectors (or, a n x 1568 matrix). flatten does this

        if "NTK" in activation:
            self.layerEx = torch.nn.Linear(784*k,512)

        self.layer1 = torch.nn.Linear(784 * k, 512)
        self.layer2 = torch.nn.Linear(512, 2) 

        if "Gaussian" in activation:
            self.layer1.weight.data.normal_(0,np.sqrt(6 / float(784*k + 512)))

    def forward(self, X): 
        out = self.flat(X)
        if "linear" in self.activation:
            out = self.layer1(out)
        elif "NTK" in self.activation:
            out = torch.nn.functional.glu(torch.cat((self.layer1(out), self.layerEx(out)),1))
        else:
            out = self.layer1(out)
            out = torch.nn.functional.relu(out)

        out = self.layer2(out)
        return out
